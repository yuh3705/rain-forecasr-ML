# Rainfall Forecast Course ML

Dự án này xây dựng pipeline học máy dự báo lượng mưa tại Hà Nội trong 6 giờ tiếp theo từ dữ liệu thời tiết theo giờ của Open-Meteo. Project tập trung vào bài toán hồi quy `target_rain_next_6h`, sau đó suy ra nhãn cảnh báo mưa `target_rain_alert_6h` bằng ngưỡng lượng mưa dự đoán.

Phiên bản hiện tại là một project course ML gọn nhẹ: không dùng Spark, Kafka, Airflow hay MLflow; các bước chính được thực hiện bằng Python, pandas, scikit-learn, notebook và Streamlit.

## Mục Tiêu Bài Toán

- Dự báo tổng lượng mưa trong 6 giờ tiếp theo tại Hà Nội.
- Tạo cảnh báo mưa nếu lượng mưa dự đoán lớn hơn hoặc bằng `1.0 mm`.
- So sánh các mô hình đơn, tuning hyperparameter, sau đó dùng OSEL (Optimized Stacking Ensemble Learning) làm mô hình ensemble chính.

## Luồng Xử Lý

```text
Open-Meteo
-> Tiền xử lý và tạo đặc trưng
-> Chia train/test theo thời gian
-> Tuning và đánh giá các mô hình đơn
-> Lựa chọn base learners
-> Huấn luyện OSEL
-> Dự báo lượng mưa 6 giờ tiếp theo
-> Tạo cảnh báo mưa
-> Đánh giá và lưu kết quả
```

Trong pipeline chính, OSEL được train trực tiếp trong `src/aq_course_ml/train_model.py`. Các notebook mô hình đơn trong `Notebooks/` được dùng cho giai đoạn thực nghiệm, tuning và so sánh mô hình.

## Cấu Trúc Thư Mục

```text
air-quality-course-ml/
|-- data/
|   |-- raw/
|   |   `-- rain_weather_hourly.csv
|   `-- processed/
|       `-- rainfall_features.csv
|-- Notebooks/
|   |-- Data_preprocessing.ipynb
|   |-- Raw_Data_Analysis.ipynb
|   |-- Exploratory_Data_Analysis.ipynb
|   |-- Decision_Tree.ipynb
|   |-- Random_Forest.ipynb
|   |-- Extra_Trees.ipynb
|   |-- Gradient_Boosting.ipynb
|   |-- AdaBoost.ipynb
|   |-- XGBoost.ipynb
|   |-- LightGBM.ipynb
|   |-- Ridge.ipynb
|   |-- OSEL.ipynb
|   `-- models/
|-- Results/
|   |-- *_metrics.json
|   |-- *_predictions.png
|   |-- model_comparison.csv
|   |-- rain_alert_model_comparison.csv
|   `-- feature_description.csv
|-- src/
|   `-- aq_course_ml/
|       |-- config.py
|       |-- crawl_data.py
|       |-- preprocess.py
|       |-- osel.py
|       |-- train_model.py
|       |-- predict_by_time.py
|       `-- utils.py
|-- app.py
|-- Dockerfile
|-- docker-compose.yml
`-- requirements.txt
```

## Cài Đặt

```bash
cd air-quality-course-ml
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu muốn chạy các notebook boosting, cần cài thêm các thư viện tương ứng nếu môi trường chưa có:

```bash
pip install xgboost lightgbm
```

## Chạy Pipeline Bằng Script

### 1. Thu thập dữ liệu

```bash
python -m src.aq_course_ml.crawl_data --start-date 2020-01-01 --end-date 2026-05-26
```

Kết quả:

```text
data/raw/rain_weather_hourly.csv
```

Script lấy dữ liệu thời tiết theo giờ của Hà Nội từ Open-Meteo, bao gồm lượng mưa, nhiệt độ, độ ẩm, áp suất, gió, mây và một số biến khí tượng khác.

### 2. Tiền xử lý và tạo đặc trưng

```bash
python -m src.aq_course_ml.preprocess
```

Kết quả:

```text
data/processed/rainfall_features.csv
```

Bước này thực hiện:

- Đọc dữ liệu raw.
- Chuyển `timestamp` về dạng thời gian.
- Sắp xếp dữ liệu theo thời gian.
- Nội suy và điền giá trị thiếu cho các cột số.
- Tạo đặc trưng thời gian, đặc trưng chu kỳ, lag features và rolling features.
- Tạo biến mục tiêu `target_rain_next_6h`.
- Tạo nhãn cảnh báo `target_rain_alert_6h`.

### 3. Huấn luyện OSEL

```bash
python -m src.aq_course_ml.train_model
```

Kết quả:

```text
Notebooks/models/osel_regressor.joblib
Results/metrics.json
Results/rainfall_predictions.png
```

Script `train_model.py` chia dữ liệu theo thứ tự thời gian, train OSEL regression model, dự báo lượng mưa trên tập test, suy ra cảnh báo mưa từ ngưỡng `1.0 mm`, sau đó lưu metrics và biểu đồ.

## Chia Train/Test Và TimeSeriesSplit

Với các notebook so sánh mô hình, dữ liệu được chia theo mốc thời gian:

- Train: từ `2020-01-01` đến `2025-12-31`.
- Test: từ `2026-01-01` đến thời điểm mới nhất trong dataset.

Tập test năm 2026 được giữ riêng để đánh giá cuối cùng. Trong quá trình tuning trên tập train, project dùng `TimeSeriesSplit` thay vì chia ngẫu nhiên. Cách này đảm bảo mỗi lần validation đều train trên dữ liệu quá khứ và validate trên một khoảng thời gian ở tương lai, tránh rò rỉ dữ liệu từ tương lai vào quá trình huấn luyện.

Ví dụ minh họa:

```text
Lần 1: train trên giai đoạn đầu của tập train -> validate trên giai đoạn ngay sau đó
Lần 2: mở rộng vùng train về phía trước -> validate trên giai đoạn tiếp theo
Lần 3: tiếp tục mở rộng train -> validate trên khoảng thời gian sau nữa
```

Trong các notebook mô hình đơn, `GridSearchCV` kết hợp `TimeSeriesSplit` được dùng để chọn hyperparameter tốt hơn trước khi train lại mô hình và đánh giá trên tập test 2026.

## OSEL Là Gì Trong Project Này?

OSEL là viết tắt của Optimized Stacking Ensemble Learning. Trong project này, OSEL được cài đặt trong `src/aq_course_ml/osel.py` và gồm hai ý chính:

- Stacking nhiều base learners để tạo dự báo mạnh hơn một mô hình đơn.
- Tuning cấu hình OSEL bằng `ParameterSampler` và `TimeSeriesSplit`.

Base learners trong implementation chính gồm:

- `RandomForestRegressor`
- `ExtraTreesRegressor`
- `GradientBoostingRegressor`
- `Ridge`

Meta learner là `Ridge`. Các meta-features được tạo bằng out-of-fold predictions theo thứ tự thời gian, giúp quá trình stacking phù hợp với bài toán time series.

## Các Mô Hình Đơn

Thư mục `Notebooks/` gồm các notebook train và tuning từng mô hình riêng:

| Notebook | Mô hình |
|---|---|
| `Decision_Tree.ipynb` | Decision Tree Regressor |
| `Random_Forest.ipynb` | Random Forest Regressor |
| `Extra_Trees.ipynb` | Extra Trees Regressor |
| `Gradient_Boosting.ipynb` | Gradient Boosting Regressor |
| `AdaBoost.ipynb` | AdaBoost Regressor |
| `XGBoost.ipynb` | XGBoost Regressor |
| `LightGBM.ipynb` | LightGBM Regressor |
| `Ridge.ipynb` | Ridge Regression |
| `OSEL.ipynb` | Optimized Stacking Ensemble Learning |

Kết quả so sánh được lưu trong `Results/`, gồm metrics JSON, prediction plots, `model_comparison.csv` và `rain_alert_model_comparison.csv`.

## Feature Set

| Nhóm | Feature | Ý nghĩa |
|---|---|---|
| Vị trí | `latitude` | Vĩ độ của Hà Nội |
| Vị trí | `longitude` | Kinh độ của Hà Nội |
| Mưa | `rain` | Lượng mưa theo giờ, đơn vị mm |
| Nhiệt độ | `temperature_2m` | Nhiệt độ không khí ở độ cao 2m |
| Độ ẩm | `relative_humidity_2m` | Độ ẩm tương đối ở độ cao 2m |
| Điểm sương | `dew_point_2m` | Nhiệt độ điểm sương ở độ cao 2m |
| Áp suất | `pressure_msl` | Áp suất mực nước biển |
| Áp suất | `surface_pressure` | Áp suất bề mặt |
| Gió | `wind_speed_10m` | Tốc độ gió ở độ cao 10m |
| Gió | `wind_direction_10m` | Hướng gió ở độ cao 10m |
| Gió | `wind_gusts_10m` | Gió giật ở độ cao 10m |
| Mây | `cloud_cover` | Tổng độ che phủ mây |
| Mây | `cloud_cover_low` | Độ che phủ mây tầng thấp |
| Mây | `cloud_cover_mid` | Độ che phủ mây tầng trung |
| Mây | `cloud_cover_high` | Độ che phủ mây tầng cao |
| Bốc thoát hơi | `et0_fao_evapotranspiration` | Lượng bốc thoát hơi tham chiếu FAO ET0 |
| Hơi nước | `vapour_pressure_deficit` | Độ thiếu hụt áp suất hơi nước |
| Thời gian | `hour` | Giờ trong ngày, từ 0 đến 23 |
| Thời gian | `month` | Tháng trong năm, từ 1 đến 12 |
| Chu kỳ thời gian | `hour_sin` | Mã hóa chu kỳ giờ bằng hàm sin |
| Chu kỳ thời gian | `hour_cos` | Mã hóa chu kỳ giờ bằng hàm cos |
| Chu kỳ thời gian | `month_sin` | Mã hóa chu kỳ tháng bằng hàm sin |
| Chu kỳ thời gian | `month_cos` | Mã hóa chu kỳ tháng bằng hàm cos |
| Lag feature | `rain_lag_1h` | Lượng mưa của 1 giờ trước |
| Lag feature | `rain_lag_3h` | Lượng mưa của 3 giờ trước |
| Rolling feature | `rain_roll_sum_6h` | Tổng lượng mưa trong 6 giờ trước, không tính giờ hiện tại |
| Rolling feature | `rain_roll_sum_12h` | Tổng lượng mưa trong 12 giờ trước, không tính giờ hiện tại |
| Lag feature | `temp_lag_1h` | Nhiệt độ của 1 giờ trước |
| Lag feature | `humidity_lag_1h` | Độ ẩm của 1 giờ trước |
| Lag feature | `dew_point_lag_1h` | Điểm sương của 1 giờ trước |
| Lag feature | `cloud_cover_lag_1h` | Độ che phủ mây của 1 giờ trước |

Cột `precipitation` không được dùng làm feature vì trong bộ dữ liệu này trùng với `rain`. Project chỉ giữ `rain` để tránh trùng lặp tín hiệu.

## Biến Mục Tiêu

| Biến | Loại | Ý nghĩa |
|---|---|---|
| `target_rain_next_6h` | Regression target | Tổng lượng mưa từ `t+1` đến `t+6` |
| `target_rain_alert_6h` | Binary target | Bằng 1 nếu `target_rain_next_6h >= 1.0 mm`, ngược lại bằng 0 |

## Metrics

Bài toán hồi quy được đánh giá bằng:

- MAE
- RMSE
- R2

Nhãn cảnh báo mưa suy ra từ dự báo hồi quy được đánh giá bằng:

- Accuracy
- F1-score
- Confusion matrix
- Classification report

## Chạy Streamlit App

Chạy trực tiếp:

```bash
streamlit run app.py
```

Sau đó mở:

```text
http://127.0.0.1:8501
```

Chạy bằng Docker:

```bash
docker compose up -d --build
```

Tắt Docker app:

```bash
docker compose down
```

Docker mount thư mục `Notebooks/models` từ máy local vào container, nên model đã train có thể được dùng trong app mà không cần commit file `.joblib` lên GitHub.

## Ghi Chú Báo Cáo

Khi trình bày trong báo cáo, có thể mô tả ngắn gọn:

```text
Dữ liệu thời tiết theo giờ được thu thập từ Open-Meteo cho khu vực Hà Nội. Sau đó dữ liệu được tiền xử lý, tạo các đặc trưng thời gian, lag và rolling để phục vụ bài toán dự báo mưa. Tập dữ liệu được chia theo thứ tự thời gian để tránh rò rỉ dữ liệu. Các mô hình đơn được tuning bằng GridSearchCV kết hợp TimeSeriesSplit, từ đó lựa chọn các base learners phù hợp cho OSEL. Mô hình OSEL dự báo tổng lượng mưa trong 6 giờ tiếp theo, sau đó hệ thống suy ra cảnh báo mưa dựa trên ngưỡng 1.0 mm và đánh giá bằng các chỉ số hồi quy và phân loại.
```
