# Rainfall Forecast Course ML

Project rut gon cho bai tap mon hoc may, tap trung vao du doan luong mua tai Ha Noi bang thuat toan OSEL (Optimized Stacking Ensemble Learning). Phien ban nay bo Spark, Kafka, Airflow va MLflow; chi giu pipeline hoc may cot loi:

1. Crawl du lieu thoi tiet theo gio tai thanh pho Ha Noi tu Open-Meteo.
2. Tien xu ly va tao dac trung thoi gian, lag, rolling.
3. Huan luyen model hoi quy OSEL du bao luong mua sau 6 gio, sau do suy ra canh bao mua tu gia tri du doan.

## Cau truc

```text
air-quality-course-ml/
|-- Data/
|   |-- raw/                  # Du lieu crawl tu API
|   `-- processed/            # Du lieu da tien xu ly
|-- Notebooks/
|   `-- models/               # Model da train
|-- Results/                  # Metrics va bieu do
|   `-- Compare_models.py     # So sanh baseline bang time split
|-- src/
|   `-- aq_course_ml/
|       |-- config.py         # Cau hinh duong dan, thanh pho, features
|       |-- crawl_data.py     # Crawl Open-Meteo
|       |-- preprocess.py     # Lam sach + feature engineering
|       |-- osel.py           # Cai dat Optimized Stacking Ensemble Learning
|       |-- train_model.py    # Train OSEL model
|       `-- utils.py
|-- app.py                    # Streamlit app
|-- run_pipeline.py           # Chay tat ca cac buoc
`-- requirements.txt
```

## Cai dat

```bash
cd air-quality-course-ml
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chay tung buoc

### 1. Crawl data

```bash
python -m src.aq_course_ml.crawl_data --start-date 2025-05-01 --end-date 2025-10-31
```

Ket qua: `Data/raw/rain_weather_hourly.csv`

### 2. Tien xu ly

```bash
python -m src.aq_course_ml.preprocess
```

Ket qua: `Data/processed/rainfall_features.csv`

### 3. Huan luyen model

```bash
python -m src.aq_course_ml.train_model
```

Ket qua:

- `Notebooks/models/rainfall_osel_regressor.joblib`: model hoi quy OSEL du bao tong luong mua tai Ha Noi trong 6 gio toi.
- `Results/metrics.json`: chi so danh gia.
- `Results/rainfall_predictions.png`: bieu do du bao.

## Chay toan bo pipeline

```bash
python run_pipeline.py --start-date 2025-05-01 --end-date 2025-10-31
```

For the model-comparison notebooks, crawl the full requested period:

```bash
python -m src.aq_course_ml.crawl_data --start-date 2020-01-01 --end-date 2026-05-26
python -m src.aq_course_ml.preprocess
python Results\Compare_models.py
```

Notebook split:

- Train: `2020-01-01` to `2025-12-31`
- Test: `2026-01-01` to the latest crawled timestamp

Each single-model notebook uses `GridSearchCV` with `TimeSeriesSplit` on the 2020-2025 train period, then refits the best estimator and evaluates on 2026. KNN and SVR use capped rows for the grid search because full kernel/distance search on 6 years of hourly data is slow. OSEL is kept in its own notebook because optimized time-series stacking is much heavier than single estimators.

Optional boosting notebooks require extra packages:

```bash
pip install xgboost lightgbm
```

## Chay Streamlit app

```bash
streamlit run app.py
```

## Bai toan hoc may

Project tap trung vao 1 bai toan chinh va 1 buoc suy luan canh bao:

- Regression: du bao `target_rain_next_6h`, tuc tong luong mua trong 6 gio toi.
- Rain alert classification: lay luong mua du doan tu model hoi quy, neu `predicted_rain_next_6h >= 1.0 mm` thi canh bao co mua dang ke, nguoc lai la khong.
- Thuat toan: OSEL ket hop nhieu base learners (`RandomForestRegressor`, `ExtraTreesRegressor`, `GradientBoostingRegressor`, `Ridge`) bang stacking va toi uu hyperparameter bang `TimeSeriesSplit`. Meta learner chi duoc train tu out-of-fold prediction theo dung thu tu thoi gian: moi fold train tren qua khu va predict tren khoang validation o tuong lai, tranh ro ri du lieu.

Feature set gom:

- Du lieu goc: `rain`, `precipitation`, nhiet do, do am, diem suong, ap suat, toc do/huong/gust gio, do che phu may theo tang, evapotranspiration, vapour pressure deficit.
- Dac trung thoi gian: gio trong ngay va thang trong nam.
- Dac trung chu ky: sin/cos theo gio va thang.
- Dac trung chuoi thoi gian: rain lag 1h, lag 3h, rolling sum 6h, rolling sum 12h, precipitation rolling sum 6h, lag nhiet do, do am, diem suong va cloud cover.

## Goi y viet bao cao

Ban co the trinh bay pipeline theo luong:

```text
Open-Meteo API -> raw CSV -> clean/feature engineering -> train/test split theo thoi gian -> train OSEL regression model -> threshold rainfall prediction -> metrics + bieu do
```

So voi repo goc, project nay don gian hon vi khong dung he sinh thai big data va chi tap trung vao Ha Noi, nhung van giu y tuong chinh: thu thap du lieu moi truong, xu ly thanh feature set, roi train model du bao.
