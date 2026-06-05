# Notebooks

Notebook layout following the reference project:

- `Data_preprocessing.ipynb`
- `Raw_Data_Analysis.ipynb`
- `Exploratory_Data_Analysis.ipynb`
- `Decision_Tree.ipynb`
- `Random_Forest.ipynb`
- `Extra_Trees.ipynb`
- `Gradient_Boosting.ipynb`
- `AdaBoost.ipynb`
- `KNN.ipynb`
- `SVR.ipynb`
- `XGBoost.ipynb`
- `LightGBM.ipynb`
- `Ridge.ipynb`
- `OSEL.ipynb` contains the full Optimized Stacking Ensemble Learning implementation.

Each single-model notebook trains from 2020-01-01 through 2025-12-31, tunes hyperparameters with `GridSearchCV` + `TimeSeriesSplit`, and tests from 2026-01-01 through the latest crawled timestamp.
