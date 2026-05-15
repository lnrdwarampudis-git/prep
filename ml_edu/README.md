# 🎓 ML Educational Series — Supervised Learning Curriculum

## Overview
A complete, industry-grade supervised machine learning curriculum covering
binary classification and regression on 3 real-world datasets.

## Notebooks

| File | Dataset | Type | Cells | Key Topics |
|------|---------|------|-------|-----------|
| `00_Master_Comparison_All_Datasets.ipynb` | All 3 | Overview | 17 | Side-by-side comparison, decision guide |
| `01_Titanic_Complete_ML_Pipeline.ipynb` | Titanic | Binary Classif. | 61 | Full pipeline Phases 1–9 |
| `02_BreastCancer_Complete_ML_Pipeline.ipynb` | Breast Cancer | Medical Classif. | 40 | Feature selection, threshold tuning |
| `03_Housing_Complete_Regression_Pipeline.ipynb` | California Housing | Regression | 39 | Stacking, prediction intervals |

## Concept Coverage

### Titanic (Phases 1–9)
- Phase 1: EDA — MCAR/MAR/MNAR missing value taxonomy
- Phase 2: Feature Engineering — Title, FamilyBin, HasCabin, log1p, group imputation  
- Phase 3: Baseline Logistic Regression — coefficients, multicollinearity, ROC/CM
- Phase 4: 6-Algorithm Comparison — LR, DT, RF, SVM, KNN, XGBoost (5-fold CV)
- Phase 5: Learning Curves — bias-variance diagnosis
- Phase 6: Hyperparameter Tuning — coarse RandomizedSearch → fine GridSearch
- Phase 7: Production Pipelines — custom sklearn Transformer, .pkl export
- Phase 8: SHAP — global importance, beeswarm, local force plots
- Phase 9: MLflow — experiment tracking, run comparison

### Breast Cancer (Phases 1–9)
- High-dimensional EDA (PCA 2D projection, scree plot, correlation blocks)
- Feature Selection: SelectKBest (ANOVA-F + Mutual Info), RFECV, SHAP-based
- Medical metrics: Sensitivity, Specificity, PPV, NPV, AUC-PR
- Decision threshold tuning with asymmetric cost matrix (FN=$50k, FP=$500)
- Nested CV — unbiased evaluation for regulated systems
- RobustScaler vs StandardScaler rationale
- ColumnTransformer production pipeline

### California Housing (Phases 1–9)
- Regression EDA: target distribution, spatial maps, log-transform target
- Spatial feature engineering: K-Means on lat/lon → geo_cluster
- Engineered features: rooms_per_household, distance to cities, income_group
- Regression metrics: MAE, RMSE, MAPE, R², Adjusted R²
- Algorithms: LinearRegression, Ridge, Lasso, ElasticNet, DT, RF, XGBoost, LightGBM, KNN
- Residual analysis: 6-panel diagnostic (homoscedasticity, Q-Q, actual vs predicted)
- Model Stacking: XGB + LGBM + RF + Ridge → Ridge meta-learner
- Quantile Regression prediction intervals (5th/50th/95th percentile)
- MLflow logging

## Setup

```bash
pip install scikit-learn xgboost lightgbm shap mlflow imbalanced-learn \
            pandas numpy matplotlib seaborn scipy joblib
```

## Usage

```bash
cd notebooks
jupyter notebook
# Open 00_Master_Comparison_All_Datasets.ipynb to start
```

## Data
All datasets in `data/` directory:
- `titanic.csv` — 891 rows, 12 cols
- `breast_cancer.csv` — 569 rows, 31 cols (Wisconsin dataset features)
- `housing.csv` — 20,640 rows, 9 cols (California census blocks)
