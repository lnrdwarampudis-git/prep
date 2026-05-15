"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     MASTER GUIDE: HANDLING MISSING VALUES ACROSS ALL ML DOMAINS            ║
║     Titanic | Heart Disease | Credit Fraud | Breast Cancer |               ║
║     Housing | Finance | Clinical — Full Implementation                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

LEARNING PATH:
  Level 1 → Understand WHY values are missing (MCAR / MAR / MNAR)
  Level 2 → Domain-specific strategy selection
  Level 3 → Implementation with sklearn Pipelines (production-ready)
  Level 4 → Advanced techniques (KNN, MICE/IterativeImputer, MissForest)
  Level 5 → Validation — did imputation hurt or help the model?
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches
import seaborn as sns

from sklearn.pipeline          import Pipeline
from sklearn.compose           import ColumnTransformer
from sklearn.preprocessing     import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.experimental      import enable_iterative_imputer   # noqa
from sklearn.impute            import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model      import LogisticRegression, BayesianRidge
from sklearn.model_selection   import cross_val_score, train_test_split
from sklearn.metrics           import (accuracy_score, roc_auc_score,
                                       classification_report, mean_squared_error)

from pathlib import Path

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (used in every plot for visual consistency)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "primary"   : "#2563EB",
    "secondary" : "#7C3AED",
    "success"   : "#059669",
    "warning"   : "#D97706",
    "danger"    : "#DC2626",
    "info"      : "#0891B2",
    "bg"        : "#0F172A",
    "card"      : "#1E293B",
    "text"      : "#F1F5F9",
    "muted"     : "#64748B",
}

plt.rcParams.update({
    "figure.facecolor" : PALETTE["bg"],
    "axes.facecolor"   : PALETTE["card"],
    "axes.edgecolor"   : PALETTE["muted"],
    "axes.labelcolor"  : PALETTE["text"],
    "xtick.color"      : PALETTE["text"],
    "ytick.color"      : PALETTE["text"],
    "text.color"       : PALETTE["text"],
    "grid.color"       : PALETTE["muted"],
    "grid.alpha"       : 0.3,
    "font.family"      : "monospace",
})


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def section_header(title: str, subtitle: str = ""):
    bar = "═" * 70
    print(f"\n{bar}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{bar}\n")


def missing_report(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Return a tidy DataFrame showing missing-value statistics."""
    miss  = df.isnull().sum()
    pct   = (miss / len(df) * 100).round(2)
    dtype = df.dtypes
    report = pd.DataFrame({
        "Missing Count" : miss,
        "Missing %"     : pct,
        "Dtype"         : dtype,
        "Unique Values" : df.nunique(),
    }).sort_values("Missing %", ascending=False)
    report = report[report["Missing Count"] > 0]
    if label:
        print(f"\n{'─'*55}")
        print(f"  Missing-Value Report — {label}")
        print(f"{'─'*55}")
        print(report.to_string())
        print(f"  Total rows : {len(df):,}")
        print(f"  Total cols : {df.shape[1]}")
    return report


def plot_missing_heatmap(df: pd.DataFrame, title: str, ax):
    """White = present, coloured = missing."""
    mask = df.isnull().astype(int)
    cols_with_missing = mask.columns[mask.sum() > 0]
    if len(cols_with_missing) == 0:
        ax.text(0.5, 0.5, "No Missing Values", ha="center", va="center",
                fontsize=14, color=PALETTE["success"])
        ax.set_title(title, color=PALETTE["text"], fontsize=12, fontweight="bold")
        return
    sub = mask[cols_with_missing].T
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "miss", [PALETTE["card"], PALETTE["danger"]])
    ax.imshow(sub, aspect="auto", cmap=cmap, interpolation="none")
    ax.set_yticks(range(len(cols_with_missing)))
    ax.set_yticklabels(cols_with_missing, fontsize=7)
    ax.set_xlabel("Row index", fontsize=8)
    ax.set_title(title, color=PALETTE["text"], fontsize=11, fontweight="bold")


def model_score(X, y, imputer, label="", task="classification"):
    """Fit imputer → model pipeline and return CV score."""
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    transformers = [("num", Pipeline([("imp", imputer),
                                       ("sc",  StandardScaler())]), num_cols)]
    if cat_cols:
        transformers.append(("cat",
                              Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                               unknown_value=-1))]),
                              cat_cols))

    pre = ColumnTransformer(transformers)
    if task == "classification":
        model = Pipeline([("pre", pre),
                          ("clf", RandomForestClassifier(n_estimators=100,
                                                         random_state=42))])
        scores = cross_val_score(model, X, y, cv=5,
                                 scoring="roc_auc", error_score="raise")
        metric = "ROC-AUC"
    else:
        model = Pipeline([("pre", pre),
                          ("reg", RandomForestRegressor(n_estimators=100,
                                                        random_state=42))])
        scores = cross_val_score(model, X, y, cv=5,
                                 scoring="neg_root_mean_squared_error")
        metric = "RMSE"

    mean, std = scores.mean(), scores.std()
    sign = 1 if task == "classification" else -1
    print(f"  {label:<35} {metric}: {sign*mean:.4f} ± {std:.4f}")
    return sign * mean


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET GENERATORS  (realistic synthetic data when real CSVs unavailable)
# ══════════════════════════════════════════════════════════════════════════════

def make_titanic():
    TITANIC_DATA_PATH   = Path('/Users/lakshmikalyani/claude-prep-ml/prep/practice/datasets/kaggle/titanic/train.csv')
    TITANIC_MODEL_DIR   = Path('/Users/lakshmikalyani/claude-prep-ml/prep/practice/models/titanic'); TITANIC_MODEL_DIR.mkdir(exist_ok=True)
    TITANIC_OUTPUT_DIR  = Path('/Users/lakshmikalyani/claude-prep-ml/prep/practice/outputs/titanic'); TITANIC_OUTPUT_DIR.mkdir(exist_ok=True)
    
    # n = 891
    # survived   = np.random.binomial(1, 0.38, n)
    # pclass     = np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
    # sex        = np.random.choice(["male", "female"], n, p=[0.65, 0.35])
    # age        = np.where(np.random.rand(n) < 0.20, np.nan,
    #                       np.clip(np.random.normal(30, 14, n), 0.5, 80))
    # sibsp      = np.random.poisson(0.52, n)
    # parch      = np.random.poisson(0.38, n)
    # fare       = np.where(np.random.rand(n) < 0.001, np.nan,
    #                       np.clip(np.random.exponential(32, n), 0, 512))
    # embarked   = np.random.choice(["S", "C", "Q"], n, p=[0.72, 0.19, 0.09]).astype(object)
    # embarked[np.random.rand(n) < 0.002] = None
    # cabin      = np.random.choice(["A", "B", "C", "D", "E", "F", "G"], n).astype(object)
    # cabin[np.random.rand(n) < 0.77]     = None
    # df = pd.DataFrame({"Survived": survived, "Pclass": pclass, "Sex": sex,
    #                    "Age": age, "SibSp": sibsp, "Parch": parch,
    #                    "Fare": fare, "Embarked": embarked, "Cabin": cabin})
    titanic_df=pd.read_csv(TITANIC_DATA_PATH)
    print(titanic_df)
    return titanic_df


def make_heart_disease():
    n = 303
    age        = np.random.randint(29, 77, n).astype(float)
    age[np.random.choice(n, 5, replace=False)] = np.nan
    sex        = np.random.choice([0, 1], n, p=[0.32, 0.68]).astype(float)
    cp         = np.random.choice([0,1,2,3], n).astype(float)
    trestbps   = np.clip(np.random.normal(131, 17, n), 94, 200).astype(float)
    trestbps[np.random.choice(n, 8, replace=False)] = np.nan
    chol       = np.clip(np.random.normal(246, 51, n), 126, 564).astype(float)
    chol[np.random.choice(n, 6, replace=False)] = np.nan
    fbs        = np.random.choice([0, 1], n, p=[0.85, 0.15]).astype(float)
    fbs[np.random.choice(n, 4, replace=False)] = np.nan
    restecg    = np.random.choice([0,1,2], n).astype(float)
    thalach    = np.clip(np.random.normal(149, 22, n), 71, 202).astype(float)
    thalach[np.random.choice(n, 7, replace=False)] = np.nan
    exang      = np.random.choice([0,1], n).astype(float)
    oldpeak    = np.clip(np.random.exponential(1.0, n), 0, 6.2).astype(float)
    oldpeak[np.random.choice(n, 10, replace=False)] = np.nan
    slope      = np.random.choice([0,1,2], n).astype(float)
    ca         = np.where(np.random.rand(n) < 0.013, np.nan,
                          np.random.choice([0,1,2,3], n)).astype(float)
    thal       = np.where(np.random.rand(n) < 0.013, np.nan,
                          np.random.choice([1,2,3], n)).astype(float)
    target     = np.random.binomial(1, 0.46, n)
    return pd.DataFrame({"age":age,"sex":sex,"cp":cp,"trestbps":trestbps,
                         "chol":chol,"fbs":fbs,"restecg":restecg,"thalach":thalach,
                         "exang":exang,"oldpeak":oldpeak,"slope":slope,
                         "ca":ca,"thal":thal,"target":target})


def make_credit_fraud():
    n = 284807
    # Highly imbalanced — 0.17 % fraud
    fraud = np.zeros(n, dtype=int)
    fraud_idx = np.random.choice(n, 492, replace=False)
    fraud[fraud_idx] = 1
    amount = np.where(np.random.rand(n) < 0.001, np.nan,
                      np.clip(np.random.exponential(88, n), 0, 25691))
    time_  = np.arange(n, dtype=float)
    time_[np.random.choice(n, int(n*0.0005), replace=False)] = np.nan
    # 28 PCA features (V1-V28)
    V = np.random.randn(n, 28)
    # inject 0.05 % missing per V column
    for j in range(28):
        idx = np.random.choice(n, int(n*0.0005), replace=False)
        V[idx, j] = np.nan
    cols = {f"V{i+1}": V[:, i] for i in range(28)}
    cols.update({"Time": time_, "Amount": amount, "Class": fraud})
    return pd.DataFrame(cols)


def make_breast_cancer_diagnosis():
    """Wisconsin Diagnostic Breast Cancer — binary classification."""
    from sklearn.datasets import load_breast_cancer
    bc = load_breast_cancer()
    df = pd.DataFrame(bc.data, columns=bc.feature_names)
    df["target"] = bc.target
    # Inject realistic missingness
    high_miss_cols  = bc.feature_names[:5]   # 15-25 % missing
    low_miss_cols   = bc.feature_names[5:15]  # 3-8 % missing
    for col in high_miss_cols:
        idx = np.random.choice(len(df), int(len(df)*np.random.uniform(0.15,0.25)), replace=False)
        df.loc[idx, col] = np.nan
    for col in low_miss_cols:
        idx = np.random.choice(len(df), int(len(df)*np.random.uniform(0.03,0.08)), replace=False)
        df.loc[idx, col] = np.nan
    return df


def make_breast_cancer_study():
    """Survival-study flavour — prognosis after treatment (regression target)."""
    n = 600
    age          = np.clip(np.random.normal(55, 12, n), 25, 85).astype(float)
    age[np.random.choice(n, 20, replace=False)] = np.nan
    tumor_size   = np.clip(np.random.exponential(2.5, n), 0.1, 12).astype(float)
    tumor_size[np.random.choice(n, 30, replace=False)] = np.nan
    lymph_nodes  = np.random.poisson(1.5, n).astype(float)
    lymph_nodes[np.random.choice(n, 25, replace=False)] = np.nan
    er_status    = np.random.choice(["Positive","Negative",None], n, p=[0.63,0.27,0.10])
    pr_status    = np.random.choice(["Positive","Negative",None], n, p=[0.572,0.308,0.12])
    her2_status  = np.random.choice(["Positive","Negative",None], n, p=[0.17,0.68,0.15])
    stage        = np.random.choice(["I","II","III","IV",None], n, p=[0.23,0.368,0.23,0.092,0.08])
    grade        = np.where(np.random.rand(n) < 0.07, np.nan,
                            np.random.choice([1,2,3], n, p=[0.2,0.5,0.3])).astype(float)
    ki67         = np.where(np.random.rand(n) < 0.30, np.nan,
                            np.clip(np.random.beta(2,5,n)*100, 1, 95))
    survival_months = np.clip(np.random.normal(72, 30, n), 1, 120)
    return pd.DataFrame({"age":age,"tumor_size_cm":tumor_size,
                         "lymph_nodes_positive":lymph_nodes,
                         "er_status":er_status,"pr_status":pr_status,
                         "her2_status":her2_status,"stage":stage,
                         "grade":grade,"ki67_percent":ki67,
                         "survival_months":survival_months})


def make_housing():
    n = 1460
    lot_area     = np.clip(np.random.lognormal(9.0, 0.5, n), 1300, 215000).astype(float)
    lot_area[np.random.choice(n, 5, replace=False)] = np.nan
    yr_built     = np.random.randint(1872, 2010, n).astype(float)
    overall_qual = np.random.choice(range(1,11), n).astype(float)
    overall_qual[np.random.choice(n, 10, replace=False)] = np.nan
    gr_liv_area  = np.clip(np.random.normal(1515, 525, n), 334, 5642).astype(float)
    garage_area  = np.where(np.random.rand(n) < 0.055, np.nan,
                            np.clip(np.random.normal(473, 213, n), 0, 1418))
    bsmt_sf      = np.where(np.random.rand(n) < 0.025, np.nan,
                            np.clip(np.random.normal(1057, 438, n), 0, 6110))
    pool_area    = np.where(np.random.rand(n) < 0.995, 0.0,
                            np.clip(np.random.normal(480, 150, n), 100, 800))
    _pool_choices = np.random.choice(["Ex","Gd","TA","Fa"], n)
    pool_qc      = np.where(pool_area == 0, None, _pool_choices).astype(object)
    alley        = np.random.choice(["Grvl","Pave",None], n,
                              p=[0.032, 0.032, 0.936])
    fence        = np.random.choice(["MnPrv","GdWo","GdPrv","MnWw",None], n,
                              p=[0.048, 0.048, 0.048, 0.049, 0.807])
    fireplace_qu = np.random.choice(["Ex","Gd","TA","Fa","Po",None], n,
                              p=[0.106, 0.106, 0.106, 0.106, 0.103, 0.473])
    mas_vnr_type = np.random.choice(["BrkFace","Stone","NoneType","BrkCmn",None], n,
                              p=[0.437, 0.089, 0.446, 0.022, 0.006])
    mas_vnr_area = np.where((mas_vnr_type == "NoneType") | (mas_vnr_type == None), 0.0, np.clip(np.random.normal(100, 125, n), 0, 1600))
    lot_frontage = np.where(np.random.rand(n) < 0.177, np.nan,
                            np.clip(np.random.normal(70, 24, n), 21, 313))
    sale_price   = np.clip(np.random.lognormal(11.9, 0.4, n), 34900, 755000)
    return pd.DataFrame({"LotFrontage":lot_frontage,"LotArea":lot_area,
                         "YearBuilt":yr_built,"OverallQual":overall_qual,
                         "GrLivArea":gr_liv_area,"GarageArea":garage_area,
                         "BsmtSF":bsmt_sf,"PoolArea":pool_area,
                         "PoolQC":pool_qc,"Alley":alley,"Fence":fence,
                         "FireplaceQu":fireplace_qu,"MasVnrType":mas_vnr_type,
                         "MasVnrArea":mas_vnr_area,"SalePrice":sale_price})


def make_finance():
    n = 5000
    credit_score   = np.clip(np.random.normal(680, 80, n), 300, 850).astype(float)
    credit_score[np.random.choice(n, 150, replace=False)] = np.nan
    annual_income  = np.clip(np.random.lognormal(10.8, 0.6, n), 15000, 500000).astype(float)
    annual_income[np.random.choice(n, 200, replace=False)] = np.nan
    debt_to_income = np.clip(np.random.beta(2, 5, n) * 100, 0, 80).astype(float)
    debt_to_income[np.random.choice(n, 180, replace=False)] = np.nan
    loan_amount    = np.clip(np.random.lognormal(10.0, 0.8, n), 1000, 500000).astype(float)
    loan_amount[np.random.choice(n, 50, replace=False)] = np.nan
    emp_length     = np.where(np.random.rand(n) < 0.06, np.nan,
                              np.clip(np.random.exponential(5, n), 0, 40))
    loan_purpose   = np.random.choice(["debt_consolidation","home_improvement",
                                              "business","medical","vacation",None], n,
                                             p=[0.441, 0.196, 0.147, 0.1176, 0.0784, 0.02])
    home_ownership = np.random.choice(["RENT","OWN","MORTGAGE",None], n,
                                            p=[0.396, 0.1485, 0.4455, 0.01])
    num_delinquencies = np.where(np.random.rand(n) < 0.08, np.nan,
                                 np.random.poisson(0.3, n)).astype(float)
    interest_rate  = np.clip(np.random.normal(13.5, 4.5, n), 5.5, 30.0).astype(float)
    interest_rate[np.random.choice(n, 80, replace=False)] = np.nan
    default        = np.random.binomial(1, 0.12, n)
    return pd.DataFrame({"credit_score":credit_score,"annual_income":annual_income,
                         "debt_to_income":debt_to_income,"loan_amount":loan_amount,
                         "emp_length_years":emp_length,"loan_purpose":loan_purpose,
                         "home_ownership":home_ownership,
                         "num_delinquencies":num_delinquencies,
                         "interest_rate":interest_rate,"default":default})


def make_clinical():
    n = 800
    patient_age     = np.clip(np.random.normal(58, 15, n), 18, 95).astype(float)
    patient_age[np.random.choice(n, 30, replace=False)] = np.nan
    bmi             = np.clip(np.random.normal(27, 5.5, n), 14, 55).astype(float)
    bmi[np.random.choice(n, 50, replace=False)] = np.nan
    systolic_bp     = np.clip(np.random.normal(125, 18, n), 80, 210).astype(float)
    systolic_bp[np.random.choice(n, 60, replace=False)] = np.nan
    diastolic_bp    = np.clip(np.random.normal(80, 12, n), 50, 130).astype(float)
    diastolic_bp[np.random.choice(n, 55, replace=False)] = np.nan
    glucose         = np.clip(np.random.normal(100, 25, n), 55, 350).astype(float)
    glucose[np.random.choice(n, 45, replace=False)] = np.nan
    hba1c           = np.where(np.random.rand(n) < 0.25, np.nan,
                               np.clip(np.random.normal(6.5, 1.5, n), 4.5, 14.0))
    creatinine      = np.clip(np.random.lognormal(0.1, 0.3, n), 0.4, 8.0).astype(float)
    creatinine[np.random.choice(n, 70, replace=False)] = np.nan
    wbc             = np.clip(np.random.normal(7.5, 2.5, n), 1.5, 30.0).astype(float)
    wbc[np.random.choice(n, 40, replace=False)] = np.nan
    hemoglobin      = np.clip(np.random.normal(13.5, 2.0, n), 6.0, 20.0).astype(float)
    hemoglobin[np.random.choice(n, 35, replace=False)] = np.nan
    smoking_status  = np.random.choice(["Never","Former","Current",None], n,
                                             p=[0.475, 0.285, 0.19, 0.05])
    icu_outcome     = np.random.binomial(1, 0.22, n)
    return pd.DataFrame({"patient_age":patient_age,"bmi":bmi,
                         "systolic_bp":systolic_bp,"diastolic_bp":diastolic_bp,
                         "glucose":glucose,"hba1c":hba1c,"creatinine":creatinine,
                         "wbc":wbc,"hemoglobin":hemoglobin,
                         "smoking_status":smoking_status,"icu_outcome":icu_outcome})


# ══════════════════════════════════════════════════════════════════════════════
#  CORE IMPUTATION STRATEGIES  (with explanations)
# ══════════════════════════════════════════════════════════════════════════════

def compare_imputers(X: pd.DataFrame, y: pd.Series,
                     task: str = "classification",
                     dataset_label: str = ""):
    """Run all five imputers and compare model performance."""
    section_header(f"  IMPUTER COMPARISON — {dataset_label}",
                   "Mean | Median | Mode | KNN | MICE vs Random-Forest baseline")

    imputers = {
        "Mean / Most-Frequent"  : SimpleImputer(strategy="mean"),
        "Median / Most-Frequent": SimpleImputer(strategy="median"),
        "KNN (k=5)"             : KNNImputer(n_neighbors=5),
        "MICE (IterativeImputer)": IterativeImputer(estimator=BayesianRidge(),
                                                     max_iter=10, random_state=42),
        "Constant (-999)"       : SimpleImputer(strategy="constant",
                                                fill_value=-999),
    }
    results = {}
    for name, imp in imputers.items():
        score = model_score(X, y, imp, label=name, task=task)
        results[name] = score
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 1 — TITANIC  (Passenger survival)
# ══════════════════════════════════════════════════════════════════════════════

def run_titanic():
    section_header("DATASET 1 — TITANIC",
                   "Domain: Transportation / Historical | Task: Binary Classification")

    df = make_titanic()
    missing_report(df, "Titanic Raw")

    print("""
  DOMAIN KNOWLEDGE (Titanic):
  ┌─────────────┬──────────────┬─────────────────────────────────────────────┐
  │ Column      │ Missing %    │ Strategy & Why                              │
  ├─────────────┼──────────────┼─────────────────────────────────────────────┤
  │ Age         │ ~20 %        │ KNN (correlated with Pclass, Sex, SibSp)   │
  │ Cabin       │ ~77 %        │ Create binary flag has_cabin; drop raw col  │
  │ Embarked    │ <1 %         │ Mode impute (dominant port = Southampton)   │
  │ Fare        │ <0.1 %       │ Median impute (right-skewed distribution)   │
  └─────────────┴──────────────┴─────────────────────────────────────────────┘
    """)

    # ── Step 1: Feature Engineering on raw missing info ──────────────────────
    df["has_cabin"]    = df["Cabin"].notna().astype(int)   # missingness = info!
    df["family_size"]  = df["SibSp"] + df["Parch"] + 1
    df["title"]        = "Mr"    # simplified — in real code, parse from Name

    # ── Step 2: Drop high-missingness columns ────────────────────────────────
    df.drop(columns=["Cabin"], inplace=True)

    # ── Step 3: Encode categoricals ──────────────────────────────────────────
    df["Sex_enc"]      = (df["Sex"] == "male").astype(int)
    df["Embarked_enc"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # ── Step 4: Build pipelines ───────────────────────────────────────────────
    feature_cols = ["Pclass","Sex_enc","Age","SibSp","Parch",
                    "Fare","Embarked_enc","has_cabin","family_size"]
    X = df[feature_cols]
    y = df["Survived"]

    print("  ▶ Pipeline 1: Simple Median Impute")
    pipe_median = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    scores = cross_val_score(pipe_median, X, y, cv=5, scoring="roc_auc")
    print(f"    ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    print("\n  ▶ Pipeline 2: KNN Impute (preserves correlations)")
    pipe_knn = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5)),
        ("scaler",  StandardScaler()),
        ("clf",     RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    scores_knn = cross_val_score(pipe_knn, X, y, cv=5, scoring="roc_auc")
    print(f"    ROC-AUC: {scores_knn.mean():.4f} ± {scores_knn.std():.4f}")

    print("\n  ▶ Pipeline 3: MICE (IterativeImputer) — best for MAR data")
    pipe_mice = Pipeline([
        ("imputer", IterativeImputer(max_iter=10, random_state=42)),
        ("scaler",  StandardScaler()),
        ("clf",     RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    scores_mice = cross_val_score(pipe_mice, X, y, cv=5, scoring="roc_auc")
    print(f"    ROC-AUC: {scores_mice.mean():.4f} ± {scores_mice.std():.4f}")

    print("""
  KEY LESSON (Titanic):
  • Cabin's missingness IS information (1st class cabins were recorded more)
  • Always create a binary flag before imputing or dropping
  • Age is MAR (depends on Pclass/Sex) → KNN/MICE outperform mean impute
    """)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 2 — HEART DISEASE
# ══════════════════════════════════════════════════════════════════════════════

def run_heart_disease():
    section_header("DATASET 2 — HEART DISEASE",
                   "Domain: Medical / Cardiology | Task: Binary Classification")

    df = make_heart_disease()
    missing_report(df, "Heart Disease Raw")

    print("""
  DOMAIN KNOWLEDGE (Heart Disease):
  ┌──────────────┬────────────┬──────────────────────────────────────────────┐
  │ Column       │ Missing %  │ Strategy & Why                               │
  ├──────────────┼────────────┼──────────────────────────────────────────────┤
  │ trestbps     │ ~2.6 %     │ Median (blood pressure is right-skewed)      │
  │ chol         │ ~2.0 %     │ Median (cholesterol is right-skewed)         │
  │ thalach      │ ~2.3 %     │ Median (heart rate, skewed)                  │
  │ oldpeak      │ ~3.3 %     │ KNN (correlated with thalach, exang)         │
  │ ca           │ ~1.3 %     │ KNN (ordinal: # vessels 0-3)                 │
  │ thal         │ ~1.3 %     │ Mode (categorical thalassemia type)          │
  │ fbs           │ ~1.3 %     │ Mode (binary: fasting blood sugar > 120)     │
  └──────────────┴────────────┴──────────────────────────────────────────────┘

  CLINICAL RULE: In healthcare, NEVER drop rows — each patient is irreplaceable.
  Always impute, and always document what was imputed vs observed.
    """)

    # ── Flag missingness before imputing ─────────────────────────────────────
    clinical_cols = ["trestbps", "chol", "thalach", "oldpeak", "ca"]
    for col in clinical_cols:
        df[f"{col}_was_missing"] = df[col].isna().astype(int)

    X = df.drop(columns=["target"])
    y = df["target"]

    # ── ColumnTransformer: different strategy per clinical variable ───────────
    knn_cols    = ["oldpeak", "ca", "thalach"]
    median_cols = ["trestbps", "chol", "age"]
    mode_cols   = ["thal", "fbs"]
    flag_cols   = [c for c in X.columns if c.endswith("_was_missing")]
    rest_cols   = [c for c in X.columns
                   if c not in knn_cols + median_cols + mode_cols + flag_cols]

    pre = ColumnTransformer([
        ("knn",    KNNImputer(n_neighbors=5),                   knn_cols),
        ("median", SimpleImputer(strategy="median"),            median_cols),
        ("mode",   SimpleImputer(strategy="most_frequent"),     mode_cols),
        ("flag",   "passthrough",                               flag_cols),
        ("rest",   SimpleImputer(strategy="median"),            rest_cols),
    ])

    pipe = Pipeline([
        ("pre", pre),
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=150, random_state=42)),
    ])

    scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    print(f"  ColumnTransformer Pipeline ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    print("""
  KEY LESSON (Heart Disease):
  • Use ColumnTransformer so EACH clinical variable gets the RIGHT strategy
  • Keep missingness flags as features — they may be prognostically meaningful
  • In healthcare: document imputation in model card / clinical validation report
    """)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 3 — CREDIT CARD FRAUD
# ══════════════════════════════════════════════════════════════════════════════

def run_credit_fraud():
    section_header("DATASET 3 — CREDIT CARD FRAUD",
                   "Domain: Finance / Security | Task: Anomaly / Imbalanced Classification")

    # Use a sample to keep runtime reasonable
    df = make_credit_fraud().sample(10000, random_state=42).reset_index(drop=True)
    missing_report(df, "Credit Fraud (10k sample)")

    print("""
  DOMAIN KNOWLEDGE (Credit Fraud):
  ┌──────────────────────┬───────────┬──────────────────────────────────────┐
  │ Column               │ Missing % │ Strategy & Why                       │
  ├──────────────────────┼───────────┼──────────────────────────────────────┤
  │ Amount               │ ~0.1 %    │ Median (right-skewed, log-transform) │
  │ Time                 │ ~0.05 %   │ Forward-fill (time-series ordering)  │
  │ V1–V28 (PCA features)│ ~0.05%ea  │ Zero (PCA components center at 0)   │
  └──────────────────────┴───────────┴──────────────────────────────────────┘

  FRAUD-SPECIFIC RULES:
  • Imputation must be done SEPARATELY on train/test to prevent leakage
  • For PCA features: missing → 0 is mathematically justified (mean of PCA = 0)
  • Amount: log1p transform THEN median impute (or impute then log)
  • Time gaps in transaction streams → forward fill within user session
  • With 0.17% fraud rate: imputation error in fraud rows matters more!
    """)

    # Sort by time (transaction stream)
    df = df.sort_values("Time").reset_index(drop=True)

    # Forward fill Time (stream continuity)
    df["Time"] = df["Time"].fillna(method="ffill")

    # Log-transform + median for Amount
    df["Amount_log"] = np.log1p(df["Amount"].fillna(df["Amount"].median()))

    # PCA features: fill with 0
    v_cols = [f"V{i}" for i in range(1, 29)]
    df[v_cols] = df[v_cols].fillna(0)

    X = df[v_cols + ["Amount_log"]].copy()
    y = df["Class"]

    print(f"  Fraud rate in sample: {y.mean()*100:.2f}%")

    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                       random_state=42)),
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    print(f"  Pipeline ROC-AUC (after imputation): {scores.mean():.4f} ± {scores.std():.4f}")

    print("""
  KEY LESSON (Credit Fraud):
  • Impute PCA features with 0 (not mean/median) — mathematical correctness
  • Time-ordered data: forward-fill respects causality (no future info)
  • Always fit imputer on TRAIN only, transform BOTH train and test
  • Missing Amount could indicate a data-pipeline issue → alert + fill median
    """)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 4 — BREAST CANCER (Diagnosis — Classification)
# ══════════════════════════════════════════════════════════════════════════════

def run_breast_cancer_diagnosis():
    section_header("DATASET 4A — BREAST CANCER DIAGNOSIS",
                   "Domain: Oncology / Radiology | Task: Malignant vs Benign")

    df = make_breast_cancer_diagnosis()
    missing_report(df, "Breast Cancer Diagnosis")

    print("""
  DOMAIN KNOWLEDGE (Breast Cancer Diagnosis):
  ┌──────────────────────────────┬───────────┬──────────────────────────────┐
  │ Feature Group                │ Missing % │ Strategy                     │
  ├──────────────────────────────┼───────────┼──────────────────────────────┤
  │ mean_* (first 5 features)   │ 15-25 %   │ MICE — strong inter-feature  │
  │                              │           │ correlations (nuclear props) │
  │ Other radius/texture/etc    │ 3-8 %     │ KNN (n_neighbors=7)          │
  └──────────────────────────────┴───────────┴──────────────────────────────┘

  ONCOLOGY RULE: Features like mean_radius and mean_area are geometrically
  correlated (area ~ π*r²). MICE leverages this — DO NOT use mean/median
  impute which ignores inter-feature structure.
    """)

    X = df.drop(columns=["target"])
    y = df["target"]
    feat_names = X.columns.tolist()

    high_miss = feat_names[:5]
    low_miss  = [c for c in feat_names if c not in high_miss]

    pre = ColumnTransformer([
        ("mice", IterativeImputer(max_iter=10, random_state=42), high_miss),
        ("knn",  KNNImputer(n_neighbors=7),                      low_miss),
    ])

    pipe = Pipeline([
        ("pre", pre),
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=150, random_state=42)),
    ])

    scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    print(f"  MICE + KNN Pipeline ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    print("""
  KEY LESSON (Breast Cancer Diagnosis):
  • MICE is the gold standard when features are highly correlated
  • Separate columns by missingness level — heavy miss cols need MICE, light miss KNN
  • Never use mean impute on oncology imaging features
    """)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 5 — BREAST CANCER (Survival Study — Regression)
# ══════════════════════════════════════════════════════════════════════════════

def run_breast_cancer_study():
    section_header("DATASET 4B — BREAST CANCER SURVIVAL STUDY",
                   "Domain: Oncology / Epidemiology | Task: Survival Regression (months)")

    df = make_breast_cancer_study()
    missing_report(df, "Breast Cancer Study")

    print("""
  DOMAIN KNOWLEDGE (Survival Study):
  ┌──────────────────────┬───────────┬──────────────────────────────────────┐
  │ Column               │ Missing % │ Strategy                             │
  ├──────────────────────┼───────────┼──────────────────────────────────────┤
  │ ki67_percent         │ ~30 %     │ Separate indicator column + MICE     │
  │ her2_status          │ ~15 %     │ "Unknown" category (clinically valid)│
  │ pr_status            │ ~12 %     │ "Unknown" category                   │
  │ er_status            │ ~10 %     │ "Unknown" category                   │
  │ stage                │ ~8 %      │ Mode impute (ordinal)                │
  │ grade                │ ~7 %      │ Median impute (ordinal 1-3)          │
  │ lymph_nodes_positive │ ~4 %      │ KNN impute                           │
  │ tumor_size_cm        │ ~5 %      │ Median impute                        │
  │ age                  │ ~3.3 %    │ Median impute                        │
  └──────────────────────┴───────────┴──────────────────────────────────────┘

  CLINICAL RULE: ki67 >30% missing is MNAR (not tested in lower-resource
  settings). Add indicator column — the missingness itself carries prognostic
  information (resource disparity, clinical decision).
    """)

    # ki67 indicator (MNAR — missingness is clinically informative)
    df["ki67_missing"] = df["ki67_percent"].isna().astype(int)
    df["ki67_percent"] = df["ki67_percent"].fillna(df["ki67_percent"].median())

    # Ordinal receptor status as "Unknown" category
    for col in ["er_status", "pr_status", "her2_status"]:
        df[col] = df[col].fillna("Unknown")

    # Stage: mode fill
    df["stage"] = df["stage"].fillna(df["stage"].mode()[0])

    # Encode categoricals
    for col in ["er_status", "pr_status", "her2_status", "stage"]:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Grade: median
    df["grade"] = df["grade"].fillna(df["grade"].median())

    # Remaining numerics: KNN
    num_cols = ["age","tumor_size_cm","lymph_nodes_positive"]
    knn_imp  = KNNImputer(n_neighbors=5)
    df[num_cols] = knn_imp.fit_transform(df[num_cols])

    X = df.drop(columns=["survival_months"])
    y = df["survival_months"]

    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("reg", RandomForestRegressor(n_estimators=150, random_state=42)),
    ])
    scores = cross_val_score(pipe, X, y, cv=5,
                             scoring="neg_root_mean_squared_error")
    print(f"  Pipeline RMSE: {-scores.mean():.2f} ± {scores.std():.2f} months")

    print("""
  KEY LESSON (Survival Study):
  • MNAR data → ADD indicator column + impute (never just impute silently)
  • "Unknown" is a VALID clinical category for hormone receptor status
  • Stage/grade are ordinal — median impute is appropriate
    """)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 6 — HOUSING  (Ames / Kaggle style)
# ══════════════════════════════════════════════════════════════════════════════

def run_housing():
    section_header("DATASET 5 — HOUSING PRICES",
                   "Domain: Real Estate | Task: Regression (SalePrice)")

    df = make_housing()
    missing_report(df, "Housing Raw")

    print("""
  DOMAIN KNOWLEDGE (Housing):
  ┌──────────────────┬───────────┬──────────────────────────────────────────┐
  │ Column           │ Missing % │ Real Meaning & Strategy                  │
  ├──────────────────┼───────────┼──────────────────────────────────────────┤
  │ PoolQC           │ ~99.5 %   │ No pool → fill "No Pool" (not missing!)  │
  │ Alley            │ ~93.6 %   │ No alley access → fill "No Alley"       │
  │ Fence            │ ~80.7 %   │ No fence → fill "No Fence"              │
  │ FireplaceQu      │ ~47.3 %   │ No fireplace → fill "No Fireplace"      │
  │ LotFrontage      │ ~17.7 %   │ MAR: fill by neighbourhood median       │
  │ MasVnrType       │ ~0.6 %    │ Mode impute                             │
  │ MasVnrArea       │ linked    │ Tied to MasVnrType → fill 0 if None     │
  │ GarageArea       │ ~5.5 %    │ No garage → fill 0                      │
  │ BsmtSF           │ ~2.5 %    │ No basement → fill 0                    │
  └──────────────────┴───────────┴──────────────────────────────────────────┘

  HOUSING RULE: Most "missing" values are STRUCTURAL — the feature doesn't
  exist for that property. NaN ≠ unknown; NaN = absence.
    """)

    # ── Domain-driven fills (NOT statistical imputation) ──────────────────────
    df["PoolQC"]      = df["PoolQC"].fillna("No Pool")
    df["Alley"]       = df["Alley"].fillna("No Alley")
    df["Fence"]       = df["Fence"].fillna("No Fence")
    df["FireplaceQu"] = df["FireplaceQu"].fillna("No Fireplace")
    df["GarageArea"]  = df["GarageArea"].fillna(0)
    df["BsmtSF"]      = df["BsmtSF"].fillna(0)
    df["MasVnrArea"]  = df["MasVnrArea"].fillna(0)
    df["MasVnrType"]  = df["MasVnrType"].fillna(df["MasVnrType"].mode()[0])

    # LotFrontage: fill by median (simulate neighbourhood grouping)
    df["LotFrontage"] = df["LotFrontage"].fillna(df["LotFrontage"].median())

    # ── Encode categoricals ───────────────────────────────────────────────────
    cat_cols = ["PoolQC","Alley","Fence","FireplaceQu","MasVnrType"]
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop(columns=["SalePrice"])
    y = np.log1p(df["SalePrice"])   # log-transform target

    # Remaining numeric NaN (OverallQual etc.)
    X = X.fillna(X.median(numeric_only=True))

    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("reg", RandomForestRegressor(n_estimators=150, random_state=42)),
    ])
    scores = cross_val_score(pipe, X, y, cv=5,
                             scoring="neg_root_mean_squared_error")
    print(f"  Pipeline Log-RMSE: {-scores.mean():.4f} ± {scores.std():.4f}")

    print("""
  KEY LESSON (Housing):
  • STRUCTURAL missing ≠ RANDOM missing — always interrogate the data dictionary
  • NaN in PoolQC means NO POOL, not unknown pool quality → fill "No Pool"
  • LotFrontage is the only genuinely random missing → neighbourhood median
  • Log-transform the target (SalePrice) before modelling
    """)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 7 — FINANCE (Loan Default)
# ══════════════════════════════════════════════════════════════════════════════

def run_finance():
    section_header("DATASET 6 — FINANCE / LOAN DEFAULT",
                   "Domain: Banking / Credit Risk | Task: Binary Classification")

    df = make_finance()
    missing_report(df, "Finance / Loan Dataset")

    print("""
  DOMAIN KNOWLEDGE (Finance):
  ┌──────────────────────┬───────────┬──────────────────────────────────────┐
  │ Column               │ Missing % │ Strategy & Regulatory Consideration  │
  ├──────────────────────┼───────────┼──────────────────────────────────────┤
  │ credit_score         │ ~3 %      │ Median (not mean — right-skewed)     │
  │                      │           │ + flag: no_credit_score              │
  │ annual_income        │ ~4 %      │ Median + flag (MNAR: refused answer) │
  │ debt_to_income       │ ~3.6 %    │ Median (bounded 0-100)               │
  │ emp_length_years     │ ~6 %      │ 0 (unemployed) + flag                │
  │ interest_rate        │ ~1.6 %    │ Median per loan_purpose group        │
  │ num_delinquencies    │ ~8 %      │ 0 (no record = no delinquency)       │
  │ loan_purpose         │ ~2 %      │ "Unknown" category                   │
  │ home_ownership       │ ~1 %      │ Mode impute                          │
  └──────────────────────┴───────────┴──────────────────────────────────────┘

  REGULATORY RULE (Basel III / Fair Lending):
  • You MUST document every imputation decision
  • Imputing income with mean can introduce proxy discrimination
  • Missing credit score → must be modelled as distinct risk segment
    """)

    # ── MNAR flags (missing income/credit = different risk segment) ───────────
    df["no_credit_score"]  = df["credit_score"].isna().astype(int)
    df["income_not_given"] = df["annual_income"].isna().astype(int)
    df["emp_missing"]      = df["emp_length_years"].isna().astype(int)

    # emp_length: 0 if missing (unemployed interpretation)
    df["emp_length_years"] = df["emp_length_years"].fillna(0)

    # num_delinquencies: 0 if no record
    df["num_delinquencies"] = df["num_delinquencies"].fillna(0)

    # loan_purpose / home_ownership: categorical fills
    df["loan_purpose"]  = df["loan_purpose"].fillna("Unknown")
    df["home_ownership"]= df["home_ownership"].fillna(df["home_ownership"].mode()[0])

    # Encode categoricals
    for col in ["loan_purpose", "home_ownership"]:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Credit score / income / DTI: median
    for col in ["credit_score","annual_income","debt_to_income","interest_rate"]:
        df[col] = df[col].fillna(df[col].median())

    X = df.drop(columns=["default"])
    X = X.fillna(X.median(numeric_only=True))
    y = df["default"]

    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    print(f"  GBM Pipeline ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    print("""
  KEY LESSON (Finance):
  • Missing financial data is often MNAR (people withhold unfavourable info)
  • Create RISK SEGMENTS from missingness, not just flags
  • Regulatory compliance: document imputation decisions formally
  • Group-based imputation (interest rate by loan purpose) > global median
    """)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET 8 — CLINICAL  (ICU / Hospital)
# ══════════════════════════════════════════════════════════════════════════════

def run_clinical():
    section_header("DATASET 7 — CLINICAL / ICU DATA",
                   "Domain: Hospital / Critical Care | Task: ICU Outcome Classification")

    df = make_clinical()
    missing_report(df, "Clinical / ICU Data")

    print("""
  DOMAIN KNOWLEDGE (Clinical / ICU):
  ┌──────────────────┬───────────┬──────────────────────────────────────────┐
  │ Column           │ Missing % │ Strategy                                 │
  ├──────────────────┼───────────┼──────────────────────────────────────────┤
  │ hba1c            │ ~25 %     │ MNAR: only tested if diabetic suspected. │
  │                  │           │ Add flag + median impute                 │
  │ creatinine       │ ~8.8 %    │ KNN (correlated with age, bmi)           │
  │ systolic_bp      │ ~7.5 %    │ KNN (correlated with diastolic_bp)       │
  │ diastolic_bp     │ ~6.9 %    │ KNN (paired with systolic)               │
  │ bmi              │ ~6.25 %   │ KNN (correlated with age)                │
  │ glucose          │ ~5.6 %    │ Median (right-skewed)                    │
  │ wbc              │ ~5 %      │ Median (white blood cell count)          │
  │ hemoglobin       │ ~4.4 %    │ KNN (sex-stratified ideally)             │
  │ smoking_status   │ ~5 %      │ "Unknown" → valid clinical category      │
  │ patient_age      │ ~3.75 %   │ Median                                   │
  └──────────────────┴───────────┴──────────────────────────────────────────┘

  ICU RULE: Paired vitals (systolic/diastolic, WBC/hemoglobin) should be
  imputed TOGETHER using multivariate methods (KNN/MICE) to preserve
  physiological relationships. NEVER impute systolic independently of diastolic.
    """)

    # HbA1c is MNAR — tested only if diabetes suspected
    df["hba1c_tested"]  = df["hba1c"].notna().astype(int)
    df["hba1c"]         = df["hba1c"].fillna(df["hba1c"].median())

    # Smoking: "Unknown" category
    df["smoking_status"] = df["smoking_status"].fillna("Unknown")
    df["smoking_enc"]    = LabelEncoder().fit_transform(df["smoking_status"])

    # KNN for paired vitals and correlated labs
    knn_cols = ["systolic_bp","diastolic_bp","bmi","creatinine","hemoglobin"]
    knn_imp  = KNNImputer(n_neighbors=7)
    df[knn_cols] = knn_imp.fit_transform(df[knn_cols])

    # Median for remaining labs
    for col in ["glucose","wbc","patient_age"]:
        df[col] = df[col].fillna(df[col].median())

    X = df.drop(columns=["icu_outcome","smoking_status"])
    y = df["icu_outcome"]

    pipe = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                       random_state=42)),
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    print(f"  Pipeline ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    print("""
  KEY LESSON (Clinical):
  • Paired physiological variables → KNN/MICE together (not independently!)
  • HbA1c MNAR: tested when clinician suspects diabetes — missingness = clinical info
  • ICU data often has time-series structure → consider LOCF (last obs carried fwd)
  • Class imbalance (ICU mortality ~20%) → use class_weight='balanced'
    """)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MASTER COMPARISON CHART  — all datasets side by side
# ══════════════════════════════════════════════════════════════════════════════

def plot_master_summary(datasets: dict):
    fig = plt.figure(figsize=(22, 18), facecolor=PALETTE["bg"])
    fig.suptitle("MISSING VALUE STRATEGIES — ALL DOMAINS",
                 fontsize=18, fontweight="bold", color=PALETTE["text"],
                 y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.55, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.04)

    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]

    titles = list(datasets.keys())
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["success"],
              PALETTE["warning"], PALETTE["danger"], PALETTE["info"],
              PALETTE["primary"], PALETTE["secondary"], PALETTE["muted"]]

    for i, (name, df) in enumerate(datasets.items()):
        if i >= 9:
            break
        ax = axes[i]
        miss_pct = (df.isnull().sum() / len(df) * 100)
        miss_pct = miss_pct[miss_pct > 0].sort_values(ascending=True)
        if len(miss_pct) == 0:
            ax.text(0.5, 0.5, "Fully Imputed", ha="center", va="center",
                    color=PALETTE["success"], fontsize=10)
        else:
            bars = ax.barh(miss_pct.index, miss_pct.values,
                           color=colors[i], alpha=0.85, height=0.65)
            ax.axvline(5,  color=PALETTE["warning"], lw=1, ls="--", alpha=0.6)
            ax.axvline(30, color=PALETTE["danger"],  lw=1, ls="--", alpha=0.6)
            for bar, val in zip(bars, miss_pct.values):
                ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                        f"{val:.1f}%", va="center", fontsize=6,
                        color=PALETTE["text"])
        ax.set_title(name, color=PALETTE["text"], fontsize=9,
                     fontweight="bold", pad=5)
        ax.set_xlabel("% Missing", fontsize=7, color=PALETTE["muted"])
        ax.tick_params(labelsize=6)
        ax.set_xlim(0, max(miss_pct.max() * 1.25 if len(miss_pct) > 0 else 10, 10))

    plt.savefig("/Users/lakshmikalyani/claude-prep-ml/prep/practice/datasets/kaggle/outputs/01_missing_patterns_all_domains.png",
                dpi=140, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("\n  ✅ Saved: 01_missing_patterns_all_domains.png")


# ══════════════════════════════════════════════════════════════════════════════
#  STRATEGY DECISION FRAMEWORK  (decision tree diagram as text + chart)
# ══════════════════════════════════════════════════════════════════════════════

def plot_decision_framework():
    fig, ax = plt.subplots(figsize=(18, 11), facecolor=PALETTE["bg"])
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.axis("off")
    ax.set_facecolor(PALETTE["bg"])

    fig.suptitle("MISSING VALUE STRATEGY DECISION FRAMEWORK",
                 fontsize=15, fontweight="bold", color=PALETTE["text"], y=0.98)

    def box(x, y, txt, color, fontsize=8, width=2.6, height=0.7):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x - width/2, y - height/2), width, height,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor=PALETTE["text"],
                                        linewidth=0.8, alpha=0.92))
        ax.text(x, y, txt, ha="center", va="center",
                fontsize=fontsize, color=PALETTE["text"], fontweight="bold",
                wrap=True, multialignment="center")

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["muted"], lw=1.2))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx+0.15, my, label, fontsize=7, color=PALETTE["warning"])

    # Level 0
    box(9, 10.2, "Missing Value Detected", PALETTE["secondary"], fontsize=10, width=4)
    arrow(9, 9.85, 9, 9.35)

    # Level 1: Missing %
    box(9, 9.0, "> 60% missing?", PALETTE["card"], fontsize=9, width=3.5)
    arrow(9, 8.65, 9, 8.15, "NO")
    arrow(9, 8.65, 14.5, 8.15, "YES")
    box(14.5, 7.8, "DROP column\n(usually)", PALETTE["danger"], fontsize=8)

    # Level 2: MCAR/MAR/MNAR
    box(9, 7.8, "What TYPE of missing?", PALETTE["card"], fontsize=9, width=4)
    arrow(9, 7.45, 4, 6.8, "MCAR\n(random)")
    arrow(9, 7.45, 9, 6.8, "MAR\n(depends on\nother cols)")
    arrow(9, 7.45, 14, 6.8, "MNAR\n(systematic)")

    # Level 3 branches
    box(4,  6.5, "Simple Impute OK\n(mean/median/mode)", PALETTE["success"], fontsize=7.5)
    box(9,  6.5, "Multivariate Impute\n(KNN / MICE)", PALETTE["primary"],   fontsize=7.5)
    box(14, 6.5, "Flag + Impute\n(add indicator col)", PALETTE["warning"],   fontsize=7.5)

    arrow(4,  6.15, 4,  5.5)
    arrow(9,  6.15, 9,  5.5)
    arrow(14, 6.15, 14, 5.5)

    # Level 4: data type
    box(4,  5.2, "Numeric?\nMedian   Skewed\nMean     Symmetric", PALETTE["card"], fontsize=7)
    box(9,  5.2, "Numeric?\nKNN (correlated)\nMICE (high miss %)", PALETTE["card"], fontsize=7)
    box(14, 5.2, "Add is_missing col\nthen apply\nMAR strategy", PALETTE["card"], fontsize=7)

    # Domain examples
    examples = [
        (2,  3.8, "Titanic Fare\nCredit Fraud V*\nHousing BsmtSF",  PALETTE["info"]),
        (6,  3.8, "Titanic Age\nHeart Oldpeak\nClinical BP pair",   PALETTE["primary"]),
        (10, 3.8, "Finance Income\nClinical HbA1c\nCancer Ki67",    PALETTE["warning"]),
        (14, 3.8, "Housing PoolQC\nHousing Alley\n(structural NaN)", PALETTE["success"]),
    ]
    for ex in examples:
        box(ex[0], ex[1], ex[2], ex[3], fontsize=6.5, width=3.5, height=1.0)

    arrow(4,  4.85, 2,  4.35)
    arrow(9,  4.85, 6,  4.35)
    arrow(14, 4.85, 10, 4.35)

    # Domain rule box
    box(14, 3.8, "Domain Rule First!\nStructural absence\n→ fill 'None'/0",
        PALETTE["success"], fontsize=6.5, width=3.5, height=1.0)

    # Footer rule
    ax.text(9, 2.4,
            "GOLDEN RULE: Fit imputer ONLY on TRAIN data → transform BOTH train and test",
            ha="center", fontsize=9, color=PALETTE["danger"],
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor=PALETTE["card"],
                      edgecolor=PALETTE["danger"], linewidth=1.5))

    ax.text(9, 1.6,
            "MCAR = Missing Completely At Random    "
            "MAR = Missing At Random    "
            "MNAR = Missing Not At Random",
            ha="center", fontsize=7.5, color=PALETTE["muted"])

    plt.savefig("/Users/lakshmikalyani/claude-prep-ml/prep/practice/datasets/kaggle/outputs/02_decision_framework.png",
                dpi=140, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("  ✅ Saved: 02_decision_framework.png")


# ══════════════════════════════════════════════════════════════════════════════
#  IMPUTER PERFORMANCE COMPARISON CHART
# ══════════════════════════════════════════════════════════════════════════════

def plot_imputer_comparison():
    """Visual bar chart: 5 imputers × 3 datasets."""
    section_header("IMPUTER BENCHMARKING",
                   "Which imputer wins on which dataset type?")

    datasets_bench = {
        "Titanic (Classification)" : (make_titanic,       "classification"),
        "Heart Disease (Classif.)" : (make_heart_disease, "classification"),
    }

    imputer_names = ["Mean", "Median", "KNN-5", "MICE", "Constant"]
    imputer_objs  = [
        SimpleImputer(strategy="mean"),
        SimpleImputer(strategy="median"),
        KNNImputer(n_neighbors=5),
        IterativeImputer(max_iter=8, random_state=42),
        SimpleImputer(strategy="constant", fill_value=-999),
    ]
    colors_bar = [PALETTE["muted"], PALETTE["info"], PALETTE["primary"],
                  PALETTE["success"], PALETTE["danger"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=PALETTE["bg"])
    fig.suptitle("Imputer Benchmark — ROC-AUC (5-fold CV)",
                 fontsize=14, fontweight="bold", color=PALETTE["text"])

    for ax, (ds_name, (ds_fn, task)) in zip(axes, datasets_bench.items()):
        df  = ds_fn()
        tgt = "Survived" if "Titanic" in ds_name else "target"
        X   = df.drop(columns=[tgt])
        # encode categoricals simply
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category").cat.codes.astype(float).replace(-1, np.nan)
        y   = df[tgt]

        scores_list = []
        for imp in imputer_objs:
            sc = model_score(X, y, imp, label="", task=task)
            scores_list.append(round(sc, 4))

        bars = ax.bar(imputer_names, scores_list, color=colors_bar,
                      alpha=0.9, width=0.6, edgecolor=PALETTE["bg"], linewidth=0.5)
        ax.set_ylim(min(scores_list) * 0.96, max(scores_list) * 1.02)
        ax.set_title(ds_name, color=PALETTE["text"], fontsize=10, fontweight="bold")
        ax.set_ylabel("ROC-AUC", color=PALETTE["text"], fontsize=9)
        ax.tick_params(labelsize=8)
        best_idx = scores_list.index(max(scores_list))
        for j, (bar, val) in enumerate(zip(bars, scores_list)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=7.5,
                    color=PALETTE["success"] if j == best_idx else PALETTE["text"],
                    fontweight="bold" if j == best_idx else "normal")
        ax.axhline(max(scores_list), color=PALETTE["success"], lw=1,
                   ls="--", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("/Users/lakshmikalyani/claude-prep-ml/prep/practice/datasets/kaggle/outputs/03_imputer_benchmark.png",
                dpi=140, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print("  ✅ Saved: 03_imputer_benchmark.png")


# ══════════════════════════════════════════════════════════════════════════════
#  CHEAT SHEET TABLE  — all datasets summarised
# ══════════════════════════════════════════════════════════════════════════════

def print_master_cheatsheet():
    section_header("MASTER CHEAT SHEET — ALL DATASETS & STRATEGIES")

    cheatsheet = """
  ┌──────────────────────┬─────────────────┬──────────────────────────────────────────────────────────┐
  │ DATASET              │ DOMAIN          │ KEY MISSING-VALUE STRATEGIES                             │
  ├──────────────────────┼─────────────────┼──────────────────────────────────────────────────────────┤
  │ Titanic              │ Historical/     │ • Age (20%): KNN (MAR — depends on Pclass/Sex)          │
  │                      │ Transportation  │ • Cabin (77%): create has_cabin flag → drop raw col     │
  │                      │                 │ • Embarked (<1%): mode impute                           │
  │                      │                 │ • Fare (<0.1%): median impute                           │
  ├──────────────────────┼─────────────────┼──────────────────────────────────────────────────────────┤
  │ Heart Disease        │ Cardiology      │ • ColumnTransformer per clinical variable                │
  │                      │                 │ • KNN for correlated vitals (oldpeak, ca, thalach)      │
  │                      │                 │ • Median for skewed labs (trestbps, chol)               │
  │                      │                 │ • NEVER drop rows in healthcare data                    │
  ├──────────────────────┼─────────────────┼──────────────────────────────────────────────────────────┤
  │ Credit Fraud         │ Finance/        │ • PCA features (V1-V28): fill 0 (mathematical mean)    │
  │                      │ Cybersecurity   │ • Amount: median after log transform                    │
  │                      │                 │ • Time (stream): forward fill                           │
  │                      │                 │ • Imbalance: class_weight='balanced'                    │
  ├──────────────────────┼─────────────────┼──────────────────────────────────────────────────────────┤
  │ Breast Cancer        │ Oncology/       │ • High-miss features (15-25%): MICE                    │
  │ (Diagnosis)          │ Radiology       │ • Low-miss features (3-8%): KNN                        │
  │                      │                 │ • Geometric features correlated → always multivariate  │
  ├──────────────────────┼─────────────────┼──────────────────────────────────────────────────────────┤
  │ Breast Cancer        │ Oncology/       │ • Ki67 (30%): MNAR → flag + median                    │
  │ (Survival Study)     │ Epidemiology    │ • ER/PR/HER2: "Unknown" is valid clinical category    │
  │                      │                 │ • Stage/Grade: mode/median (ordinal)                   │
  │                      │                 │ • Age/Tumor: KNN (correlated)                          │
  ├──────────────────────┼─────────────────┼──────────────────────────────────────────────────────────┤
  │ Housing Prices       │ Real Estate     │ • PoolQC/Alley/Fence: structural NaN → "No X"          │
  │                      │                 │ • GarageArea/BsmtSF: structural NaN → 0               │
  │                      │                 │ • LotFrontage (18%): neighbourhood median              │
  │                      │                 │ • Log-transform SalePrice target                       │
  ├──────────────────────┼─────────────────┼──────────────────────────────────────────────────────────┤
  │ Finance / Loans      │ Banking /       │ • Income/Credit: MNAR → flag + median                 │
  │                      │ Credit Risk     │ • Emp_length: 0 (unemployed interpretation)            │
  │                      │                 │ • Delinquencies: 0 (no record = clean)                 │
  │                      │                 │ • Document ALL imputation for compliance                │
  ├──────────────────────┼─────────────────┼──────────────────────────────────────────────────────────┤
  │ Clinical / ICU       │ Critical Care   │ • HbA1c (25%): MNAR → flag + median                  │
  │                      │                 │ • Paired vitals (BP, WBC+Hgb): KNN together           │
  │                      │                 │ • Smoking: "Unknown" category                          │
  │                      │                 │ • Time-series: LOCF for repeated measurements          │
  └──────────────────────┴─────────────────┴──────────────────────────────────────────────────────────┘

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  UNIVERSAL PRODUCTION RULES:
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. ALWAYS fit imputer on TRAIN data only — never on full dataset
  2. ALWAYS create missingness indicator flags BEFORE imputing
  3. ALWAYS interrogate WHY values are missing (MCAR/MAR/MNAR) first
  4. ALWAYS use sklearn Pipelines — never impute outside a pipeline
  5. ALWAYS validate: does imputation improve or hurt model performance?
  6. DOMAIN KNOWLEDGE > statistical technique (always ask domain expert)
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    print(cheatsheet)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    section_header("MISSING VALUE MASTERY — ALL DOMAINS",
                   "Running 7 datasets • 8 strategy patterns • Production pipelines")

    datasets_raw = {}

    datasets_raw["1. Titanic"]               = run_titanic()
    datasets_raw["2. Heart Disease"]         = run_heart_disease()
    datasets_raw["3. Credit Fraud"]          = make_credit_fraud().sample(5000, random_state=42)
    datasets_raw["4A. BC Diagnosis"]         = make_breast_cancer_diagnosis()
    datasets_raw["4B. BC Survival"]          = make_breast_cancer_study()
    datasets_raw["5. Housing"]               = make_housing()
    datasets_raw["6. Finance"]               = run_finance()
    datasets_raw["7. Clinical ICU"]          = run_clinical()
    # run remaining interactively printed ones
    run_credit_fraud()
    run_breast_cancer_diagnosis()
    run_breast_cancer_study()
    run_housing()

    print("\n\n  Generating visualisations …")
    plot_master_summary(datasets_raw)
    plot_decision_framework()
    plot_imputer_comparison()
    print_master_cheatsheet()

    section_header("COMPLETE ✅",
                   "All 7 datasets × 5 imputer strategies demonstrated")
    print("  Output files:")
    print("  • 01_missing_patterns_all_domains.png")
    print("  • 02_decision_framework.png")
    print("  • 03_imputer_benchmark.png")
    print("  • missing_values_master.py  (this file — full source code)\n")