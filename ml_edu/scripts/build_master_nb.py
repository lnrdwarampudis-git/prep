"""
Build: 00_Master_Comparison_All_Datasets.ipynb
Side-by-side comparison across all 3 datasets showing:
- Curriculum overview map
- Key results comparison table
- Cross-dataset concept matrix
- When to use what (decision guide)
"""
import json, textwrap
from pathlib import Path

NB_DIR = Path(__file__).parent.parent / "notebooks"

def md(*lines):
    return {"cell_type":"markdown","metadata":{},"source":"\n".join(lines)}
def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":textwrap.dedent(src).strip()}
def div(title):
    return md(f"---\n# {title}")

cells = []

cells += [md(
"# 🎓 ML Educational Series — Master Comparison Notebook",
"## Side-by-Side Analysis: Titanic | Breast Cancer | California Housing",
"",
"This notebook runs **all 3 datasets in one place** to show:",
"- How the same phases apply differently across problem types",
"- Direct metric comparisons",
"- Decision framework: *which technique to use when*",
"",
"| Notebook | Dataset | Problem Type | Key Challenge |",
"|----------|---------|-------------|---------------|",
"| 01 | Titanic (891 rows, 12 cols) | Binary Classification | Missing data, feature engineering |",
"| 02 | Breast Cancer (569 rows, 30 cols) | Binary Classification | High-dimensional, medical metrics |",
"| 03 | California Housing (20,640 rows, 9 cols) | Regression | Spatial patterns, prediction intervals |",
)]

cells += [div("⚙️ Setup")]
cells += [code("""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
    mean_absolute_error, r2_score, mean_absolute_percentage_error,
    mean_squared_error)
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor
import shap

sns.set_theme(style='whitegrid', palette='husl', font_scale=1.05)
plt.rcParams.update({'figure.dpi':100, 'figure.facecolor':'white'})
SEED = 42; np.random.seed(SEED)

DATA_DIR   = Path('../data')
OUTPUT_DIR = Path('../outputs'); OUTPUT_DIR.mkdir(exist_ok=True)
print("✅ Setup complete")
""")]

cells += [div("Section 1 — Dataset Overview & EDA Comparison")]

cells += [md(
"## 📘 The 3 Core ML Problem Types",
"",
"> | Type | Example Here | Target | Primary Metrics |",
"> |------|-------------|--------|----------------|",
"> | **Binary Classification** | Titanic survival | 0/1 discrete | AUC-ROC, F1, Accuracy |",
"> | **Binary Classification (medical)** | Cancer diagnosis | 0/1 asymmetric cost | Sensitivity, AUC-PR |",
"> | **Regression** | House prices | Continuous | RMSE, MAE, R² |",
)]

cells += [code("""
# ── Load all 3 datasets ──────────────────────────────────────────────────────
df_titanic = pd.read_csv(DATA_DIR / 'titanic.csv')
df_cancer  = pd.read_csv(DATA_DIR / 'breast_cancer.csv')
df_housing = pd.read_csv(DATA_DIR / 'housing.csv')

datasets = {
    'Titanic'       : df_titanic,
    'Breast Cancer' : df_cancer,
    'Housing'       : df_housing,
}

print("Dataset Summary")
print("=" * 65)
for name, df in datasets.items():
    n, p = df.shape
    miss = df.isnull().sum().sum()
    print(f"  {name:<16}  {n:>6,} rows  {p:>3} cols  "
          f"{miss:>4} nulls ({miss/(n*p)*100:.1f}%)")
""")]

cells += [code("""
# ── Side-by-side EDA comparison ──────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle('Side-by-Side EDA: All 3 Datasets', fontsize=15, fontweight='bold')

# ─── Titanic ────────────────────────────────────────────────────────────────
ax_t1 = fig.add_subplot(gs[0, 0])
surv = df_titanic.groupby('Sex')['Survived'].mean()
ax_t1.bar(surv.index, surv.values, color=['#e74c3c','#3498db'], width=0.4)
ax_t1.set_title('Titanic: Survival by Sex', fontweight='bold')
ax_t1.set_ylabel('Survival Rate'); ax_t1.set_ylim(0,1)
for i,(k,v) in enumerate(surv.items()): ax_t1.text(i,v+0.02,f'{v:.0%}',ha='center',fontweight='bold')

ax_t2 = fig.add_subplot(gs[0, 1])
df_titanic['Age'].dropna().hist(bins=30, ax=ax_t2, color='#3498db', edgecolor='none', alpha=0.8)
ax_t2.set_title('Titanic: Age Distribution\n(20% missing → MAR)', fontweight='bold')
ax_t2.set_xlabel('Age')

ax_t3 = fig.add_subplot(gs[0, 2])
miss_t = df_titanic.isnull().mean().sort_values(ascending=False)
miss_t = miss_t[miss_t > 0]
ax_t3.bar(miss_t.index, miss_t.values, color=['#e74c3c','#f39c12','#3498db'])
ax_t3.set_title('Titanic: Missing Data\n(MCAR/MAR/MNAR types)', fontweight='bold')
ax_t3.set_ylabel('% Missing'); ax_t3.tick_params(axis='x', rotation=30)

# ─── Breast Cancer ─────────────────────────────────────────────────────────
ax_b1 = fig.add_subplot(gs[1, 0])
target_counts = df_cancer['target'].value_counts()
ax_b1.bar(['Malignant (0)','Benign (1)'], target_counts.values,
           color=['#e74c3c','#2ecc71'], width=0.5)
ax_b1.set_title('Breast Cancer: Class Balance\n(62% benign, 38% malignant)', fontweight='bold')
ax_b1.set_ylabel('Count')

ax_b2 = fig.add_subplot(gs[1, 1])
feat_cols_bc = [c for c in df_cancer.columns if c != 'target']
corr_with_target = df_cancer[feat_cols_bc].corrwith(df_cancer['target']).abs().sort_values(ascending=False)
corr_with_target.head(10).plot(kind='bar', ax=ax_b2, color='#9b59b6', edgecolor='white')
ax_b2.set_title('Breast Cancer: Top 10\nFeature-Target Correlations', fontweight='bold')
ax_b2.set_ylabel('|Pearson r|'); ax_b2.tick_params(axis='x', rotation=45)

ax_b3 = fig.add_subplot(gs[1, 2])
from sklearn.decomposition import PCA
X_bc = df_cancer[feat_cols_bc].fillna(df_cancer[feat_cols_bc].median())
X_bc_s = StandardScaler().fit_transform(X_bc)
pca2 = PCA(n_components=2, random_state=SEED).fit_transform(X_bc_s)
colors_bc = df_cancer['target'].map({0:'#e74c3c',1:'#2ecc71'})
ax_b3.scatter(pca2[:,0], pca2[:,1], c=colors_bc, alpha=0.5, s=10)
ax_b3.set_title('Breast Cancer: PCA 2D\n(well-separated classes)', fontweight='bold')
ax_b3.set_xlabel('PC1'); ax_b3.set_ylabel('PC2')
for lbl, col in [('Malignant','#e74c3c'),('Benign','#2ecc71')]:
    ax_b3.scatter([],[],c=col,label=lbl,s=40); ax_b3.legend(fontsize=8)

# ─── Housing ────────────────────────────────────────────────────────────────
ax_h1 = fig.add_subplot(gs[2, 0])
ax_h1.hist(df_housing['MedHouseVal'], bins=50, color='#f39c12', edgecolor='none', alpha=0.8)
ax_h1.set_title(f'Housing: Target Distribution\nSkew={df_housing.MedHouseVal.skew():.2f} → log-transform',
                fontweight='bold')
ax_h1.set_xlabel('Med House Value ($100k)')

ax_h2 = fig.add_subplot(gs[2, 1])
ax_h2.scatter(df_housing['MedInc'], df_housing['MedHouseVal'],
               alpha=0.05, s=3, color='#1abc9c')
ax_h2.set_title('Housing: Income vs Value\n(strong non-linear relationship)', fontweight='bold')
ax_h2.set_xlabel('Median Income'); ax_h2.set_ylabel('Med House Value')

ax_h3 = fig.add_subplot(gs[2, 2])
sc = ax_h3.scatter(df_housing['Longitude'], df_housing['Latitude'],
                    c=df_housing['MedHouseVal'], cmap='RdYlGn', s=1, alpha=0.3)
plt.colorbar(sc, ax=ax_h3, label='Value ($100k)')
ax_h3.set_title('Housing: Geographic Price Map\n(spatial autocorrelation)', fontweight='bold')
ax_h3.set_xlabel('Longitude'); ax_h3.set_ylabel('Latitude')

plt.savefig(OUTPUT_DIR/'master_eda_comparison.png', bbox_inches='tight', dpi=100)
plt.show()
""")]

cells += [div("Section 2 — Model Performance Comparison")]

cells += [code("""
# ── Quick benchmark: best model per dataset ──────────────────────────────────
print("Running quick benchmarks across all 3 datasets...")
print("(XGBoost with default params, 5-fold CV)")
print()

# ─── Titanic ────────────────────────────────────────────────────────────────
import re
def extract_title(name):
    match = re.search(r',\\s*([^.]+)\\.', str(name))
    if match:
        t = match.group(1).strip()
        return 'Rare' if t not in ['Mr','Mrs','Miss','Master'] else t
    return 'Mr'

df_t = df_titanic.copy()
df_t['Title']      = df_t['Name'].apply(extract_title)
df_t['FamilySize'] = df_t['SibSp'] + df_t['Parch'] + 1
df_t['HasCabin']   = df_t['Cabin'].notna().astype(int)
df_t['Fare_log']   = np.log1p(df_t['Fare'].fillna(df_t['Fare'].median()))
df_t['Age']        = df_t.groupby(['Pclass','Sex'])['Age'].transform(
                         lambda x: x.fillna(x.median()))
df_t['Age']        = df_t['Age'].fillna(df_t['Age'].median())
df_t['Embarked']   = df_t['Embarked'].fillna('S')
df_t_enc = pd.get_dummies(df_t[['Survived','Pclass','Sex','Age','SibSp','Parch',
                                  'Fare_log','HasCabin','FamilySize','Title','Embarked']],
                            columns=['Sex','Title','Embarked'], drop_first=True)
X_t = df_t_enc.drop('Survived',axis=1); y_t = df_t_enc['Survived']
cv_t = StratifiedKFold(5, shuffle=True, random_state=SEED)
xgb_t = XGBClassifier(random_state=SEED, eval_metric='logloss', verbosity=0, n_jobs=-1)
auc_t = cross_val_score(xgb_t, X_t, y_t, cv=cv_t, scoring='roc_auc').mean()
acc_t = cross_val_score(xgb_t, X_t, y_t, cv=cv_t, scoring='accuracy').mean()
print(f"  Titanic        XGBoost  AUC={auc_t:.4f}  Accuracy={acc_t:.4f}")

# ─── Breast Cancer ──────────────────────────────────────────────────────────
X_bc = df_cancer.drop('target',axis=1).fillna(df_cancer.median(numeric_only=True))
y_bc = df_cancer['target']
cv_bc = StratifiedKFold(5, shuffle=True, random_state=SEED)
xgb_bc = XGBClassifier(random_state=SEED, eval_metric='logloss', verbosity=0, n_jobs=-1)
auc_bc = cross_val_score(xgb_bc, X_bc, y_bc, cv=cv_bc, scoring='roc_auc').mean()
acc_bc = cross_val_score(xgb_bc, X_bc, y_bc, cv=cv_bc, scoring='accuracy').mean()
print(f"  Breast Cancer  XGBoost  AUC={auc_bc:.4f}  Accuracy={acc_bc:.4f}")

# ─── Housing ────────────────────────────────────────────────────────────────
X_h  = df_housing.drop('MedHouseVal',axis=1)
y_h  = np.log1p(df_housing['MedHouseVal'])
cv_h = KFold(5, shuffle=True, random_state=SEED)
xgb_h = XGBRegressor(random_state=SEED, verbosity=0, n_jobs=-1)
rmse_h = -cross_val_score(xgb_h, X_h, y_h, cv=cv_h, scoring='neg_root_mean_squared_error').mean()
r2_h   = cross_val_score(xgb_h, X_h, y_h, cv=cv_h, scoring='r2').mean()
print(f"  Housing        XGBoost  RMSE={rmse_h:.4f}  R²={r2_h:.4f}")
""")]

cells += [code("""
# ── Comprehensive performance chart ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('XGBoost Performance Across All 3 Datasets (5-fold CV)',
             fontsize=13, fontweight='bold')

# Titanic metrics
xgb_t.fit(X_t, y_t)
Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(X_t, y_t, test_size=0.2,
                                                 random_state=SEED, stratify=y_t)
xgb_t.fit(Xt_tr, yt_tr)
yt_proba = xgb_t.predict_proba(Xt_te)[:,1]
yt_pred  = xgb_t.predict(Xt_te)

metrics_t = {'Accuracy':accuracy_score(yt_te,yt_pred),
              'AUC-ROC':roc_auc_score(yt_te,yt_proba),
              'F1-Score':f1_score(yt_te,yt_pred)}
axes[0].bar(metrics_t.keys(), metrics_t.values(), color=['#3498db','#2ecc71','#e74c3c'], width=0.5)
axes[0].set_ylim(0, 1.05)
axes[0].set_title('Titanic\n(Binary Classification)', fontweight='bold')
for i,(k,v) in enumerate(metrics_t.items()): axes[0].text(i,v+0.01,f'{v:.4f}',ha='center',fontsize=9)

# Breast Cancer
Xbc_tr, Xbc_te, ybc_tr, ybc_te = train_test_split(X_bc, y_bc, test_size=0.2,
                                                     random_state=SEED, stratify=y_bc)
xgb_bc.fit(Xbc_tr, ybc_tr)
ybc_proba = xgb_bc.predict_proba(Xbc_te)[:,1]
ybc_pred  = xgb_bc.predict(Xbc_te)
from sklearn.metrics import recall_score, average_precision_score
metrics_bc = {'Accuracy':accuracy_score(ybc_te,ybc_pred),
               'AUC-ROC':roc_auc_score(ybc_te,ybc_proba),
               'Sensitivity':recall_score(ybc_te,ybc_pred)}
axes[1].bar(metrics_bc.keys(), metrics_bc.values(), color=['#3498db','#2ecc71','#9b59b6'], width=0.5)
axes[1].set_ylim(0,1.05)
axes[1].set_title('Breast Cancer\n(Medical Classification)', fontweight='bold')
for i,(k,v) in enumerate(metrics_bc.items()): axes[1].text(i,v+0.01,f'{v:.4f}',ha='center',fontsize=9)

# Housing
Xh_tr, Xh_te, yh_tr_log, yh_te_log = train_test_split(X_h, y_h, test_size=0.2, random_state=SEED)
yh_te_raw = np.expm1(yh_te_log)
xgb_h.fit(Xh_tr, yh_tr_log)
yh_pred = np.expm1(xgb_h.predict(Xh_te))
mae_h_dollars = mean_absolute_error(yh_te_raw, yh_pred) * 100000
mape_h = mean_absolute_percentage_error(yh_te_raw, yh_pred)*100
r2_test = r2_score(yh_te_raw, yh_pred)
metrics_h = {'MAE ($K)':mae_h_dollars/1000, 'MAPE (%)':mape_h, 'R² (×10)':r2_test*10}
axes[2].bar(metrics_h.keys(), metrics_h.values(), color=['#e74c3c','#f39c12','#1abc9c'], width=0.5)
axes[2].set_title('Housing\n(Regression)', fontweight='bold')
for i,(k,v) in enumerate(metrics_h.items()): axes[2].text(i,v+0.3,f'{v:.2f}',ha='center',fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR/'master_performance.png', bbox_inches='tight')
plt.show()
print(f"Housing: MAE=${mae_h_dollars:,.0f} | MAPE={mape_h:.1f}% | R²={r2_test:.4f}")
""")]

cells += [div("Section 3 — SHAP Comparison Across All Datasets")]

cells += [code("""
# ── SHAP importance for all 3 datasets ──────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 7))
fig.suptitle('SHAP Feature Importance: All 3 Datasets', fontsize=13, fontweight='bold')

datasets_shap = [
    ('Titanic',       xgb_t,  Xt_te,  'Binary Classification'),
    ('Breast Cancer', xgb_bc, Xbc_te, 'Medical Classification'),
    ('Housing',       xgb_h,  Xh_te,  'Regression'),
]

for ax, (name, model, Xte, problem_type) in zip(axes, datasets_shap):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xte[:200])
    if isinstance(sv, list): sv = sv[1]
    
    mean_abs = np.abs(sv).mean(0)
    feat_names = Xte.columns.tolist()
    imp = pd.Series(mean_abs, index=feat_names).sort_values(ascending=True).tail(12)
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(imp)))
    ax.barh(range(len(imp)), imp.values, color=colors, edgecolor='white')
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels([f.replace('_',' ')[:25] for f in imp.index], fontsize=7)
    ax.set_title(f'{name}\\n{problem_type}\\nTop 12 SHAP Features', fontweight='bold', fontsize=9)
    ax.set_xlabel('Mean |SHAP value|')

plt.tight_layout()
plt.savefig(OUTPUT_DIR/'master_shap_comparison.png', bbox_inches='tight')
plt.show()
""")]

cells += [div("Section 4 — Concept Decision Framework")]

cells += [md(
"## 📘 When to Use What — Complete Decision Guide",
"",
"### Problem Type Selection",
"",
"```",
"Is your target variable continuous?",
"  YES → Regression (Housing notebook)",
"  NO  → Classification",
"        Is class imbalance > 4:1?",
"          YES → Use SMOTE / class_weight + AUC-PR metric",
"          NO  → Standard classification",
"               Is cost of FN >> cost of FP? (medical)",
"                 YES → Tune decision threshold with cost matrix",
"                 NO  → Use F1 or accuracy",
"```",
"",
"### Algorithm Selection Guide",
"",
"| Situation | Recommended Algorithm | Why |",
"|-----------|----------------------|-----|",
"| Need interpretability | Logistic Regression, Decision Tree | Coefficients / rules readable |",
"| Tabular data, performance matters | XGBoost / LightGBM | State of art for tabular |",
"| Small dataset (< 1000 rows) | SVM, Logistic Regression | Less prone to overfit |",
"| High-dimensional (> 100 features) | Lasso / ElasticNet (linear), RF/XGB (non-linear) | Feature selection built-in |",
"| Geographic/spatial data | Add K-Means cluster features first | Captures spatial autocorrelation |",
"| Time is critical (production) | LightGBM > XGBoost | 3–10× faster training |",
"| Need prediction intervals | Quantile Regression | Direct interval estimation |",
"| Maximum accuracy | Stacking Ensemble | Combines diverse model strengths |",
"",
"### Metric Selection Guide",
"",
"| Scenario | Primary Metric | Secondary |",
"|----------|---------------|-----------|",
"| Balanced binary classification | AUC-ROC | F1, Accuracy |",
"| Medical / high FN cost | Sensitivity (Recall) | AUC-PR |",
"| Regression, \\$ interpretable | MAE | R² |",
"| Regression, large errors costly | RMSE | R² |",
"| Regression, relative errors | MAPE | R² |",
"| Class imbalance | AUC-PR | F1 |",
"",
"### Feature Engineering Decision Tree",
"",
"```",
"Does your data have:",
"  Missing values?",
"    |-- < 5% missing   → Median/mode imputation (MCAR)",
"    |-- 5–30% missing  → Check if MAR → group-based imputation",
"    └── > 30% or MNAR  → Binary flag + impute",
"  Text columns?",
"    └── Extract structure (Titanic: title from name)",
"  Skewed numeric features (|skew| > 1)?",
"    └── log1p transform",
"  Geographic coordinates?",
"    └── K-Means spatial clusters",
"  High correlation between features (|r| > 0.9)?",
"    └── Remove one of each correlated pair",
"  Non-linear relationship with target?",
"    └── Binning, interaction terms, polynomial features",
"```",
)]

cells += [code("""
# ── Summary table ─────────────────────────────────────────────────────────────
summary = pd.DataFrame({
    'Dataset'       : ['Titanic','Breast Cancer','Housing'],
    'Problem Type'  : ['Binary Classif.','Medical Classif.','Regression'],
    'Rows'          : [891, 569, 20640],
    'Features'      : [12, 30, 9],
    'Key Challenge' : ['Missing data (MCAR/MAR/MNAR)',
                        'High-dim, asymmetric costs',
                        'Non-linear, spatial patterns'],
    'Best Algo'     : ['XGBoost','XGBoost/RF','LightGBM/Stacking'],
    'Key Technique' : ['Title FE + HasCabin flag',
                        'RFECV + threshold tuning',
                        'Spatial clusters + log target'],
    'Top Metric'    : ['AUC-ROC','Sensitivity','MAE ($)'],
})
print("Complete ML Educational Series — Summary")
print("=" * 95)
print(summary.to_string(index=False))
print()
print("✅ All 3 notebooks complete.")
print("   Titanic       → 01_Titanic_Complete_ML_Pipeline.ipynb")
print("   Breast Cancer → 02_BreastCancer_Complete_ML_Pipeline.ipynb")
print("   Housing       → 03_Housing_Complete_Regression_Pipeline.ipynb")
print("   Master        → 00_Master_Comparison_All_Datasets.ipynb")
""")]

# CONCEPT COVERAGE MATRIX
cells += [div("Section 5 — Concept Coverage Matrix")]

cells += [code("""
# ── Visual concept coverage map ───────────────────────────────────────────────
concepts = [
    'EDA + Missing Analysis (MCAR/MAR/MNAR)',
    'Feature Engineering (domain knowledge)',
    'Log Transform (skewed features/target)',
    'Baseline Logistic Regression',
    '6-Algorithm CV Comparison',
    'Learning Curves (Bias-Variance)',
    'GridSearchCV + RandomizedSearchCV',
    'Production sklearn Pipeline',
    'SHAP Explainability',
    'MLflow Experiment Tracking',
    'Feature Selection (RFE/SelectKBest)',
    'Medical Metrics (Sensitivity/AUC-PR)',
    'Decision Threshold Tuning',
    'Nested Cross-Validation',
    'Spatial Feature Engineering',
    'Regression Metrics (RMSE/MAE/R²)',
    'Residual Analysis',
    'Model Stacking Ensemble',
    'Prediction Intervals (Quantile)',
    'RobustScaler vs StandardScaler',
]

coverage = {
    'Titanic'       : [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
    'Breast Cancer' : [1,1,0,1,1,0,1,1,1,0,1,1,1,1,0,0,0,0,0,1],
    'Housing'       : [1,1,1,0,1,0,1,1,1,1,0,0,0,0,1,1,1,1,1,1],
}

cov_df = pd.DataFrame(coverage, index=concepts)

fig, ax = plt.subplots(figsize=(10, 12))
colors  = {0:'#ecf0f1', 1:'#27ae60'}
cmap    = plt.cm.colors.ListedColormap([colors[0], colors[1]])

heatmap_data = cov_df.values
im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)

ax.set_xticks(range(3))
ax.set_xticklabels(['Titanic\n🚢', 'Breast Cancer\n🏥', 'Housing\n🏠'],
                    fontsize=11, fontweight='bold')
ax.set_yticks(range(len(concepts)))
ax.set_yticklabels(concepts, fontsize=8)

for i in range(len(concepts)):
    for j in range(3):
        symbol = '✓' if heatmap_data[i,j] else '·'
        color  = 'white' if heatmap_data[i,j] else '#bdc3c7'
        ax.text(j, i, symbol, ha='center', va='center',
                fontsize=12, color=color, fontweight='bold')

ax.set_title('ML Concept Coverage Matrix\n✓ = covered in depth  · = not in this notebook',
             fontsize=12, fontweight='bold', pad=15)

# Add coverage count
for j, ds in enumerate(['Titanic','Breast Cancer','Housing']):
    n_covered = cov_df[ds].sum()
    ax.text(j, len(concepts)+0.6, f'{n_covered}/{len(concepts)} topics',
            ha='center', fontsize=9, color='#2c3e50')

plt.tight_layout()
plt.savefig(OUTPUT_DIR/'master_concept_matrix.png', bbox_inches='tight')
plt.show()
print(f"\\nCoverage totals:")
for ds in ['Titanic','Breast Cancer','Housing']:
    print(f"  {ds:<16}: {cov_df[ds].sum()}/{len(concepts)} concepts")
print(f"  Combined      : {(cov_df.max(axis=1)==1).sum()}/{len(concepts)} unique concepts covered")
""")]

save_path = NB_DIR / "00_Master_Comparison_All_Datasets.ipynb"
with open(save_path, "w") as f:
    json.dump({"nbformat":4,"nbformat_minor":5,
               "metadata":{"kernelspec":{"display_name":"Python 3",
                                         "language":"python","name":"python3"},
                            "language_info":{"name":"python","version":"3.12.0"}},
               "cells":cells}, f, indent=1)
print(f"\nSaved: {save_path}")
print(f"Cell count: {len(cells)}")
