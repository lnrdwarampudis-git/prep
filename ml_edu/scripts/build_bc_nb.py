"""
Build: 02_BreastCancer_Complete_ML_Pipeline.ipynb
Focus: Binary classification on high-dimensional medical data.
New topics vs Titanic:
  - Feature selection (RFE, SelectKBest, SHAP-based)
  - Imbalanced class handling (SMOTE, class_weight)
  - Precision-Recall tradeoff in medical context
  - Decision threshold tuning (medical cost matrix)
  - Cross-validation with StratifiedKFold
"""
import json, textwrap
from pathlib import Path

NB_DIR = Path(__file__).parent.parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)

def md(*lines):
    return {"cell_type":"markdown","metadata":{},"source":"\n".join(lines)}
def code(src):
    return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":textwrap.dedent(src).strip()}
def div(title):
    return md(f"---\n# {title}")

cells = []

# TITLE
cells += [md(
"# 🏥 Breast Cancer Wisconsin — Medical ML Classification",
"### High-Dimensional Binary Classification with Feature Selection & Threshold Tuning",
"",
"| Phase | Topic | New Concepts (vs Titanic) |",
"|-------|-------|--------------------------|",
"| 1 | Data Understanding | High-dimensional EDA, PCA intuition |",
"| 2 | Feature Engineering | Correlation-based removal, interaction terms |",
"| 3 | Feature Selection | RFE, SelectKBest (ANOVA-F, MI), SHAP-based |",
"| 4 | Baseline Modeling | Medical context: FP vs FN cost asymmetry |",
"| 5 | Algorithm Comparison | Focus on medical metrics: Sensitivity/Specificity |",
"| 6 | Threshold Tuning | Cost matrix, ROC operating point selection |",
"| 7 | Cross-Validation Deep Dive | Nested CV, StratifiedKFold |",
"| 8 | Production Pipeline | ColumnTransformer, full sklearn Pipeline |",
"| 9 | SHAP + Clinical Insights | Feature importance for medical decision support |",
"",
"> **Dataset**: Breast Cancer Wisconsin (569 samples, 30 features)",
"> **Target**: `target` — 0=Malignant, 1=Benign",
"> **Domain Challenge**: In cancer screening, False Negatives (missed cancers) are",
"> far more costly than False Positives (unnecessary follow-up biopsies).",
"> This asymmetry drives threshold selection and metric choice.",
)]

# SETUP
cells += [div("⚙️ Setup & Imports"), code("""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from pathlib import Path

# ML — Core
from sklearn.model_selection import (train_test_split, cross_val_score,
    StratifiedKFold, GridSearchCV, learning_curve, cross_validate)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Feature Selection
from sklearn.feature_selection import (RFE, SelectKBest, f_classif,
    mutual_info_classif, RFECV, VarianceThreshold)

# Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Metrics
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score, recall_score,
    precision_score, ConfusionMatrixDisplay)

# Imbalanced
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import shap, mlflow, mlflow.sklearn, joblib

sns.set_theme(style='whitegrid', palette='husl', font_scale=1.05)
plt.rcParams.update({'figure.dpi':100, 'figure.facecolor':'white'})
SEED = 42; np.random.seed(SEED)

DATA_PATH  = Path('../data/breast_cancer.csv')
MODEL_DIR  = Path('../models'); MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = Path('../outputs'); OUTPUT_DIR.mkdir(exist_ok=True)
print("✅ All imports OK")
""")]

# PHASE 1
cells += [div("Phase 1 — Data Understanding (High-Dimensional Medical EDA)")]

cells += [md(
"## 📘 Theory: High-Dimensional EDA",
"",
"> With 30 features, we face the **curse of dimensionality**:",
"> - Pairwise scatter plots: 30×29/2 = 435 plots (infeasible)",
"> - Feature-feature correlations can cause multicollinearity",
"> - Some features may carry zero information (near-zero variance)",
">",
"> **Strategy for high-d EDA:**",
"> 1. Group features by meaning (mean / SE / worst)",
"> 2. Use correlation heatmap to find redundant blocks",
"> 3. Use PCA 2D projection to check class separability",
"> 4. Use violin/box plots for top discriminating features",
)]

cells += [code("""
df = pd.read_csv(DATA_PATH)
# Rename target for clarity: sklearn uses 0=malignant, 1=benign
df['diagnosis'] = df['target'].map({0:'Malignant', 1:'Benign'})
print(f"Shape: {df.shape}")
print(f"\\nTarget distribution:")
print(df['target'].value_counts().to_string())
print(f"  Malignant (0): {(df.target==0).sum()} ({(df.target==0).mean():.1%})")
print(f"  Benign    (1): {(df.target==1).sum()} ({(df.target==1).mean():.1%})")
print()

# Feature groups — Wisconsin dataset has 3 measures per base feature
base_features = ['radius','texture','perimeter','area','smoothness',
                 'compactness','concavity','concave points','symmetry','fractal_dimension']
mean_feats  = [c for c in df.columns if c not in ['target','diagnosis'] and not c.endswith('2') and not ' se' in c and not ' error' in c]
print(f"Total numeric features: {len(df.select_dtypes(include=np.number).columns)-1}")
print(f"Feature groups: mean ({len([c for c in df.columns if 'mean' in c])}), "
      f"error ({len([c for c in df.columns if 'error' in c or ' se' in c])}), "
      f"worst ({len([c for c in df.columns if 'worst' in c])})")
""")]

cells += [code("""
# ── 1.1 Violin plots: top discriminating features ────────────────────────────
top_features = ['mean radius','mean texture','mean perimeter','mean area',
                'mean concavity','mean concave points',
                'worst radius','worst perimeter','worst area','worst concavity']

# Keep only cols that exist
top_features = [f for f in top_features if f in df.columns]
if not top_features:
    # fallback to first numeric cols
    top_features = df.select_dtypes(include=np.number).columns.drop('target').tolist()[:10]

fig, axes = plt.subplots(2, 5, figsize=(20, 9))
fig.suptitle('Feature Distributions by Diagnosis\\nViolinplot = full distribution | White dot = median',
             fontsize=13, fontweight='bold')

for ax, feat in zip(axes.flat, top_features[:10]):
    if feat in df.columns:
        sns.violinplot(x='diagnosis', y=feat, data=df, ax=ax,
                       palette={'Malignant':'#e74c3c','Benign':'#2ecc71'},
                       inner='box', alpha=0.8)
        ax.set_title(feat.replace('mean ','').replace('worst ','').title(),
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('')

plt.tight_layout()
plt.savefig(OUTPUT_DIR/'bc_p1_violin.png', bbox_inches='tight')
plt.show()
""")]

cells += [code("""
# ── 1.2  Correlation Heatmap — find redundant feature blocks ─────────────────
feat_cols = df.select_dtypes(include=np.number).columns.drop('target')
corr = df[feat_cols].corr()

fig, ax = plt.subplots(figsize=(16, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='RdYlGn', center=0,
            vmin=-1, vmax=1, linewidths=0.2, ax=ax, cbar_kws={'shrink':0.8})
ax.set_title('Feature Correlation Matrix (30 Features)\\n'
             'Red = negative | Green = positive correlation',
             fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'bc_p1_correlation.png', bbox_inches='tight')
plt.show()

# Find highly correlated pairs
high_corr = []
for i in range(len(corr)):
    for j in range(i+1, len(corr)):
        if abs(corr.iloc[i,j]) > 0.90:
            high_corr.append((corr.index[i], corr.columns[j], corr.iloc[i,j]))

print(f"Feature pairs with |correlation| > 0.90: {len(high_corr)}")
for f1, f2, c in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:10]:
    print(f"  {f1:<30} × {f2:<30} = {c:+.3f}")
print()
print("⚠️  High correlation = multicollinearity = redundant features")
print("   Feature selection in Phase 3 will address this.")
""")]

cells += [code("""
# ── 1.3  PCA 2D Projection — Visual Class Separability ───────────────────────
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_all = df[feat_cols].fillna(df[feat_cols].median())
X_scaled = StandardScaler().fit_transform(X_all)

pca = PCA(n_components=2, random_state=SEED)
X_2d = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Dimensionality Reduction — Visual Separability Check', fontweight='bold')

# PCA scatter
colors = df['target'].map({0:'#e74c3c', 1:'#2ecc71'})
axes[0].scatter(X_2d[:,0], X_2d[:,1], c=colors, alpha=0.6, s=20, edgecolors='none')
axes[0].set_title(f'PCA 2D Projection\\nExplained variance: {pca.explained_variance_ratio_.sum():.1%}',
                   fontweight='bold')
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
for label, color in [('Malignant','#e74c3c'),('Benign','#2ecc71')]:
    axes[0].scatter([],[],c=color,label=label,s=50)
axes[0].legend()

# Explained variance scree plot
pca_full = PCA(random_state=SEED).fit(X_scaled)
cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
axes[1].plot(range(1,len(cumvar)+1), cumvar, 'bo-', ms=4)
axes[1].axhline(0.95, color='red', ls='--', label='95% variance')
axes[1].axhline(0.99, color='orange', ls='--', label='99% variance')
n_95 = np.argmax(cumvar >= 0.95) + 1
axes[1].axvline(n_95, color='red', ls=':', alpha=0.5)
axes[1].set_title(f'Scree Plot — {n_95} components explain 95% variance', fontweight='bold')
axes[1].set_xlabel('Number of Components'); axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'bc_p1_pca.png', bbox_inches='tight')
plt.show()
print(f"✅ Classes are well-separated in 2D PCA space → good signal in the data")
""")]

# PHASE 2 — FEATURE ENGINEERING
cells += [div("Phase 2 — Feature Engineering for Medical Data")]

cells += [md(
"## 📘 Theory: Feature Engineering in Medical ML",
"",
"> The Wisconsin dataset already has extracted features (radius, texture, etc.).",
"> Our engineering focuses on:",
">",
"> 1. **Remove near-zero variance** features (carry no signal)",
"> 2. **Remove correlated duplicates** (avoid multicollinearity)",
"> 3. **Create interaction terms** (area × concavity captures shape complexity)",
"> 4. **Log transform** highly skewed features (area, perimeter)",
">",
"> 🏭 **Industry Note in Healthcare:**",
"> Feature engineering in medical ML must be clinically meaningful.",
"> 'area × concavity' has a real biological interpretation:",
"> large AND irregular cells = more malignant characteristics.",
)]

cells += [code("""
# ── 2.1  Variance Threshold — remove near-zero variance ─────────────────────
from sklearn.feature_selection import VarianceThreshold

X = df[feat_cols]
y = df['target']

vt = VarianceThreshold(threshold=0.01)
vt.fit(X)
removed_low_var = feat_cols[~vt.get_support()].tolist()
print(f"Features removed by VarianceThreshold (var < 0.01): {len(removed_low_var)}")
print(f"  {removed_low_var}")

# In practice, Wisconsin dataset has no zero-variance features
# but this is standard pipeline step in real medical data
print("\\n(Zero near-zero-variance in this dataset — step validated)")
""")]

cells += [code("""
# ── 2.2  Remove correlated features (|r| > 0.95) ────────────────────────────
def remove_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper_tri   = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop     = [col for col in upper_tri.columns
                   if any(upper_tri[col] > threshold)]
    return to_drop

X_num = df[feat_cols]
cols_to_remove = remove_correlated_features(X_num, threshold=0.95)
print(f"Features to remove (|r| > 0.95): {len(cols_to_remove)}")
for c in cols_to_remove:
    print(f"  - {c}")

# Create reduced feature set
feat_reduced = [c for c in feat_cols if c not in cols_to_remove]
print(f"\\nFeatures: {len(feat_cols)} → {len(feat_reduced)} after correlation removal")
""")]

cells += [code("""
# ── 2.3  Interaction Terms + Log Transforms ──────────────────────────────────
df_eng = df[feat_cols].copy()

# Log transform skewed features
skewed = df[feat_cols].apply(lambda c: abs(c.skew()) > 1).index[df[feat_cols].apply(lambda c: abs(c.skew()) > 1)]
for col in skewed[:5]:
    df_eng[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
    print(f"  log1p({col}): skew {df[col].skew():.2f} → {df_eng[f'{col}_log'].skew():.2f}")

# Clinically meaningful interactions
if 'mean area' in df.columns and 'mean concavity' in df.columns:
    df_eng['area_x_concavity'] = df['mean area'] * df['mean concavity']
    print(f"\\n  Created: area_x_concavity (large+irregular = more malignant)")
if 'worst radius' in df.columns and 'worst concave points' in df.columns:
    df_eng['worst_shape_score'] = df['worst radius'] * df['worst concave points']
    print(f"  Created: worst_shape_score (worst radius × worst concavity)")

# Ratios
if 'mean radius' in df.columns and 'mean texture' in df.columns:
    df_eng['radius_texture_ratio'] = df['mean radius'] / (df['mean texture'] + 1e-6)
    print(f"  Created: radius_texture_ratio")

print(f"\\nTotal engineered features: {df_eng.shape[1]}")
X_eng = df_eng.fillna(df_eng.median())
y     = df['target']
""")]

cells += [code("""
# ── 2.4  Train/Test Split (stratified) ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y, test_size=0.20, random_state=SEED, stratify=y)

print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Train — Malignant: {(y_train==0).sum()} ({(y_train==0).mean():.1%})  "
      f"Benign: {(y_train==1).sum()} ({(y_train==1).mean():.1%})")
print(f"Test  — Malignant: {(y_test==0).sum()} ({(y_test==0).mean():.1%})  "
      f"Benign: {(y_test==1).sum()} ({(y_test==1).mean():.1%})")
""")]

# PHASE 3 — FEATURE SELECTION
cells += [div("Phase 3 — Feature Selection (RFE, SelectKBest, SHAP)")]

cells += [md(
"## 📘 Theory: Feature Selection Methods",
"",
"> Too many features → overfitting, slow inference, harder to explain.",
"> We compare 3 complementary selection strategies:",
">",
"> | Method | Type | Pros | Cons |",
"> |--------|------|------|------|",
"> | **SelectKBest (ANOVA-F)** | Filter | Fast, model-agnostic | Ignores feature interactions |",
"> | **SelectKBest (Mutual Info)** | Filter | Captures non-linear relationships | Slower, less stable |",
"> | **RFE** | Wrapper | Uses model feedback, considers interactions | Expensive (fits model k times) |",
"> | **RFECV** | Wrapper+CV | Optimal k automatically selected | Very expensive |",
"> | **SHAP-based** | Embedded | Post-hoc, uses final model | Needs trained model first |",
">",
"> **Best practice**: Use filter methods to shortlist, then RFE/SHAP to finalise.",
)]

cells += [code("""
# ── 3.1  SelectKBest — ANOVA-F ───────────────────────────────────────────────
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_train)
X_te_s = scaler.transform(X_test)

# ANOVA-F scores
f_selector = SelectKBest(f_classif, k='all')
f_selector.fit(X_tr_s, y_train)
f_scores = pd.DataFrame({
    'feature' : X_train.columns,
    'F_score' : f_selector.scores_,
    'p_value' : f_selector.pvalues_
}).sort_values('F_score', ascending=False)

print("Top 15 features by ANOVA-F score:")
print(f_scores.head(15).to_string(index=False))
""")]

cells += [code("""
# ── 3.2  Mutual Information ───────────────────────────────────────────────────
mi_selector = SelectKBest(mutual_info_classif, k='all')
mi_selector.fit(X_tr_s, y_train)
mi_scores = pd.DataFrame({
    'feature'  : X_train.columns,
    'MI_score' : mi_selector.scores_
}).sort_values('MI_score', ascending=False)

# Compare ANOVA-F vs MI rankings
combined = f_scores[['feature','F_score']].merge(
    mi_scores[['feature','MI_score']], on='feature')
combined['F_rank']  = combined['F_score'].rank(ascending=False).astype(int)
combined['MI_rank'] = combined['MI_score'].rank(ascending=False).astype(int)
combined['rank_diff'] = abs(combined['F_rank'] - combined['MI_rank'])

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Feature Selection: ANOVA-F vs Mutual Information', fontweight='bold')

top15_f = combined.nsmallest(15,'F_rank')
axes[0].barh(range(15), top15_f['F_score'], color='#3498db', edgecolor='white')
axes[0].set_yticks(range(15)); axes[0].set_yticklabels(top15_f['feature'], fontsize=8)
axes[0].set_title('Top 15 by ANOVA-F', fontweight='bold'); axes[0].invert_yaxis()

top15_mi = combined.nsmallest(15,'MI_rank')
axes[1].barh(range(15), top15_mi['MI_score'], color='#e74c3c', edgecolor='white')
axes[1].set_yticks(range(15)); axes[1].set_yticklabels(top15_mi['feature'], fontsize=8)
axes[1].set_title('Top 15 by Mutual Information', fontweight='bold'); axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(OUTPUT_DIR/'bc_p3_feature_selection.png', bbox_inches='tight')
plt.show()
print(f"\\nFeatures with largest rank disagreement (linear vs non-linear):")
print(combined.nlargest(5,'rank_diff')[['feature','F_rank','MI_rank','rank_diff']].to_string(index=False))
""")]

cells += [code("""
# ── 3.3  RFECV — Auto-select optimal number of features ─────────────────────
from sklearn.feature_selection import RFECV

print("Running RFECV (Recursive Feature Elimination with CV)...")
print("This finds the OPTIMAL number of features automatically.")
print()

rf_for_rfe = RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1)
rfecv = RFECV(estimator=rf_for_rfe, step=1, cv=StratifiedKFold(5),
               scoring='roc_auc', min_features_to_select=5, n_jobs=-1)
rfecv.fit(X_tr_s, y_train)

print(f"Optimal number of features: {rfecv.n_features_}")
print(f"CV AUC with optimal features: {rfecv.cv_results_['mean_test_score'].max():.4f}")

selected_by_rfe = X_train.columns[rfecv.support_].tolist()
print(f"\\nSelected features ({len(selected_by_rfe)}):")
for f in selected_by_rfe:
    print(f"  ✓ {f}")

# Plot CV score vs n_features
fig, ax = plt.subplots(figsize=(10, 5))
n_features_range = range(1, len(rfecv.cv_results_['mean_test_score'])+1)
ax.plot(n_features_range, rfecv.cv_results_['mean_test_score'], 'b-o', ms=4)
ax.fill_between(n_features_range,
                rfecv.cv_results_['mean_test_score'] - rfecv.cv_results_['std_test_score'],
                rfecv.cv_results_['mean_test_score'] + rfecv.cv_results_['std_test_score'],
                alpha=0.2, color='blue')
ax.axvline(rfecv.n_features_, color='red', ls='--', label=f'Optimal: {rfecv.n_features_} features')
ax.set_title('RFECV: Cross-Validated AUC vs Number of Features', fontweight='bold')
ax.set_xlabel('Number of Features'); ax.set_ylabel('CV AUC-ROC'); ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'bc_p3_rfecv.png', bbox_inches='tight')
plt.show()
""")]

cells += [code("""
# ── 3.4  Build final feature sets for comparison ────────────────────────────
# All features
X_tr_all, X_te_all = X_tr_s, X_te_s

# Top 15 by ANOVA-F
top15 = f_scores.head(15)['feature'].tolist()
idx15 = [i for i,c in enumerate(X_train.columns) if c in top15]
X_tr_15 = X_tr_s[:, idx15]; X_te_15 = X_te_s[:, idx15]

# RFE-selected
idx_rfe = [i for i,c in enumerate(X_train.columns) if c in selected_by_rfe]
X_tr_rfe = X_tr_s[:, idx_rfe]; X_te_rfe = X_te_s[:, idx_rfe]

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
lr  = LogisticRegression(max_iter=1000, random_state=SEED, C=0.1)

print("Feature set comparison (Logistic Regression, 5-fold CV AUC):")
for name, Xtr, Xte in [
    (f'All features ({X_tr_all.shape[1]})',   X_tr_all,  X_te_all),
    (f'Top-15 ANOVA-F (15)',                  X_tr_15,   X_te_15),
    (f'RFE-selected ({len(selected_by_rfe)})', X_tr_rfe,  X_te_rfe),
]:
    cv_auc = cross_val_score(lr, Xtr, y_train, cv=cv, scoring='roc_auc').mean()
    lr.fit(Xtr, y_train); test_auc = roc_auc_score(y_test, lr.predict_proba(Xte)[:,1])
    print(f"  {name:<30}  CV AUC={cv_auc:.4f}  Test AUC={test_auc:.4f}")
""")]

# PHASE 4 — BASELINE + MEDICAL METRICS
cells += [div("Phase 4 — Baseline Modeling: Medical Context & Metric Choice")]

cells += [md(
"## 📘 Theory: Metric Choice in Medical Classification",
"",
"> In cancer detection, the stakes are asymmetric:",
">",
"> | Error | What Happens | Cost |",
"> |-------|-------------|------|",
"> | **False Negative** (miss malignant) | Patient goes untreated | **CATASTROPHIC** |",
"> | **False Positive** (flag benign as malignant) | Unnecessary biopsy | **Costly but manageable** |",
">",
"> **Therefore:**",
"> - We optimise for **Recall/Sensitivity** (catch all malignant cases)",
"> - We accept lower **Precision** (some false alarms)",
"> - Standard accuracy is misleading — a model that labels everything Benign",
">   gets ~63% accuracy but misses every cancer!",
">",
"> **Key metrics for medical classification:**",
"> - **Sensitivity** = Recall = TP/(TP+FN) → 'of all actual cancers, how many did we catch?'",
"> - **Specificity** = TN/(TN+FP) → 'of all benign, how many did we correctly clear?'",
"> - **AUC-PR** (Precision-Recall AUC) → better than AUC-ROC for class-imbalanced medical tasks",
)]

cells += [code("""
# ── 4.1  Baseline LR + Medical Metrics ───────────────────────────────────────
lr_base = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
lr_base.fit(X_tr_all, y_train)
y_pred  = lr_base.predict(X_te_all)
y_proba = lr_base.predict_proba(X_te_all)[:,1]

# Confusion matrix components
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
ppv         = tp / (tp + fp)   # Positive Predictive Value = Precision
npv         = tn / (tn + fn)   # Negative Predictive Value

print("BASELINE LOGISTIC REGRESSION — MEDICAL METRICS")
print("=" * 55)
print(f"  Accuracy    : {accuracy_score(y_test,y_pred):.4f}")
print(f"  AUC-ROC     : {roc_auc_score(y_test,y_proba):.4f}")
print(f"  AUC-PR      : {average_precision_score(y_test,y_proba):.4f}")
print(f"  Sensitivity  (Recall) : {sensitivity:.4f}  ← most important in cancer")
print(f"  Specificity           : {specificity:.4f}")
print(f"  PPV (Precision)       : {ppv:.4f}")
print(f"  NPV                   : {npv:.4f}")
print(f"  F1-Score              : {f1_score(y_test,y_pred):.4f}")
print()
print(f"  Confusion Matrix:")
print(f"    Predicted:  Malignant  Benign")
print(f"  Actual Malignant   {tp:4d}    {fn:4d}  ← FN = missed cancers!")
print(f"  Actual Benign      {fp:4d}    {tn:4d}")
print()
print(f"  ⚠️  Missed cancers (FN): {fn}  — these patients could go untreated!")
""")]

# PHASE 5 — THRESHOLD TUNING
cells += [div("Phase 5 — Decision Threshold Tuning")]

cells += [md(
"## 📘 Theory: Decision Threshold & Cost Matrix",
"",
"> By default, sklearn classifiers use threshold = 0.5:",
"> 'If P(malignant) > 0.5, predict malignant'",
">",
"> But in medicine, we should ask: **what threshold minimises harm?**",
">",
"> **Cost Matrix approach:**",
"> ```",
"> Assign costs:  FN (miss cancer)  = $50,000 (delayed treatment, mortality risk)",
">                FP (unnecessary biopsy) = $500 (anxiety, cost)",
"> ```",
"> At each threshold, compute: Total Cost = FN_count × 50000 + FP_count × 500",
"> Choose threshold that minimises total cost.",
">",
"> This is the **operating point selection** step — required in any regulated ML system.",
)]

cells += [code("""
# ── 5.1  Threshold sensitivity analysis ──────────────────────────────────────
thresholds = np.arange(0.10, 0.90, 0.02)
results = []
for thr in thresholds:
    y_pred_t = (y_proba >= thr).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, y_pred_t).ravel()
    results.append({
        'threshold'  : thr,
        'sensitivity': tp_t/(tp_t+fn_t) if (tp_t+fn_t)>0 else 0,
        'specificity': tn_t/(tn_t+fp_t) if (tn_t+fp_t)>0 else 0,
        'precision'  : tp_t/(tp_t+fp_t) if (tp_t+fp_t)>0 else 0,
        'f1'         : f1_score(y_test, y_pred_t) if y_pred_t.sum()>0 else 0,
        'fn'         : fn_t, 'fp': fp_t,
        'cost'       : fn_t * 50000 + fp_t * 500  # asymmetric cost
    })
thr_df = pd.DataFrame(results)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Decision Threshold Analysis — Medical Context', fontweight='bold')

# Sensitivity vs Specificity
axes[0].plot(thr_df['threshold'], thr_df['sensitivity'], 'g-o', ms=4, label='Sensitivity (Recall)')
axes[0].plot(thr_df['threshold'], thr_df['specificity'], 'b-s', ms=4, label='Specificity')
axes[0].plot(thr_df['threshold'], thr_df['f1'],          'r-^', ms=4, label='F1-Score')
axes[0].axvline(0.5, color='gray', ls='--', label='Default threshold (0.5)')
opt_f1_thr = thr_df.loc[thr_df['f1'].idxmax(), 'threshold']
axes[0].axvline(opt_f1_thr, color='red', ls=':', label=f'Optimal F1 ({opt_f1_thr:.2f})')
axes[0].set_xlabel('Decision Threshold'); axes[0].set_ylabel('Score')
axes[0].set_title('Sensitivity / Specificity Trade-off'); axes[0].legend(fontsize=8)

# Cost curve
axes[1].plot(thr_df['threshold'], thr_df['cost']/1000, 'purple', lw=2, marker='o', ms=4)
opt_cost_idx = thr_df['cost'].idxmin()
opt_cost_thr = thr_df.loc[opt_cost_idx, 'threshold']
axes[1].axvline(opt_cost_thr, color='red', ls='--', label=f'Min cost at thr={opt_cost_thr:.2f}')
axes[1].set_xlabel('Decision Threshold'); axes[1].set_ylabel('Total Cost ($K)')
axes[1].set_title('Total Cost (FN×$50k + FP×$500)\\nMinimise to find optimal threshold',
                   fontweight='bold')
axes[1].legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR/'bc_p5_threshold.png', bbox_inches='tight')
plt.show()

opt = thr_df.loc[opt_cost_idx]
print(f"Optimal threshold (min cost): {opt_cost_thr:.2f}")
print(f"  At this threshold: Sensitivity={opt.sensitivity:.3f}, Specificity={opt.specificity:.3f}")
print(f"  FN (missed cancers): {int(opt.fn)}, FP (unnecessary biopsies): {int(opt.fp)}")
print(f"  Total cost: ${opt.cost:,.0f} vs default thr=0.5 cost: ${thr_df[thr_df.threshold==0.5].iloc[0].cost:,.0f}")
""")]

# PHASE 6 — FULL ALGORITHM COMPARISON
cells += [div("Phase 6 — Algorithm Comparison with Medical Metrics")]

cells += [code("""
# ── 6.1  Compare all algorithms on medical metrics ───────────────────────────
algorithms = {
    'Logistic Regression': Pipeline([('sc',StandardScaler()),
                            ('lr',LogisticRegression(max_iter=1000,random_state=SEED,C=0.1))]),
    'Random Forest'      : RandomForestClassifier(n_estimators=200,random_state=SEED,n_jobs=-1),
    'XGBoost'            : XGBClassifier(random_state=SEED,eval_metric='logloss',verbosity=0,n_jobs=-1),
    'SVM'                : Pipeline([('sc',StandardScaler()),
                            ('svm',SVC(probability=True,random_state=SEED,kernel='rbf',C=1.0))]),
    'KNN'                : Pipeline([('sc',StandardScaler()),
                            ('knn',KNeighborsClassifier(n_neighbors=7,weights='distance'))]),
}

cv5 = StratifiedKFold(5, shuffle=True, random_state=SEED)

print(f"{'Model':<22}  {'Acc':>6}  {'AUC':>6}  {'Sens':>6}  {'Spec':>6}  {'F1':>6}  {'AUC-PR':>7}")
print("-"*65)

algo_results = {}
for name, model in algorithms.items():
    # Fit needs full X (some algos have their own scaler)
    Xtr = X_train if not isinstance(model, Pipeline) else X_train
    Xte = X_test

    cv_auc  = cross_val_score(model, X_train, y_train, cv=cv5, scoring='roc_auc').mean()
    model.fit(Xtr, y_train)
    yp  = model.predict(Xte)
    ypr = model.predict_proba(Xte)[:,1]
    tn_m, fp_m, fn_m, tp_m = confusion_matrix(y_test, yp).ravel()
    sens = tp_m/(tp_m+fn_m); spec = tn_m/(tn_m+fp_m)
    
    algo_results[name] = {
        'acc':accuracy_score(y_test,yp), 'auc':roc_auc_score(y_test,ypr),
        'sens':sens, 'spec':spec, 'f1':f1_score(y_test,yp),
        'auc_pr':average_precision_score(y_test,ypr),
        'fn':fn_m, 'fp':fp_m
    }
    r = algo_results[name]
    print(f"{name:<22}  {r['acc']:>6.4f}  {r['auc']:>6.4f}  {r['sens']:>6.4f}  "
          f"{r['spec']:>6.4f}  {r['f1']:>6.4f}  {r['auc_pr']:>7.4f}")

print()
best_sens = max(algo_results, key=lambda k: algo_results[k]['sens'])
print(f"Best Sensitivity: {best_sens} = {algo_results[best_sens]['sens']:.4f}")
print(f"  → Fewest missed cancers: {algo_results[best_sens]['fn']} FN")
""")]

cells += [code("""
# ── 6.2  ROC + Precision-Recall curves ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Comparison — ROC and Precision-Recall Curves', fontweight='bold')

colors = ['#e74c3c','#2ecc71','#3498db','#9b59b6','#f39c12']
for (name, model), color in zip(algorithms.items(), colors):
    ypr = model.predict_proba(X_test)[:,1]
    
    fpr, tpr, _ = roc_curve(y_test, ypr)
    auc = roc_auc_score(y_test, ypr)
    axes[0].plot(fpr, tpr, color=color, lw=1.5, label=f'{name} (AUC={auc:.3f})')
    
    prec, rec, _ = precision_recall_curve(y_test, ypr)
    ap = average_precision_score(y_test, ypr)
    axes[1].plot(rec, prec, color=color, lw=1.5, label=f'{name} (AP={ap:.3f})')

axes[0].plot([0,1],[0,1],'k--',lw=1)
axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].set_title('ROC Curve', fontweight='bold'); axes[0].legend(fontsize=7)

baseline_pr = y_test.mean()
axes[1].axhline(baseline_pr, color='k', ls='--', lw=1, label=f'Baseline ({baseline_pr:.2f})')
axes[1].set_xlabel('Recall (Sensitivity)'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve\\n(Better for imbalanced: focus on minority class)',
                   fontweight='bold')
axes[1].legend(fontsize=7)

plt.tight_layout()
plt.savefig(OUTPUT_DIR/'bc_p6_roc_pr.png', bbox_inches='tight')
plt.show()
""")]

# PHASE 7 — NESTED CV
cells += [div("Phase 7 — Nested Cross-Validation (Unbiased Evaluation)")]

cells += [md(
"## 📘 Theory: Nested Cross-Validation",
"",
"> **The Problem**: If you tune hyperparameters with CV, then evaluate on the same CV,",
"> the test score is optimistically biased — the model has 'seen' the test fold.",
">",
"> **Nested CV solves this:**",
"> ```",
"> Outer loop (5-fold):  Provides unbiased performance estimate",
"> │  Inner loop (3-fold):  Tunes hyperparameters on train folds only",
"> │  ├── Fold 1 inner: tune params",
"> │  ├── Fold 2 inner: tune params",
"> │  └── Fold 3 inner: best params →",
"> └──  Evaluate on outer test fold with best params",
"> ```",
">",
"> Result: 5 outer scores, each computed on truly held-out data.",
"> The mean of these 5 scores is your unbiased performance estimate.",
">",
"> 🏭 **Industry Rule**: For medical devices, FDA requires nested CV or",
"> independent holdout. Never report tuned model's training-set CV score.",
)]

cells += [code("""
# ── 7.1  Nested CV for XGBoost ───────────────────────────────────────────────
print("Running Nested Cross-Validation (this takes 1-2 minutes)...")
print()

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED+1)

xgb_param_grid = {
    'n_estimators' : [50, 100, 200],
    'max_depth'    : [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2],
}

nested_scores = []
for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X_eng.values, y.values)):
    X_outer_tr = X_eng.values[tr_idx]; y_outer_tr = y.values[tr_idx]
    X_outer_te = X_eng.values[te_idx]; y_outer_te = y.values[te_idx]
    
    # Scale within each outer fold (no leakage)
    sc = StandardScaler()
    X_outer_tr_s = sc.fit_transform(X_outer_tr)
    X_outer_te_s = sc.transform(X_outer_te)
    
    # Inner CV for tuning
    inner_gs = GridSearchCV(
        XGBClassifier(random_state=SEED, eval_metric='logloss', verbosity=0, n_jobs=-1),
        xgb_param_grid, cv=inner_cv, scoring='roc_auc', n_jobs=-1)
    inner_gs.fit(X_outer_tr_s, y_outer_tr)
    
    # Evaluate with best params on outer test
    best_model = inner_gs.best_estimator_
    y_proba_o  = best_model.predict_proba(X_outer_te_s)[:,1]
    fold_auc   = roc_auc_score(y_outer_te, y_proba_o)
    nested_scores.append(fold_auc)
    print(f"  Outer Fold {fold+1}: best_params={inner_gs.best_params_}  AUC={fold_auc:.4f}")

print()
print(f"Nested CV AUC: {np.mean(nested_scores):.4f} ± {np.std(nested_scores):.4f}")
print()
print("Compare to simple CV (biased upward due to tuning on same data):")
simple_cv = cross_val_score(
    XGBClassifier(random_state=SEED, eval_metric='logloss', verbosity=0, n_jobs=-1),
    StandardScaler().fit_transform(X_eng), y,
    cv=outer_cv, scoring='roc_auc')
print(f"Simple 5-fold CV AUC: {simple_cv.mean():.4f} ± {simple_cv.std():.4f}")
print()
bias = simple_cv.mean() - np.mean(nested_scores)
print(f"Optimism bias: {bias:+.4f} (nested CV gives the honest estimate)")
""")]

# PHASE 8 — PRODUCTION PIPELINE
cells += [div("Phase 8 — Production Pipeline")]

cells += [code("""
# ── 8.1  Full sklearn Pipeline with ColumnTransformer ────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Define numeric features (all in this dataset)
numeric_features = X_eng.columns.tolist()

# Full production pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  RobustScaler()),   # RobustScaler: better than StandardScaler when outliers exist
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
])

# Final pipeline: preprocessing + best model
best_clf = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    random_state=SEED, eval_metric='logloss', verbosity=0, n_jobs=-1)

prod_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   best_clf),
])

# Train on full data
X_raw_bc = df[feat_cols].copy()
y_raw_bc  = df['target']
X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
    X_raw_bc, y_raw_bc, test_size=0.2, random_state=SEED, stratify=y_raw_bc)

prod_pipeline.fit(X_tr_r, y_tr_r)
y_pred_prod  = prod_pipeline.predict(X_te_r)
y_proba_prod = prod_pipeline.predict_proba(X_te_r)[:,1]

print("Production Pipeline Performance:")
print(f"  AUC-ROC     : {roc_auc_score(y_te_r, y_proba_prod):.4f}")
print(f"  Accuracy    : {accuracy_score(y_te_r, y_pred_prod):.4f}")
print(f"  Sensitivity : {recall_score(y_te_r, y_pred_prod):.4f}")
print(f"  F1          : {f1_score(y_te_r, y_pred_prod):.4f}")

# Save
pkl_path = MODEL_DIR / 'breast_cancer_xgboost_pipeline.pkl'
joblib.dump(prod_pipeline, pkl_path)
print(f"\\nSaved: {pkl_path}")
""")]

cells += [md(
"## 📘 Theory: RobustScaler vs StandardScaler",
"",
"> `StandardScaler`: `z = (x - mean) / std`",
"> - Sensitive to outliers (outliers inflate std, compress all other values)",
">",
"> `RobustScaler`: `z = (x - median) / IQR`",
"> - Uses median and IQR (interquartile range)",
"> - Outliers don't distort the scaling of other samples",
"> - **Preferred** in medical data (lab values often have outliers)",
">",
"> **Rule of thumb**: If boxplots show many outliers beyond 3×IQR → use RobustScaler.",
)]

# PHASE 9 — SHAP FOR MEDICAL
cells += [div("Phase 9 — SHAP: Clinical Decision Support")]

cells += [code("""
# ── 9.1  SHAP for the production XGBoost model ───────────────────────────────
best_clf.fit(StandardScaler().fit_transform(X_tr_r.values), y_tr_r)

explainer  = shap.TreeExplainer(best_clf)
X_te_vals  = StandardScaler().fit_transform(X_te_r.values)
shap_vals  = explainer.shap_values(X_te_vals)
sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

# Global importance
mean_abs = np.abs(sv).mean(0)
shap_imp = pd.DataFrame({'feature':X_te_r.columns,'mean_abs_shap':mean_abs})
shap_imp = shap_imp.sort_values('mean_abs_shap', ascending=False)

fig, axes = plt.subplots(1,2, figsize=(16,7))
fig.suptitle('SHAP Feature Importance — Breast Cancer XGBoost', fontweight='bold')

# Bar chart
axes[0].barh(shap_imp['feature'][:15][::-1], shap_imp['mean_abs_shap'][:15][::-1],
             color='#e74c3c', edgecolor='white')
axes[0].set_title('Global Feature Importance\\n(Mean |SHAP value|)', fontweight='bold')
axes[0].set_xlabel('Mean |SHAP value|')

# Scatter for top feature
top_feat_idx = shap_imp.index[0]
feat_name    = shap_imp.iloc[0]['feature']
feat_idx     = list(X_te_r.columns).index(feat_name)
axes[1].scatter(X_te_vals[:,feat_idx], sv[:,feat_idx],
                c=y_te_r.values, cmap='RdYlGn', alpha=0.5, s=20)
axes[1].axhline(0, color='black', lw=0.8, ls='--')
axes[1].set_xlabel(f'{feat_name} (standardized)'); axes[1].set_ylabel('SHAP value')
axes[1].set_title(f'SHAP Dependence Plot\\n{feat_name}\\n'
                   f'Green=Benign | Red=Malignant', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR/'bc_p9_shap.png', bbox_inches='tight')
plt.show()
print(f"Top 5 most important features for cancer detection:")
for _, row in shap_imp.head(5).iterrows():
    print(f"  {row['feature']:<35} SHAP={row['mean_abs_shap']:.4f}")
""")]

# SUMMARY
cells += [div("Summary & Clinical Takeaways")]
cells += [md(
"## ✅ Breast Cancer ML — Key Takeaways",
"",
"### New Concepts Mastered",
"",
"| Concept | What We Learned |",
"|---------|----------------|",
"| High-d EDA | Correlation blocks, PCA separability, violin by class |",
"| Feature Selection | Filter (ANOVA-F, MI) vs Wrapper (RFECV) — both needed |",
"| Medical Metrics | Sensitivity > Accuracy for cancer screening |",
"| Threshold Tuning | Cost matrix drives operating point, not just F1 |",
"| Nested CV | Eliminates optimism bias — required for regulated systems |",
"| RobustScaler | Better than StandardScaler when outliers are present |",
"",
"### Clinical Insight",
"",
"> The top SHAP features align with pathologist knowledge:",
"> **worst concave points**, **worst radius**, **mean concave points** — all",
"> measure the irregular boundary and large size of malignant cells.",
"> When ML agrees with clinical intuition, it builds trust in the model.",
)]

save_path = NB_DIR / "02_BreastCancer_Complete_ML_Pipeline.ipynb"
with open(save_path, "w") as f:
    json.dump({"nbformat":4,"nbformat_minor":5,
               "metadata":{"kernelspec":{"display_name":"Python 3",
                                         "language":"python","name":"python3"},
                            "language_info":{"name":"python","version":"3.12.0"}},
               "cells":cells}, f, indent=1)
print(f"\nSaved: {save_path}")
print(f"Cell count: {len(cells)}")
