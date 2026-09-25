import os
import sys
import io
import numpy as np

# Ensure UTF-8 stdout encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

print("="*90)
print("STEP 8: GROUPED HOUSEHOLD & CLUSTER SENSITIVITY ANALYSIS (ITEM 8)")
print("="*90)

# Load data
df = pd.read_csv("final_malaria_sdoh_matrix.csv")

# Create composite household ID
df['hh_id'] = df['hv001'].astype(str) + "_" + df['hv002'].astype(str)
clusters = df['hv001'].values
households = df['hh_id'].values

n_total = len(df)
n_pos = df['hml32'].sum()
n_clusters = df['hv001'].nunique()
n_households = df['hh_id'].nunique()

print(f"Analytic Cohort Structure:")
print(f" - Total children: {n_total}")
print(f" - Microscopy Positive (hml32=1): {int(n_pos)} ({n_pos/n_total*100:.2f}%)")
print(f" - Unique Enumeration Areas (Clusters, hv001): {n_clusters}")
print(f" - Unique Households (hv001 + hv002): {n_households}")
print(f" - Mean children per household: {n_total/n_households:.2f}")

continuous_feature = ['b19']
nominal_features = ['b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128', 'v129', 'v113', 'v116', 'hml20']
feature_cols = continuous_feature + nominal_features

X_raw = df[feature_cols].copy()
y_raw = df['hml32'].astype(int).values
cat_indices = list(range(1, len(feature_cols)))

def get_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_feature),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features)
        ]
    )

def eval_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = (tp + tn) / (tp + tn + fp + fn) * 100.0 if (tp + tn + fp + fn) > 0 else 0.0
    sens = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
    prec_pos = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    f1_pos = 2 * prec_pos * sens / (prec_pos + sens) if (prec_pos + sens) > 0 else 0.0
    prec_neg = tn / (tn + fn) * 100.0 if (tn + fn) > 0 else 0.0
    f1_neg = 2 * prec_neg * spec / (prec_neg + spec) if (prec_neg + spec) > 0 else 0.0
    macro_f1 = (f1_pos + f1_neg) / 2.0
    
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except Exception:
            roc_auc = 0.5000
        try:
            p_c, r_c, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(r_c, p_c)
        except Exception:
            pr_auc = np.mean(y_true)
    else:
        roc_auc = 0.5000
        pr_auc = np.mean(y_true)
        
    return {
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "Precision_Pos": prec_pos,
        "F1_Pos": f1_pos,
        "Macro_F1": macro_f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc
    }

# Function to run cross-validation under a specific splitting generator
def run_cv_experiment(cv_generator, scheme_name):
    print(f"\n--- Running 5-Fold Evaluation: {scheme_name} ---")
    
    results = {
        "Logistic Regression": {k: [] for k in ["Accuracy", "Sensitivity", "Specificity", "Precision_Pos", "F1_Pos", "Macro_F1", "ROC_AUC", "PR_AUC"]},
        "Random Forest": {k: [] for k in ["Accuracy", "Sensitivity", "Specificity", "Precision_Pos", "F1_Pos", "Macro_F1", "ROC_AUC", "PR_AUC"]},
        "Support Vector Machine": {k: [] for k in ["Accuracy", "Sensitivity", "Specificity", "Precision_Pos", "F1_Pos", "Macro_F1", "ROC_AUC", "PR_AUC"]}
    }
    
    fold = 1
    for train_idx, test_idx in cv_generator:
        X_train_f = X_raw.iloc[train_idx].copy()
        y_train_f = y_raw[train_idx]
        X_test_f = X_raw.iloc[test_idx].copy()
        y_test_f = y_raw[test_idx]
        
        # Check if test fold has positive cases
        if np.sum(y_test_f == 1) == 0:
            continue
            
        # Format types
        X_train_clean = X_train_f.copy()
        X_test_clean = X_test_f.copy()
        X_train_clean['b19'] = X_train_clean['b19'].astype(float)
        X_test_clean['b19'] = X_test_clean['b19'].astype(float)
        for col in nominal_features:
            X_train_clean[col] = X_train_clean[col].astype(int).astype(str)
            X_test_clean[col] = X_test_clean[col].astype(int).astype(str)
            
        # SMOTE-NC on training fold only
        try:
            smote = SMOTENC(categorical_features=cat_indices, random_state=42, k_neighbors=3)
            X_train_res, y_train_res = smote.fit_resample(X_train_clean, y_train_f)
        except Exception:
            X_train_res, y_train_res = X_train_clean, y_train_f
            
        # 1. LR
        pipe_lr = Pipeline([('prep', get_preprocessor()), ('clf', LogisticRegression(C=1.0, random_state=42, max_iter=1000))])
        pipe_lr.fit(X_train_res, y_train_res)
        m_lr = eval_metrics(y_test_f, pipe_lr.predict(X_test_clean), pipe_lr.predict_proba(X_test_clean)[:, 1])
        for k in m_lr: results["Logistic Regression"][k].append(m_lr[k])
        
        # 2. RF
        pipe_rf = Pipeline([('prep', get_preprocessor()), ('clf', RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, random_state=42, n_jobs=1))])
        pipe_rf.fit(X_train_res, y_train_res)
        m_rf = eval_metrics(y_test_f, pipe_rf.predict(X_test_clean), pipe_rf.predict_proba(X_test_clean)[:, 1])
        for k in m_rf: results["Random Forest"][k].append(m_rf[k])
        
        # 3. SVM
        pipe_svm = Pipeline([('prep', get_preprocessor()), ('clf', SVC(C=100, gamma=1, kernel='rbf', probability=True, random_state=42))])
        pipe_svm.fit(X_train_res, y_train_res)
        m_svm = eval_metrics(y_test_f, pipe_svm.predict(X_test_clean), pipe_svm.predict_proba(X_test_clean)[:, 1])
        for k in m_svm: results["Support Vector Machine"][k].append(m_svm[k])
        
        fold += 1
        
    means = {}
    for model_name, metrics in results.items():
        means[model_name] = {k: np.mean(v) for k, v in metrics.items()}
    return means

# 1. Scheme A: Standard Stratified 5-Fold (Random row splitting)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res_random = run_cv_experiment(skf.split(X_raw, y_raw), "Scheme A: Standard Stratified 5-Fold (Random Row Split)")

# 2. Scheme B: Household-Grouped 5-Fold (Strict household boundaries)
gkf_hh = GroupKFold(n_splits=5)
res_household = run_cv_experiment(gkf_hh.split(X_raw, y_raw, groups=households), "Scheme B: Household-Grouped 5-Fold (No Intra-Household Leakage)")

# 3. Scheme C: Cluster-Grouped 5-Fold (Spatial cluster boundaries)
gkf_cluster = GroupKFold(n_splits=5)
res_cluster = run_cv_experiment(gkf_cluster.split(X_raw, y_raw, groups=clusters), "Scheme C: Cluster-Grouped 5-Fold (Unseen Rural Clusters)")

# Synthesis Comparison Table
print("\n" + "="*120)
print("SENSITIVITY ANALYSIS SUMMARY: RANDOM SPLIT VS. HOUSEHOLD-GROUPED VS. CLUSTER-GROUPED (5-FOLD CV)")
print("="*120)
print(f"{'Validation Paradigm':<42} | {'Model':<24} | {'Accuracy':<10} | {'Sensitivity':<12} | {'Specificity':<12} | {'Macro F1':<10} | {'ROC-AUC':<9} | {'PR-AUC':<9}")
print("-" * 140)

schemes = [
    ("Scheme A: Standard Stratified K-Fold", res_random),
    ("Scheme B: Household-Grouped K-Fold", res_household),
    ("Scheme C: Cluster-Grouped K-Fold", res_cluster)
]

comparison_rows = []
for scheme_label, res_dict in schemes:
    for model_name, m in res_dict.items():
        print(f"{scheme_label:<42} | {model_name:<24} | {m['Accuracy']:<9.2f}% | {m['Sensitivity']:<11.2f}% | {m['Specificity']:<11.2f}% | {m['Macro_F1']:<9.2f}% | {m['ROC_AUC']:<9.4f} | {m['PR_AUC']:<9.4f}")
        row = {"Validation_Scheme": scheme_label, "Model": model_name}
        row.update(m)
        comparison_rows.append(row)

print("="*120)

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv("outputs/grouped_cluster_sensitivity_results.csv", index=False)
print("\nExported sensitivity analysis results to outputs/grouped_cluster_sensitivity_results.csv")
