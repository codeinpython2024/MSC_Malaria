import os
import sys
import io
import numpy as np

# Ensure UTF-8 stdout encoding for clean symbols (e.g. ±)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
import pandas as pd
from sklearn.model_selection import train_test_split
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
    auc,
    precision_score,
    recall_score,
    f1_score
)

print("="*90)
print("STEP 7: REPEATED TRAIN/TEST EXPERIMENT ACROSS MULTIPLE RANDOM SEEDS (ITEM 6)")
print("="*90)

# Load data
df = pd.read_csv("final_malaria_sdoh_matrix.csv")

continuous_feature = ['b19']
nominal_features = ['b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128', 'v129', 'v113', 'v116', 'hml20']
feature_cols = continuous_feature + nominal_features

X_raw = df[feature_cols].copy()
y_raw = df['hml32'].astype(int).values

cat_indices = list(range(1, len(feature_cols)))

# Define preprocessing pipeline for scikit-learn estimators
def get_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_feature),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features)
        ]
    )

# 10 random seeds
SEEDS = [10, 20, 30, 42, 50, 60, 70, 80, 90, 100]

def eval_metrics(y_true, y_pred, y_prob, is_zero_r=False):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = (tp + tn) / (tp + tn + fp + fn) * 100.0
    sens = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
    
    prec_pos = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    f1_pos = 2 * prec_pos * sens / (prec_pos + sens) if (prec_pos + sens) > 0 else 0.0
    
    prec_neg = tn / (tn + fn) * 100.0 if (tn + fn) > 0 else 0.0
    f1_neg = 2 * prec_neg * spec / (prec_neg + spec) if (prec_neg + spec) > 0 else 0.0
    
    macro_f1 = (f1_pos + f1_neg) / 2.0
    
    if is_zero_r:
        roc_auc = 0.5000
        pr_auc = np.mean(y_true)
    else:
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

models_results = {
    "Majority Class (Zero-R)": {k: [] for k in ["Accuracy", "Sensitivity", "Specificity", "Precision_Pos", "F1_Pos", "Macro_F1", "ROC_AUC", "PR_AUC"]},
    "Logistic Regression": {k: [] for k in ["Accuracy", "Sensitivity", "Specificity", "Precision_Pos", "F1_Pos", "Macro_F1", "ROC_AUC", "PR_AUC"]},
    "Decision Tree (C4.5 / Entropy)": {k: [] for k in ["Accuracy", "Sensitivity", "Specificity", "Precision_Pos", "F1_Pos", "Macro_F1", "ROC_AUC", "PR_AUC"]},
    "Random Forest (Ensemble)": {k: [] for k in ["Accuracy", "Sensitivity", "Specificity", "Precision_Pos", "F1_Pos", "Macro_F1", "ROC_AUC", "PR_AUC"]},
    "Support Vector Machine (RBF)": {k: [] for k in ["Accuracy", "Sensitivity", "Specificity", "Precision_Pos", "F1_Pos", "Macro_F1", "ROC_AUC", "PR_AUC"]}
}

print(f"Executing 80/20 train/test split across {len(SEEDS)} distinct random seeds with isolated SMOTE-NC...\n")

for i, seed in enumerate(SEEDS, 1):
    print(f"Seed [{i:02d}/10]: {seed} ...", flush=True)
    
    # 1. Stratified 80/20 train/test partition
    X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=seed, stratify=y_raw
    )
    
    # Cast nominals as str for preprocessing pipeline
    X_train_clean = X_train_raw.copy()
    X_test_clean = X_test_raw.copy()
    X_train_clean['b19'] = X_train_clean['b19'].astype(float)
    X_test_clean['b19'] = X_test_clean['b19'].astype(float)
    for col in nominal_features:
        X_train_clean[col] = X_train_clean[col].astype(int).astype(str)
        X_test_clean[col] = X_test_clean[col].astype(int).astype(str)
        
    # 2. Resample training fold ONLY via SMOTE-NC
    # For SMOTE-NC, categorical features must be ints/objects
    smote = SMOTENC(categorical_features=cat_indices, random_state=seed, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_clean, y_train_raw)
    
    # --- A. Zero-R Baseline ---
    y_pred_zr = np.zeros_like(y_test)
    y_prob_zr = np.full_like(y_test, fill_value=np.mean(y_train_res), dtype=float)
    m_zr = eval_metrics(y_test, y_pred_zr, y_prob_zr, is_zero_r=True)
    for k in m_zr: models_results["Majority Class (Zero-R)"][k].append(m_zr[k])
    
    # --- B. Logistic Regression ---
    pipe_lr = Pipeline([
        ('prep', get_preprocessor()),
        ('clf', LogisticRegression(C=1.0, random_state=seed, max_iter=1000))
    ])
    pipe_lr.fit(X_train_res, y_train_res)
    pred_lr = pipe_lr.predict(X_test_clean)
    prob_lr = pipe_lr.predict_proba(X_test_clean)[:, 1]
    m_lr = eval_metrics(y_test, pred_lr, prob_lr)
    for k in m_lr: models_results["Logistic Regression"][k].append(m_lr[k])
    
    # --- C. Decision Tree (C4.5 Entropy) ---
    pipe_dt = Pipeline([
        ('prep', get_preprocessor()),
        ('clf', DecisionTreeClassifier(criterion='entropy', random_state=seed, min_samples_leaf=3))
    ])
    pipe_dt.fit(X_train_res, y_train_res)
    pred_dt = pipe_dt.predict(X_test_clean)
    prob_dt = pipe_dt.predict_proba(X_test_clean)[:, 1]
    m_dt = eval_metrics(y_test, pred_dt, prob_dt)
    for k in m_dt: models_results["Decision Tree (C4.5 / Entropy)"][k].append(m_dt[k])
    
    # --- D. Random Forest ---
    pipe_rf = Pipeline([
        ('prep', get_preprocessor()),
        ('clf', RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5, random_state=seed, n_jobs=1))
    ])
    pipe_rf.fit(X_train_res, y_train_res)
    pred_rf = pipe_rf.predict(X_test_clean)
    prob_rf = pipe_rf.predict_proba(X_test_clean)[:, 1]
    m_rf = eval_metrics(y_test, pred_rf, prob_rf)
    for k in m_rf: models_results["Random Forest (Ensemble)"][k].append(m_rf[k])
    
    # --- E. Support Vector Machine (RBF) ---
    pipe_svm = Pipeline([
        ('prep', get_preprocessor()),
        ('clf', SVC(C=100, gamma=1, kernel='rbf', probability=True, random_state=seed))
    ])
    pipe_svm.fit(X_train_res, y_train_res)
    pred_svm = pipe_svm.predict(X_test_clean)
    prob_svm = pipe_svm.predict_proba(X_test_clean)[:, 1]
    m_svm = eval_metrics(y_test, pred_svm, prob_svm)
    for k in m_svm: models_results["Support Vector Machine (RBF)"][k].append(m_svm[k])

print("\nAll 10 random seeds completed successfully!\n")

# Summary Table Construction
summary_rows = []
print("="*120)
print(f"{'Model Architecture':<30} | {'Global Accuracy':<17} | {'Sensitivity (Pos)':<18} | {'Specificity (Neg)':<18} | {'Precision (Pos)':<18} | {'Macro F1':<15} | {'ROC-AUC':<14} | {'PR-AUC':<14}")
print("="*120)

for model_name, metrics in models_results.items():
    row_dict = {"Model": model_name}
    line = f"{model_name:<30} | "
    for k in ["Accuracy", "Sensitivity", "Specificity", "Precision_Pos", "Macro_F1", "ROC_AUC", "PR_AUC"]:
        vals = np.array(metrics[k])
        mean_v = np.mean(vals)
        std_v = np.std(vals, ddof=1)
        row_dict[f"{k}_Mean"] = mean_v
        row_dict[f"{k}_Std"] = std_v
        if "AUC" in k:
            val_str = f"{mean_v:.4f} ± {std_v:.4f}"
            line += f"{val_str:<14} | "
        else:
            val_str = f"{mean_v:.2f}% ± {std_v:.2f}%"
            if k == "Accuracy":
                line += f"{val_str:<17} | "
            elif "Pos" in k or "Neg" in k or k == "Sensitivity" or k == "Specificity":
                line += f"{val_str:<18} | "
            else:
                line += f"{val_str:<15} | "
    print(line[:-3])
    summary_rows.append(row_dict)

print("="*120)

# Save to CSV and Markdown
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("outputs/multiseed_stability_results.csv", index=False)
print("\nExported multi-seed stability results to outputs/multiseed_stability_results.csv")
