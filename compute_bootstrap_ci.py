import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Guarantee path discovery
sys.path.insert(0, os.getcwd())

print("[1/6] Loading data...", flush=True)
# 1. Load Data
X_train = pd.read_csv("X_train_resampled.csv")
Y_train = pd.read_csv("Y_train_resampled.csv")['hml32']
X_test = pd.read_csv("X_test.csv")
Y_test = pd.read_csv("Y_test.csv")['hml32'].values

continuous_feature = ['b19']
nominal_features = ['b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128', 'v129', 'v113', 'v116', 'hml20']

# Types for black-box / scikit-learn
X_train_bb = X_train.copy()
X_test_bb = X_test.copy()
X_train_bb['b19'] = X_train_bb['b19'].astype(float)
X_test_bb['b19'] = X_test_bb['b19'].astype(float)
for col in nominal_features:
    X_train_bb[col] = X_train_bb[col].astype(int).astype(str)
    X_test_bb[col] = X_test_bb[col].astype(int).astype(str)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_feature),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features)
    ]
)

# 2. Get predictions and probabilities for all 6 models
models_data = {}

print("[2/6] Scoring Zero-R...", flush=True)
# A. Zero-R Majority Baseline
zero_preds = np.zeros_like(Y_test)
zero_probs = np.full_like(Y_test, fill_value=9.0/49.0, dtype=float)
models_data["Majority Class (Zero-R Baseline)"] = {
    "preds": zero_preds,
    "probs": zero_probs,
    "is_baseline": True
}

print("[3/6] Fitting and scoring Logistic Regression...", flush=True)
# B. Logistic Regression Linear Baseline
lr_pipe = Pipeline([
    ('prep', preprocessor),
    ('clf', LogisticRegression(random_state=42))
])
lr_pipe.fit(X_train_bb, Y_train)
lr_preds = lr_pipe.predict(X_test_bb)
lr_probs = lr_pipe.predict_proba(X_test_bb)[:, 1]
models_data["Logistic Regression (Linear Baseline)"] = {
    "preds": lr_preds,
    "probs": lr_probs,
    "is_baseline": False
}

print("[4/6] Scoring C4.5...", flush=True)
# C. C4.5 Decision Tree
from outputs.rules.rules import findDecision
X_test_c45 = X_test.copy()
X_test_c45['b19'] = X_test_c45['b19'].astype(int)
for col in nominal_features:
    X_test_c45[col] = X_test_c45[col].astype(int).astype(str).astype(object)

c45_preds = []
for idx, row in X_test_c45.iterrows():
    obj = [
        row['b19'], row['b4'], row['v106'], row['v190'], row['v119'], 
        row['v158'], row['v127'], row['v128'], row['v129'], row['v113'], 
        row['v116'], row['hml20']
    ]
    pred_str = findDecision(obj)
    c45_preds.append(1 if pred_str == 'Positive' else 0)
c45_preds = np.array(c45_preds)
c45_probs = c45_preds.astype(float)
models_data["C4.5 Decision Tree (White-Box)"] = {
    "preds": c45_preds,
    "probs": c45_probs,
    "is_baseline": False
}

print("[5/6] Fitting and scoring RIPPER...", flush=True)
# D. RIPPER Ruleset
import wittgenstein as lw
X_train_ripper = X_train.copy()
X_test_ripper = X_test.copy()
X_train_ripper['b19'] = X_train_ripper['b19'].astype(int)
X_test_ripper['b19'] = X_test_ripper['b19'].astype(int)
for col in nominal_features:
    X_train_ripper[col] = X_train_ripper[col].astype(int).astype(str).astype(object)
    X_test_ripper[col] = X_test_ripper[col].astype(int).astype(str).astype(object)

ripper_model = lw.RIPPER(random_state=42)
ripper_model.fit(X_train_ripper, Y_train, pos_class=1)
ripper_raw_preds = ripper_model.predict(X_test_ripper)
ripper_preds = np.array([1 if p else 0 for p in ripper_raw_preds])
ripper_probs = ripper_model.predict_proba(X_test_ripper)[:, 1]
models_data["RIPPER Ruleset (White-Box)"] = {
    "preds": ripper_preds,
    "probs": ripper_probs,
    "is_baseline": False
}

print("[6/6] Scoring Random Forest and SVM...", flush=True)
# E. Random Forest
with open(os.path.join("outputs", "models", "random_forest_pipeline.pkl"), 'rb') as f:
    rf_pipe = pickle.load(f)
rf_preds = rf_pipe.predict(X_test_bb)
rf_probs = rf_pipe.predict_proba(X_test_bb)[:, 1]
models_data["Random Forest (Ensemble Benchmark)"] = {
    "preds": rf_preds,
    "probs": rf_probs,
    "is_baseline": False
}

# F. Support Vector Classifier (SVM)
with open(os.path.join("outputs", "models", "svm_pipeline.pkl"), 'rb') as f:
    svm_pipe = pickle.load(f)
svm_preds = svm_pipe.predict(X_test_bb)
svm_probs = svm_pipe.predict_proba(X_test_bb)[:, 1]
models_data["SVM (RBF Kernel Benchmark)"] = {
    "preds": svm_preds,
    "probs": svm_probs,
    "is_baseline": False
}


print("All 6 models successfully loaded and scored on test partition (N=49)!\n")

def compute_metrics(y_true, y_pred, y_prob, is_zero_r=False):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = (tp + tn) / (tp + tn + fp + fn) * 100.0
    sens = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
    
    # Precision and F1
    prec_pos = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    f1_pos = 2 * prec_pos * sens / (prec_pos + sens) if (prec_pos + sens) > 0 else 0.0
    
    prec_neg = tn / (tn + fn) * 100.0 if (tn + fn) > 0 else 0.0
    f1_neg = 2 * prec_neg * spec / (prec_neg + spec) if (prec_neg + spec) > 0 else 0.0
    
    macro_f1 = (f1_pos + f1_neg) / 2.0
    
    # ROC-AUC & PR-AUC
    if is_zero_r:
        roc_auc = 0.5000
        pr_auc = np.mean(y_true)
    else:
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_prob)
            p_c, r_c, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(r_c, p_c)
        else:
            roc_auc = np.nan
            pr_auc = np.nan
            
    return {
        "Accuracy": acc,
        "Sensitivity": sens,
        "Specificity": spec,
        "Precision_Pos": prec_pos,
        "F1_Pos": f1_pos,
        "Precision_Neg": prec_neg,
        "F1_Neg": f1_neg,
        "Macro_F1": macro_f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc
    }

# Bootstrap routine
B = 1000
np.random.seed(42)
N = len(Y_test)

bootstrap_results = {}

print("Starting bootstrap loop...", flush=True)
for name, data in models_data.items():
    print(f"Bootstrapping {name}...", flush=True)
    preds = data["preds"]
    probs = data["probs"]
    is_base = data["is_baseline"]
    
    # Point estimates
    point = compute_metrics(Y_test, preds, probs, is_zero_r=is_base)
    
    # Bootstrap replicates
    boot_metrics = {k: [] for k in point.keys()}
    
    b_count = 0
    while b_count < B:
        idx = np.random.choice(N, size=N, replace=True)
        y_b = Y_test[idx]
        if len(np.unique(y_b)) < 2:
            continue
        pred_b = preds[idx]
        prob_b = probs[idx]
        
        m_b = compute_metrics(y_b, pred_b, prob_b, is_zero_r=is_base)
        for k in boot_metrics:
            boot_metrics[k].append(m_b[k])
        b_count += 1
        
    ci_res = {}
    for k in point.keys():
        vals = np.array(boot_metrics[k])
        lo = np.percentile(vals, 2.5)
        hi = np.percentile(vals, 97.5)
        ci_res[k] = {
            "point": point[k],
            "ci_lo": lo,
            "ci_hi": hi
        }
    bootstrap_results[name] = ci_res
print("Bootstrapping complete!", flush=True)

# Print formatted summary
print("="*100)
print("NON-PARAMETRIC BOOTSTRAP 95% CONFIDENCE INTERVALS (B = 1,000, seed = 42)")
print("="*100)

for name, res in bootstrap_results.items():
    print(f"\n--- {name} ---")
    for k, v in res.items():
        if "AUC" in k:
            if name == "Majority Class (Zero-R Baseline)" and k == "ROC_AUC":
                print(f"  {k:<15}: {v['point']:.4f} [—]")
            else:
                print(f"  {k:<15}: {v['point']:.4f} [{v['ci_lo']:.2f}, {v['ci_hi']:.2f}] (or [{v['ci_lo']:.4f}, {v['ci_hi']:.4f}])")
        else:
            print(f"  {k:<15}: {v['point']:.2f}% [{v['ci_lo']:.1f}, {v['ci_hi']:.1f}]")

