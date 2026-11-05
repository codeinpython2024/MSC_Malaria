import os
import sys
import io
import json
import numpy as np

# Ensure UTF-8 stdout encoding for clean symbols (e.g. ±)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
sys.path.insert(0, os.getcwd())

import pandas as pd
import wittgenstein as lw
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score

from c45_tree_utils import fit_chefboost_isolated

print("=" * 90)
print("STEP 7: REPEATED TRAIN/TEST EXPERIMENT ACROSS MULTIPLE RANDOM SEEDS")
print("=" * 90)

df = pd.read_csv("final_malaria_sdoh_matrix.csv")

continuous_feature = ['b19']
nominal_features = ['b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128', 'v129', 'v113', 'v116', 'hml20']
feature_cols = continuous_feature + nominal_features

X_raw = df[feature_cols].copy()
y_raw = df['hml32'].astype(int).values
cat_indices = list(range(1, len(feature_cols)))

with open(os.path.join("outputs", "models", "best_params.json"), encoding="utf-8") as f:
    best_params = json.load(f)
RF_PARAMS = best_params["random_forest"]
SVM_PARAMS = best_params["svm"]
print(f"Random Forest parameters (from fifth_step.py tuning): {RF_PARAMS}")
print(f"SVM parameters (from fifth_step.py tuning): {SVM_PARAMS}")


def get_preprocessor():
    return ColumnTransformer(transformers=[
        ('num', StandardScaler(), continuous_feature),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features)
    ])


def to_sklearn(X):
    X = X.copy()
    X['b19'] = X['b19'].astype(float)
    for col in nominal_features:
        X[col] = X[col].astype(int).astype(str)
    return X


def to_ripper(X):
    X = X.copy()
    X['b19'] = X['b19'].astype(int)
    for col in nominal_features:
        X[col] = X[col].astype(int).astype(str).astype(object)
    return X


SEEDS = [10, 20, 30, 42, 50, 60, 70, 80, 90, 100]
METRICS = ["Accuracy", "Sensitivity", "Specificity", "Precision_Pos", "F1_Pos", "Macro_F1", "ROC_AUC", "PR_AUC"]


def eval_metrics(y_true, y_pred, y_prob, constant_score=False):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn) * 100.0
    sens = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
    prec_pos = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    f1_pos = 2 * prec_pos * sens / (prec_pos + sens) if (prec_pos + sens) > 0 else 0.0
    prec_neg = tn / (tn + fn) * 100.0 if (tn + fn) > 0 else 0.0
    f1_neg = 2 * prec_neg * spec / (prec_neg + spec) if (prec_neg + spec) > 0 else 0.0
    if constant_score:
        roc_auc, pr_auc = 0.5, float(np.mean(y_true))
    else:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    return {"Accuracy": acc, "Sensitivity": sens, "Specificity": spec, "Precision_Pos": prec_pos,
            "F1_Pos": f1_pos, "Macro_F1": (f1_pos + f1_neg) / 2.0, "ROC_AUC": roc_auc, "PR_AUC": pr_auc}


MODELS = ["Majority Class (Zero-R)", "Logistic Regression", "C4.5 (ChefBoost)", "RIPPER (Wittgenstein)",
          "Random Forest", "Support Vector Machine"]
models_results = {m: {k: [] for k in METRICS} for m in MODELS}
per_seed_rows = []

print(f"\nExecuting stratified 80/20 split across {len(SEEDS)} seeds; SMOTE-NC on each training split only.\n")

for i, seed in enumerate(SEEDS, 1):
    print(f"Seed [{i:02d}/{len(SEEDS)}]: {seed} ...", flush=True)
    X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=seed, stratify=y_raw)

    # Same SMOTE-NC input format as third_step.py: b19 int, nominal codes as str
    X_train_fmt = X_train_raw.copy()
    X_train_fmt['b19'] = X_train_fmt['b19'].astype(int)
    for col in nominal_features:
        X_train_fmt[col] = X_train_fmt[col].astype(int).astype(str)
    smote = SMOTENC(categorical_features=cat_indices, random_state=seed, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_fmt, y_train_raw)
    y_train_res = np.asarray(y_train_res).astype(int)

    X_tr_sk, X_te_sk = to_sklearn(X_train_res), to_sklearn(X_test_raw)
    outputs = {}

    # Zero-R
    outputs["Majority Class (Zero-R)"] = (np.zeros_like(y_test), np.full(len(y_test), 0.5), True)

    # Logistic Regression
    lr = Pipeline([('prep', get_preprocessor()),
                   ('clf', LogisticRegression(C=1.0, random_state=seed, max_iter=1000))]).fit(X_tr_sk, y_train_res)
    outputs["Logistic Regression"] = (lr.predict(X_te_sk), lr.predict_proba(X_te_sk)[:, 1], False)

    # C4.5 (ChefBoost), fitted in a temporary directory so outputs/rules/rules.py is untouched
    tree = fit_chefboost_isolated(X_train_res, y_train_res)
    tree.fit_leaf_scores(X_train_res, y_train_res)
    outputs["C4.5 (ChefBoost)"] = (tree.predict(X_test_raw), tree.predict_score(X_test_raw), False)

    # RIPPER
    rip = lw.RIPPER(random_state=seed)
    rip.fit(to_ripper(X_train_res), y_train_res, pos_class=1)
    rip_pred = np.array([1 if p else 0 for p in rip.predict(to_ripper(X_test_raw))])
    outputs["RIPPER (Wittgenstein)"] = (rip_pred, rip.predict_proba(to_ripper(X_test_raw))[:, 1], False)

    # Random Forest
    rf = Pipeline([('prep', get_preprocessor()),
                   ('clf', RandomForestClassifier(**RF_PARAMS, random_state=seed, n_jobs=1))]).fit(X_tr_sk, y_train_res)
    outputs["Random Forest"] = (rf.predict(X_te_sk), rf.predict_proba(X_te_sk)[:, 1], False)

    # SVM
    svm = Pipeline([('prep', get_preprocessor()),
                    ('clf', SVC(**SVM_PARAMS, probability=True, random_state=seed))]).fit(X_tr_sk, y_train_res)
    outputs["Support Vector Machine"] = (svm.predict(X_te_sk), svm.predict_proba(X_te_sk)[:, 1], False)

    for m in MODELS:
        pred, prob, const = outputs[m]
        res = eval_metrics(y_test, pred, prob, constant_score=const)
        for k in METRICS:
            models_results[m][k].append(res[k])
        per_seed_rows.append({"Seed": seed, "Model": m, **res})

print("\nAll seeds completed.\n")

summary_rows = []
print("=" * 160)
print(f"{'Model':<26} | {'Accuracy':<17} | {'Sensitivity':<17} | {'Specificity':<17} | {'Precision (pos)':<17} | "
      f"{'Macro F1':<17} | {'ROC-AUC':<16} | {'PR-AUC (AP)':<16}")
print("=" * 160)
for m in MODELS:
    row = {"Model": m}
    parts = []
    for k in ["Accuracy", "Sensitivity", "Specificity", "Precision_Pos", "Macro_F1", "ROC_AUC", "PR_AUC"]:
        vals = np.array(models_results[m][k])
        row[f"{k}_Mean"], row[f"{k}_Std"] = float(np.mean(vals)), float(np.std(vals, ddof=1))
        row[f"{k}_Min"], row[f"{k}_Max"] = float(np.min(vals)), float(np.max(vals))
        if "AUC" in k:
            parts.append(f"{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}".ljust(16))
        else:
            parts.append(f"{np.mean(vals):.2f}% ± {np.std(vals, ddof=1):.2f}%".ljust(17))
    print(f"{m:<26} | " + " | ".join(parts))
    summary_rows.append(row)
print("=" * 160)
print("Mean ± sample SD over 10 seeds. Zero-R ROC-AUC fixed at 0.5 and PR-AUC equals test prevalence.")
print("C4.5 ROC/PR use Laplace leaf scores from the ChefBoost tree grown for that seed.")

pd.DataFrame(summary_rows).to_csv("outputs/multiseed_stability_results.csv", index=False)
pd.DataFrame(per_seed_rows).to_csv("outputs/multiseed_stability_per_seed.csv", index=False)
print("\nExported outputs/multiseed_stability_results.csv and outputs/multiseed_stability_per_seed.csv")
