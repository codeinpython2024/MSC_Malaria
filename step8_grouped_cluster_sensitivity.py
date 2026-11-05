import os
import sys
import io
import json
import numpy as np

# Ensure UTF-8 stdout encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
sys.path.insert(0, os.getcwd())

import pandas as pd
import wittgenstein as lw
from sklearn.model_selection import StratifiedKFold, GroupKFold
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
print("STEP 8: GROUPED HOUSEHOLD & CLUSTER SENSITIVITY ANALYSIS (5-FOLD CV)")
print("=" * 90)

df = pd.read_csv("final_malaria_sdoh_matrix.csv")
df['hh_id'] = df['hv001'].astype(str) + "_" + df['hv002'].astype(str)
clusters = df['hv001'].values
households = df['hh_id'].values

n_total = len(df)
n_pos = int(df['hml32'].sum())
print("Analytic Cohort Structure:")
print(f" - Total children: {n_total}")
print(f" - Microscopy Positive (hml32=1): {n_pos} ({n_pos / n_total * 100:.2f}%)")
print(f" - Unique Enumeration Areas (Clusters, hv001): {df['hv001'].nunique()}")
print(f" - Unique Households (hv001 + hv002): {df['hh_id'].nunique()}")
print(f" - Mean children per household: {n_total / df['hh_id'].nunique():.2f}")

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


METRICS = ["Accuracy", "Sensitivity", "Specificity", "Precision_Pos", "F1_Pos", "Macro_F1", "ROC_AUC", "PR_AUC"]
MODELS = ["Logistic Regression", "C4.5 (ChefBoost)", "RIPPER (Wittgenstein)", "Random Forest",
          "Support Vector Machine"]


def eval_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn) * 100.0
    sens = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
    prec_pos = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    f1_pos = 2 * prec_pos * sens / (prec_pos + sens) if (prec_pos + sens) > 0 else 0.0
    prec_neg = tn / (tn + fn) * 100.0 if (tn + fn) > 0 else 0.0
    f1_neg = 2 * prec_neg * spec / (prec_neg + spec) if (prec_neg + spec) > 0 else 0.0
    return {"Accuracy": acc, "Sensitivity": sens, "Specificity": spec, "Precision_Pos": prec_pos,
            "F1_Pos": f1_pos, "Macro_F1": (f1_pos + f1_neg) / 2.0,
            "ROC_AUC": roc_auc_score(y_true, y_prob), "PR_AUC": average_precision_score(y_true, y_prob)}


def run_cv_experiment(splits, scheme_name):
    print(f"\n--- Running 5-Fold Evaluation: {scheme_name} ---")
    results = {m: {k: [] for k in METRICS} for m in MODELS}
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        y_train_f, y_test_f = y_raw[train_idx], y_raw[test_idx]
        n_test_pos = int(np.sum(y_test_f == 1))
        if n_test_pos == 0 or n_test_pos == len(y_test_f):
            print(f"   Fold {fold}: test fold has a single class "
                  f"(n={len(y_test_f)}, positives={n_test_pos}); ROC/PR undefined -> fold excluded")
            continue
        print(f"   Fold {fold}: train n={len(train_idx)} ({int(y_train_f.sum())} pos), "
              f"test n={len(test_idx)} ({n_test_pos} pos)", flush=True)

        X_train_fmt = X_raw.iloc[train_idx].copy()
        X_train_fmt['b19'] = X_train_fmt['b19'].astype(int)
        for col in nominal_features:
            X_train_fmt[col] = X_train_fmt[col].astype(int).astype(str)
        X_test_f = X_raw.iloc[test_idx].copy()

        # SMOTE-NC on the training fold only (same settings as third_step.py).
        # A failure stops the run rather than silently training without oversampling.
        smote = SMOTENC(categorical_features=cat_indices, random_state=42, k_neighbors=5)
        X_train_res, y_train_res = smote.fit_resample(X_train_fmt, y_train_f)
        y_train_res = np.asarray(y_train_res).astype(int)
        X_tr_sk, X_te_sk = to_sklearn(X_train_res), to_sklearn(X_test_f)

        out = {}
        lr = Pipeline([('prep', get_preprocessor()),
                       ('clf', LogisticRegression(C=1.0, random_state=42, max_iter=1000))]).fit(X_tr_sk, y_train_res)
        out["Logistic Regression"] = (lr.predict(X_te_sk), lr.predict_proba(X_te_sk)[:, 1])

        tree = fit_chefboost_isolated(X_train_res, y_train_res)
        tree.fit_leaf_scores(X_train_res, y_train_res)
        out["C4.5 (ChefBoost)"] = (tree.predict(X_test_f), tree.predict_score(X_test_f))

        rip = lw.RIPPER(random_state=42)
        rip.fit(to_ripper(X_train_res), y_train_res, pos_class=1)
        out["RIPPER (Wittgenstein)"] = (np.array([1 if p else 0 for p in rip.predict(to_ripper(X_test_f))]),
                                        rip.predict_proba(to_ripper(X_test_f))[:, 1])

        rf = Pipeline([('prep', get_preprocessor()),
                       ('clf', RandomForestClassifier(**RF_PARAMS, random_state=42, n_jobs=1))]).fit(X_tr_sk, y_train_res)
        out["Random Forest"] = (rf.predict(X_te_sk), rf.predict_proba(X_te_sk)[:, 1])

        svm = Pipeline([('prep', get_preprocessor()),
                        ('clf', SVC(**SVM_PARAMS, probability=True, random_state=42))]).fit(X_tr_sk, y_train_res)
        out["Support Vector Machine"] = (svm.predict(X_te_sk), svm.predict_proba(X_te_sk)[:, 1])

        for m in MODELS:
            res = eval_metrics(y_test_f, *out[m])
            for k in METRICS:
                results[m][k].append(res[k])
            fold_rows.append({"Validation_Scheme": scheme_name, "Fold": fold, "Model": m,
                              "Test_N": len(test_idx), "Test_Pos": n_test_pos, **res})

    means = {m: {k: float(np.mean(v)) for k, v in metrics.items()} for m, metrics in results.items()}
    sds = {m: {k: float(np.std(v, ddof=1)) if len(v) > 1 else float('nan') for k, v in metrics.items()}
           for m, metrics in results.items()}
    n_folds = {m: len(results[m]["Accuracy"]) for m in MODELS}
    return means, sds, n_folds, fold_rows


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gkf = GroupKFold(n_splits=5)
schemes = [
    ("Scheme A: Stratified K-Fold (random rows)", list(skf.split(X_raw, y_raw))),
    ("Scheme B: Household-Grouped K-Fold", list(gkf.split(X_raw, y_raw, groups=households))),
    ("Scheme C: Cluster-Grouped K-Fold", list(gkf.split(X_raw, y_raw, groups=clusters))),
]

comparison_rows, all_fold_rows = [], []
summaries = []
for label, splits in schemes:
    means, sds, n_folds, fold_rows = run_cv_experiment(splits, label)
    summaries.append((label, means, sds, n_folds))
    all_fold_rows.extend(fold_rows)

print("\n" + "=" * 150)
print("SENSITIVITY ANALYSIS SUMMARY: RANDOM vs HOUSEHOLD-GROUPED vs CLUSTER-GROUPED (5-FOLD CV, mean over folds)")
print("=" * 150)
print(f"{'Validation scheme':<42} | {'Model':<24} | {'Folds':<5} | {'Accuracy':<9} | {'Sens':<8} | "
      f"{'Spec':<8} | {'Macro F1':<9} | {'ROC-AUC':<16} | {'PR-AUC (AP)':<9}")
print("-" * 150)
for label, means, sds, n_folds in summaries:
    for m in MODELS:
        mm, ss = means[m], sds[m]
        print(f"{label:<42} | {m:<24} | {n_folds[m]:<5} | {mm['Accuracy']:<8.2f}% | {mm['Sensitivity']:<7.2f}% | "
              f"{mm['Specificity']:<7.2f}% | {mm['Macro_F1']:<8.2f}% | "
              f"{mm['ROC_AUC']:.4f} ± {ss['ROC_AUC']:.4f} | {mm['PR_AUC']:<9.4f}")
        row = {"Validation_Scheme": label, "Model": m, "Folds_Used": n_folds[m]}
        row.update(mm)
        row.update({f"{k}_SD": v for k, v in ss.items()})
        comparison_rows.append(row)
print("=" * 150)

pd.DataFrame(comparison_rows).to_csv("outputs/grouped_cluster_sensitivity_results.csv", index=False)
pd.DataFrame(all_fold_rows).to_csv("outputs/grouped_cluster_sensitivity_per_fold.csv", index=False)
print("\nExported outputs/grouped_cluster_sensitivity_results.csv and outputs/grouped_cluster_sensitivity_per_fold.csv")
