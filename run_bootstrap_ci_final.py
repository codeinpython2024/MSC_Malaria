"""
Stage 11: uncertainty for Table 4.8.

Reads the real test-set predictions and scores written by sixth_step.py
(outputs/models/test_predictions.npz) and reports, for every model:
  * point estimates,
  * Wilson score 95% intervals for the proportions (accuracy, sensitivity,
    specificity, positive and negative predictive value), and
  * non-parametric bootstrap 95% percentile intervals (B = 1000, seed 42) for
    every metric, including F1, macro F1, ROC-AUC and PR-AUC.

PR-AUC is average precision (sklearn.metrics.average_precision_score).
Nothing in this script is typed in by hand.
"""
import io
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

data = np.load(os.path.join("outputs", "models", "test_predictions.npz"))
Y_test = data["Y_test"].astype(int)
N = len(Y_test)

models = {
    "Majority Class (Zero-R)": ("zero", True),
    "Logistic Regression": ("lr", False),
    "C4.5 Decision Tree": ("c45", False),
    "RIPPER Ruleset": ("ripper", False),
    "Random Forest": ("rf", False),
    "SVM (RBF)": ("svm", False),
}


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


def calc_all_metrics(y_true, y_pred, y_prob, constant_score):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn) * 100.0
    sens = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) * 100.0 if (tn + fp) > 0 else 0.0
    prec_pos = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    prec_neg = tn / (tn + fn) * 100.0 if (tn + fn) > 0 else 0.0
    f1_pos = 2 * prec_pos * sens / (prec_pos + sens) if (prec_pos + sens) > 0 else 0.0
    f1_neg = 2 * prec_neg * spec / (prec_neg + spec) if (prec_neg + spec) > 0 else 0.0
    if constant_score:
        roc_auc, pr_auc = 0.5, float(np.mean(y_true))
    else:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    return {"acc": acc, "sens": sens, "spec": spec, "prec_pos": prec_pos, "prec_neg": prec_neg,
            "f1_pos": f1_pos, "f1_neg": f1_neg, "macro_f1": (f1_pos + f1_neg) / 2.0,
            "roc_auc": roc_auc, "pr_auc": pr_auc,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


B = 1000
rng = np.random.RandomState(42)
# One shared set of bootstrap resamples so all models are compared on identical samples
boot_idx = []
while len(boot_idx) < B:
    idx = rng.choice(N, size=N, replace=True)
    if len(np.unique(Y_test[idx])) == 2:
        boot_idx.append(idx)

rows = []
for name, (key, constant) in models.items():
    pred = data[f"pred_{key}"].astype(int)
    prob = data[f"prob_{key}"].astype(float)
    point = calc_all_metrics(Y_test, pred, prob, constant)
    boots = {k: [] for k in ["acc", "sens", "spec", "prec_pos", "prec_neg", "f1_pos", "f1_neg",
                             "macro_f1", "roc_auc", "pr_auc"]}
    for idx in boot_idx:
        m = calc_all_metrics(Y_test[idx], pred[idx], prob[idx], constant)
        for k in boots:
            boots[k].append(m[k])
    row = {"Model": name, "TP": point["tp"], "FP": point["fp"], "FN": point["fn"], "TN": point["tn"]}
    for k in boots:
        row[k] = point[k]
        row[f"{k}_boot_lo"] = float(np.percentile(boots[k], 2.5))
        row[f"{k}_boot_hi"] = float(np.percentile(boots[k], 97.5))
    tp, fp, fn, tn = point["tp"], point["fp"], point["fn"], point["tn"]
    row["acc_wilson_lo"], row["acc_wilson_hi"] = wilson(tp + tn, N)
    row["sens_wilson_lo"], row["sens_wilson_hi"] = wilson(tp, tp + fn)
    row["spec_wilson_lo"], row["spec_wilson_hi"] = wilson(tn, tn + fp)
    row["prec_pos_wilson_lo"], row["prec_pos_wilson_hi"] = wilson(tp, tp + fp)
    row["prec_neg_wilson_lo"], row["prec_neg_wilson_hi"] = wilson(tn, tn + fn)
    rows.append(row)

df = pd.DataFrame(rows)
os.makedirs("outputs", exist_ok=True)
df.to_csv(os.path.join("outputs", "table_4_8_bootstrap_ci.csv"), index=False)

print(f"Bootstrap evaluation completed: B = {B} resamples (seed 42), N = {N}, "
      f"{int(Y_test.sum())} positive; resamples lacking either class were redrawn.\n")


def pct(r, k, ci):
    lo, hi = r[f"{k}_{ci}_lo"], r[f"{k}_{ci}_hi"]
    if np.isnan(lo) or np.isnan(hi):
        # e.g. Zero-R precision: 0 positive predictions, so the proportion is undefined
        return f"{r[k]:.2f}% [— undefined, 0 predicted positive]"
    return f"{r[k]:.2f}% [{lo:.1f}, {hi:.1f}]"


def num(r, k):
    return f"{r[k]:.4f} [{r[k + '_boot_lo']:.2f}, {r[k + '_boot_hi']:.2f}]"


print("=" * 150)
print("Table 4.8 Panel A: positive class and discrimination (test partition, N = 49)")
print("Accuracy, sensitivity, precision: Wilson 95% CI. F1, ROC-AUC, PR-AUC: bootstrap 95% CI.")
print("-" * 150)
print(f"{'Model':<26} | {'Accuracy':<22} | {'Sensitivity':<22} | {'Precision (pos)':<22} | "
      f"{'F1 (pos)':<22} | {'ROC-AUC':<20} | {'PR-AUC (AP)':<20}")
for _, r in df.iterrows():
    roc = "0.5000 [—]" if r["Model"].startswith("Majority") else num(r, "roc_auc")
    print(f"{r['Model']:<26} | {pct(r, 'acc', 'wilson'):<22} | {pct(r, 'sens', 'wilson'):<22} | "
          f"{pct(r, 'prec_pos', 'wilson'):<22} | {pct(r, 'f1_pos', 'boot'):<22} | {roc:<20} | "
          f"{num(r, 'pr_auc'):<20}")

print("\n" + "=" * 150)
print("Table 4.8 Panel B: negative class and macro balance")
print("Specificity, NPV: Wilson 95% CI. F1 (neg), macro F1: bootstrap 95% CI.")
print("-" * 150)
print(f"{'Model':<26} | {'Specificity':<22} | {'NPV':<22} | {'F1 (neg)':<22} | {'Macro F1':<22}")
for _, r in df.iterrows():
    print(f"{r['Model']:<26} | {pct(r, 'spec', 'wilson'):<22} | {pct(r, 'prec_neg', 'wilson'):<22} | "
          f"{pct(r, 'f1_neg', 'boot'):<22} | {pct(r, 'macro_f1', 'boot'):<22}")

# C4.5 hard-label discrimination, for the table note
if "prob_c45_hard" in data.files:
    hard = data["prob_c45_hard"].astype(float)
    print(f"\nC4.5 hard-label (single operating point): ROC-AUC = {roc_auc_score(Y_test, hard):.4f}, "
          f"AP = {average_precision_score(Y_test, hard):.4f}")
print(f"No-skill PR-AUC baseline (test prevalence) = {Y_test.mean():.4f}")
print("\nFull table written to outputs/table_4_8_bootstrap_ci.csv")
