import sys
import io
import numpy as np

# Ensure UTF-8 stdout encoding
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)

# Load saved predictions
data = np.load("outputs/models/test_predictions.npz")
Y_test = data["Y_test"]
N = len(Y_test)

models = {
    "Majority Class (Zero-R Baseline)": {
        "pred": data["pred_zero"],
        "prob": data["prob_zero"],
        "is_zero_r": True,
        "profile": "Trivial majority assignment"
    },
    "Logistic Regression (Linear Baseline)": {
        "pred": data["pred_lr"],
        "prob": data["prob_lr"],
        "is_zero_r": False,
        "profile": "Symmetric error distribution"
    },
    "C4.5 Decision Tree (White-Box)": {
        "pred": data["pred_c45"],
        "prob": data["prob_c45"],
        "is_zero_r": False,
        "profile": "Minority recall prioritized; lower specificity"
    },
    "RIPPER Ruleset (White-Box)": {
        "pred": data["pred_ripper"],
        "prob": data["prob_ripper"],
        "is_zero_r": False,
        "profile": "Severe rule conservatism; 0% recall"
    },
    "Random Forest (Ensemble Benchmark)": {
        "pred": data["pred_rf"],
        "prob": data["prob_rf"],
        "is_zero_r": False,
        "profile": "Moderate minority detection"
    },
    "Support Vector Classifier": {
        "pred": data["pred_svm"],
        "prob": data["prob_svm"],
        "is_zero_r": False,
        "profile": "Optimal harmonic trade-off"
    }
}

def calc_all_metrics(y_true, y_pred, y_prob, is_zero_r=False):
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
        "acc": acc,
        "sens": sens,
        "prec_pos": prec_pos,
        "f1_pos": f1_pos,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "spec": spec,
        "prec_neg": prec_neg,
        "f1_neg": f1_neg,
        "macro_f1": macro_f1
    }

# Bootstrap B = 1000
B = 1000
np.random.seed(42)

bootstrap_stats = {}

for name, m in models.items():
    pred = m["pred"]
    prob = m["prob"]
    is_zr = m["is_zero_r"]
    
    # Point estimate
    point = calc_all_metrics(Y_test, pred, prob, is_zero_r=is_zr)
    
    boot_lists = {k: [] for k in point.keys()}
    b_count = 0
    while b_count < B:
        idx = np.random.choice(N, size=N, replace=True)
        y_b = Y_test[idx]
        if len(np.unique(y_b)) < 2:
            continue
        pred_b = pred[idx]
        prob_b = prob[idx]
        
        m_b = calc_all_metrics(y_b, pred_b, prob_b, is_zero_r=is_zr)
        for k in boot_lists:
            boot_lists[k].append(m_b[k])
        b_count += 1
        
    ci_dict = {}
    for k in point.keys():
        arr = np.array(boot_lists[k])
        lo = np.percentile(arr, 2.5)
        hi = np.percentile(arr, 97.5)
        ci_dict[k] = (point[k], lo, hi)
    
    bootstrap_stats[name] = ci_dict

print("Bootstrap evaluation completed successfully!\n")

# Print Panel A
print("="*120)
print("Table 4.8 Comparative Algorithmic Performance Evaluation Matrix (Clinical Test Partition, N=49)")
print("Panel A: Primary Target (Minority Positive Class) Performance & Global Discrimination")
print(f"{'Model Architecture':<38} | {'Global Accuracy (%) [95% CI]':<30} | {'Sensitivity (%) [95% CI]':<26} | {'Precision (Pos) (%) [95% CI]':<30} | {'F1-Score (Pos) (%) [95% CI]':<28} | {'ROC-AUC [95% CI]':<22} | {'PR-AUC [95% CI]':<22}")
print("-" * 200)

for name, st in bootstrap_stats.items():
    acc_pt, acc_lo, acc_hi = st['acc']
    sens_pt, sens_lo, sens_hi = st['sens']
    prec_p_pt, prec_p_lo, prec_p_hi = st['prec_pos']
    f1_p_pt, f1_p_lo, f1_p_hi = st['f1_pos']
    roc_pt, roc_lo, roc_hi = st['roc_auc']
    pr_pt, pr_lo, pr_hi = st['pr_auc']
    
    # Format Wilson for accuracy, sensitivity as in original table
    # Original table reported:
    # Zero-R: 81.63% [68.6, 90.0], Sens: 0.00% [0.0, 30.8]
    # Logistic: 71.43% [57.6, 82.2], Sens: 22.22% [6.3, 54.7]
    # C4.5: 57.14% [43.3, 69.9], Sens: 33.33% [12.1, 64.6]
    # RIPPER: 75.51% [61.9, 85.4], Sens: 0.00% [0.0, 30.8]
    # RF: 73.47% [59.7, 83.8], Sens: 22.22% [6.3, 54.7]
    # SVM: 75.51% [61.9, 85.4], Sens: 33.33% [12.1, 64.6]
    
    # Bootstrap CI for Precision (Pos), F1 (Pos), PR-AUC:
    prec_str = f"{prec_p_pt:.2f}% [{prec_p_lo:.1f}, {prec_p_hi:.1f}]"
    f1_p_str = f"{f1_p_pt:.2f}% [{f1_p_lo:.1f}, {f1_p_hi:.1f}]"
    if name == "Majority Class (Zero-R Baseline)":
        roc_str = "0.5000 [—]"
        pr_str = f"{pr_pt:.4f} [{pr_lo:.4f}, {pr_hi:.4f}]"
    else:
        roc_str = f"{roc_pt:.4f} [{roc_lo:.2f}, {roc_hi:.2f}]"
        pr_str = f"{pr_pt:.4f} [{pr_lo:.4f}, {pr_hi:.4f}]"
        
    print(f"{name:<38} | {acc_pt:.2f}% (boot [{acc_lo:.1f}, {acc_hi:.1f}]) | {sens_pt:.2f}% (boot [{sens_lo:.1f}, {sens_hi:.1f}]) | {prec_str:<30} | {f1_p_str:<28} | {roc_str:<22} | {pr_str:<22}")

print("\n" + "="*120)
print("Panel B: Non-Target (Majority Negative Class) Performance & Macro-Averaged Balance")
print(f"{'Model Architecture':<38} | {'Specificity (%) [95% CI]':<26} | {'Precision (Neg) (%) [95% CI]':<30} | {'F1-Score (Neg) (%) [95% CI]':<28} | {'Macro Avg F1 (%) [95% CI]':<26} | {'Class Balance Diagnostic Profile'}")
print("-" * 190)

for name, st in bootstrap_stats.items():
    spec_pt, spec_lo, spec_hi = st['spec']
    prec_n_pt, prec_n_lo, prec_n_hi = st['prec_neg']
    f1_n_pt, f1_n_lo, f1_n_hi = st['f1_neg']
    mf1_pt, mf1_lo, mf1_hi = st['macro_f1']
    prof = models[name]["profile"]
    
    prec_n_str = f"{prec_n_pt:.2f}% [{prec_n_lo:.1f}, {prec_n_hi:.1f}]"
    f1_n_str = f"{f1_n_pt:.2f}% [{f1_n_lo:.1f}, {f1_n_hi:.1f}]"
    mf1_str = f"{mf1_pt:.2f}% [{mf1_lo:.1f}, {mf1_hi:.1f}]"
    
    print(f"{name:<38} | {spec_pt:.2f}% (boot [{spec_lo:.1f}, {spec_hi:.1f}]) | {prec_n_str:<30} | {f1_n_str:<28} | {mf1_str:<26} | {prof}")
