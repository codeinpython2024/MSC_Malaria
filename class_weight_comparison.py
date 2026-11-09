"""SMOTE-NC versus class weighting (diagnostic; not one of the eleven pipeline stages).

Trains logistic regression, Random Forest and SVM with class_weight='balanced' and no oversampling, on the same partitions and with
the same preprocessing and search grids as fifth_step.py and sixth_step.py, and compares them with the SMOTE-NC results already
reported. C4.5 (ChefBoost) and RIPPER (Wittgenstein) are excluded because those libraries have no class-weight option.

Primary split: X_train_original.csv (N = 194) and X_test.csv (N = 49). Random Forest and SVM are tuned by 5-fold stratified grid search
on the 194 training records, scored by macro F1. Stability: the same ten seeds as step7_multiseed_stability.py, with the selected
hyperparameters held fixed. No existing output file is changed.

The class-weighted SVM's ROC-AUC and PR-AUC use decision_function scores: with class weighting, SVC's internal probability calibration
(predict_proba) was almost perfectly inverted relative to decision_function (correlation -0.995 on the primary test set), giving a spurious
ROC-AUC of 0.3611 against 0.6389. Predicted classes come from predict() and are unaffected.

Writes outputs/class_weight_comparison.csv, outputs/class_weight_multiseed.csv and outputs/logs/class_weight_comparison.log.
"""
import io
import os
import sys
import time

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

t0 = time.time()
LOG_PATH = os.path.join("outputs", "logs", "class_weight_comparison.log")
out = []


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


continuous_feature = ['b19']
nominal_features = ['b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128', 'v129', 'v113', 'v116', 'hml20']
feature_cols = continuous_feature + nominal_features


def fmt(X):
    X = X[feature_cols].copy()
    X['b19'] = X['b19'].astype(float)
    for col in nominal_features:
        X[col] = X[col].astype(int).astype(str)
    return X


def prep():
    return ColumnTransformer(transformers=[('num', StandardScaler(), continuous_feature),
                                           ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features)])


def wilson(k, n, z=1.959963984540054):
    """Same formula as run_bootstrap_ci_final.py."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    denom = 1 + z ** 2 / n
    centre = (ph + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


def metrics(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    f1p = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    f1n = 2 * npv * spec / (npv + spec) if npv + spec else 0.0
    return {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp), 'Accuracy': (tp + tn) / len(y), 'Sensitivity': sens, 'Specificity': spec,
            'Precision': prec, 'Macro_F1': (f1p + f1n) / 2, 'ROC_AUC': roc_auc_score(y, prob), 'PR_AUC_AP': average_precision_score(y, prob)}


def score(model, X):
    """Ranking score: decision_function for SVC (see module docstring), positive-class probability otherwise."""
    return model.decision_function(X) if isinstance(model.named_steps['classifier'], SVC) else model.predict_proba(X)[:, 1]


# ------------------------------------------------------------------ primary split
X_tr = fmt(pd.read_csv("X_train_original.csv")); y_tr = pd.read_csv("Y_train_original.csv")['hml32'].astype(int).values
X_te = fmt(pd.read_csv("X_test.csv")); y_te = pd.read_csv("Y_test.csv")['hml32'].astype(int).values
p("=" * 90)
p("SMOTE-NC VERSUS CLASS WEIGHTING (LOGISTIC REGRESSION, RANDOM FOREST, SVM)")
p("=" * 90)
p(f"Training records: {len(X_tr)} ({int(y_tr.sum())} positive); test records: {len(X_te)} ({int(y_te.sum())} positive). No oversampling.")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf_grid = {'classifier__n_estimators': [50, 100, 200, 300], 'classifier__max_depth': [None, 5, 10, 15], 'classifier__min_samples_split': [2, 5, 10]}
svm_grid = {'classifier__C': [0.1, 1, 10, 100], 'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1], 'classifier__kernel': ['rbf', 'linear', 'poly']}

rf_gs = GridSearchCV(Pipeline([('preprocessor', prep()), ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))]),
                     rf_grid, cv=cv, scoring='f1_macro', n_jobs=-1, refit=True).fit(X_tr, y_tr)
svm_gs = GridSearchCV(Pipeline([('preprocessor', prep()), ('classifier', SVC(probability=True, random_state=42, class_weight='balanced'))]),
                      svm_grid, cv=cv, scoring='f1_macro', n_jobs=-1, refit=True).fit(X_tr, y_tr)
lr = Pipeline([('preprocessor', prep()), ('classifier', LogisticRegression(C=1.0, random_state=42, max_iter=1000, class_weight='balanced'))]).fit(X_tr, y_tr)
strip = lambda d: {k.replace('classifier__', ''): v for k, v in d.items()}
RF_P, SVM_P = strip(rf_gs.best_params_), strip(svm_gs.best_params_)
p(f"Random Forest selected: {RF_P} (cross-validated macro F1 {rf_gs.best_score_:.4f})")
p(f"SVM selected: {SVM_P} (cross-validated macro F1 {svm_gs.best_score_:.4f})")
svm_probs, svm_decisions = svm_gs.best_estimator_.predict_proba(X_te)[:, 1], svm_gs.best_estimator_.decision_function(X_te)
p(f"Class-weighted SVM, test set: correlation of predict_proba with decision_function {np.corrcoef(svm_probs, svm_decisions)[0, 1]:.3f}; "
  f"ROC-AUC {roc_auc_score(y_te, svm_probs):.4f} with predict_proba, {roc_auc_score(y_te, svm_decisions):.4f} with decision_function")

models = {'Logistic Regression': lr, 'Random Forest': rf_gs.best_estimator_, 'SVM': svm_gs.best_estimator_}
smote = pd.read_csv(os.path.join('outputs', 'test_point_metrics.csv')).set_index('Model')
table = []
for name, m in models.items():
    r = metrics(y_te, m.predict(X_te), score(m, X_te))
    r.update({'Model': name, 'Method': 'class_weight=balanced'})
    table.append(r)
    s = smote.loc[name]
    table.append({'Model': name, 'Method': 'SMOTE-NC (reported)', 'TN': int(s.TN), 'FP': int(s.FP), 'FN': int(s.FN), 'TP': int(s.TP), 'Accuracy': s.Accuracy, 'Sensitivity': s.Sensitivity,
                  'Specificity': s.Specificity, 'Precision': s.Precision, 'Macro_F1': s.Macro_F1, 'ROC_AUC': s.ROC_AUC, 'PR_AUC_AP': s.PR_AUC_AP})
table = pd.DataFrame(table)
table.to_csv(os.path.join('outputs', 'class_weight_comparison.csv'), index=False)
p("\nPRIMARY TEST SET (N = 49, 9 positive)")
p(f"{'Model':<20} {'Method':<23} {'TN':>3} {'FP':>3} {'FN':>3} {'TP':>3} {'Acc%':>7} {'Sens% [95% CI]':>23} {'Spec%':>7} {'Prec%':>7} {'MacroF1':>8} {'ROC-AUC':>8} {'PR-AUC':>7}")
for _, r in table.iterrows():
    lo, hi = wilson(r.TP, r.TP + r.FN)
    sens = f"{r.Sensitivity * 100:.2f} [{lo:.1f}, {hi:.1f}]"
    p(f"{r.Model:<20} {r.Method:<23} {r.TN:>3} {r.FP:>3} {r.FN:>3} {r.TP:>3} {r.Accuracy * 100:>7.2f} {sens:>23} {r.Specificity * 100:>7.2f} {r.Precision * 100:>7.2f} {r.Macro_F1:>8.4f} {r.ROC_AUC:>8.4f} {r.PR_AUC_AP:>7.4f}")

# ------------------------------------------------------------------ ten seeds
df = pd.read_csv("final_malaria_sdoh_matrix.csv")
X_raw = df[feature_cols].copy(); y_raw = df['hml32'].astype(int).values
SEEDS = [10, 20, 30, 42, 50, 60, 70, 80, 90, 100]
rows = []
for seed in SEEDS:
    A, B, ya, yb = train_test_split(X_raw, y_raw, test_size=0.20, random_state=seed, stratify=y_raw)
    A, B = fmt(A), fmt(B)
    fits = {'Logistic Regression': LogisticRegression(C=1.0, random_state=seed, max_iter=1000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(**RF_P, random_state=seed, n_jobs=1, class_weight='balanced'),
            'SVM': SVC(**SVM_P, probability=True, random_state=seed, class_weight='balanced')}
    for name, clf in fits.items():
        m = Pipeline([('preprocessor', prep()), ('classifier', clf)]).fit(A, ya)
        rows.append({'Seed': seed, 'Model': name, **metrics(yb, m.predict(B), score(m, B))})
per = pd.DataFrame(rows)
sm = pd.read_csv(os.path.join('outputs', 'multiseed_stability_results.csv')).set_index('Model')
NAME = {'Logistic Regression': 'Logistic Regression', 'Random Forest': 'Random Forest', 'SVM': 'Support Vector Machine'}
res = []
p("\nTEN RANDOM PARTITIONS (mean +/- SD)")
p(f"{'Model':<20} {'Method':<23} {'Sensitivity %':>16} {'Specificity %':>16} {'Macro F1 %':>15} {'ROC-AUC':>16} {'PR-AUC':>16}")
for name in models:
    g = per[per.Model == name]
    cw = {'Model': name, 'Method': 'class_weight=balanced'}
    for k, c in [('Sensitivity', 100), ('Specificity', 100), ('Macro_F1', 100), ('ROC_AUC', 1), ('PR_AUC_AP', 1)]:
        cw[k + '_Mean'] = g[k].mean() * c; cw[k + '_SD'] = g[k].std(ddof=1) * c
    s = sm.loc[NAME[name]]
    sn = {'Model': name, 'Method': 'SMOTE-NC (reported)'}
    for k, src in [('Sensitivity', 'Sensitivity'), ('Specificity', 'Specificity'), ('Macro_F1', 'Macro_F1'), ('ROC_AUC', 'ROC_AUC'), ('PR_AUC_AP', 'PR_AUC')]:
        sn[k + '_Mean'] = s[src + '_Mean']; sn[k + '_SD'] = s[src + '_Std']
    for r in (cw, sn):
        res.append(r)
        p(f"{r['Model']:<20} {r['Method']:<23} {r['Sensitivity_Mean']:>8.2f} +/-{r['Sensitivity_SD']:>5.2f} {r['Specificity_Mean']:>8.2f} +/-{r['Specificity_SD']:>5.2f} "
          f"{r['Macro_F1_Mean']:>7.2f} +/-{r['Macro_F1_SD']:>5.2f} {r['ROC_AUC_Mean']:>8.4f} +/-{r['ROC_AUC_SD']:.3f} {r['PR_AUC_AP_Mean']:>8.4f} +/-{r['PR_AUC_AP_SD']:.3f}")
pd.DataFrame(res).to_csv(os.path.join('outputs', 'class_weight_multiseed.csv'), index=False)

elapsed = time.time() - t0
with open(LOG_PATH, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("EXECUTION LOG: Diagnostic: SMOTE-NC Versus Class Weighting (Logistic Regression, Random Forest, SVM)\n")
    f.write("Script: class_weight_comparison.py\n")
    f.write(f"Python Runtime: {sys.version.split()[0]} ({sys.executable})\n")
    f.write(f"Execution Duration: {elapsed:.2f} seconds\n")
    f.write("Exit Code: 0\n")
    f.write("=" * 80 + "\n\n")
    f.write('\n'.join(out) + '\n')
