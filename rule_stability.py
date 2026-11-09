"""Rule stability across random partitions (diagnostic; not one of the eleven pipeline stages).

Refits C4.5 (ChefBoost) and RIPPER (Wittgenstein) on the same ten seeds, splits and SMOTE-NC settings as
step7_multiseed_stability.py and records which rules are induced. For each named C4.5 pathway (C1-C5) and each RIPPER rule
(R1, R2) it reports in how many seeds the rule recurs:
  exact     - the same conditions (for C4.5, the same path; for RIPPER, the same conditions including the age bin);
  contains  - a rule of the same class contains all of the rule's nominal conditions, possibly with further conditions.
A C4.5 leaf counts only if training records reach it.

It also computes Wilson 95% intervals for rule precision on the original training records and the test set, from
outputs/c45_pathway_coverage_support.csv and outputs/ripper_rule_coverage_support.csv. Resampled records are excluded because
synthetic records are not independent observations.

Writes outputs/rule_stability_summary.csv, outputs/rule_stability_per_seed.csv, outputs/rule_precision_wilson.csv and
outputs/logs/rule_stability.log.
"""
import io
import os
import sys
import time
from collections import Counter

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import wittgenstein as lw
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split

from c45_tree_utils import PATHWAY_DEFINITIONS, PATHWAY_NAMES, fit_chefboost_isolated

t0 = time.time()
LOG_PATH = os.path.join("outputs", "logs", "rule_stability.log")
out = []


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


df = pd.read_csv("final_malaria_sdoh_matrix.csv")
continuous_feature = ['b19']
nominal_features = ['b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128', 'v129', 'v113', 'v116', 'hml20']
feature_cols = continuous_feature + nominal_features
X_raw = df[feature_cols].copy()
y_raw = df['hml32'].astype(int).values
cat_indices = list(range(1, len(feature_cols)))
SEEDS = [10, 20, 30, 42, 50, 60, 70, 80, 90, 100]

RIPPER_RULES = {
    'R1': {('v113', '31'), ('v127', '34'), ('v116', '31'), ('b4', '2'), ('v129', '31'), ('hml20', '0')},
    'R2': {('v128', '21'), ('v129', '31'), ('b19', '14.0-17.0')},
}


def to_ripper(X):
    X = X.copy()
    X['b19'] = X['b19'].astype(int)
    for col in nominal_features:
        X[col] = X[col].astype(int).astype(str).astype(object)
    return X


def nominal(conds):
    return {(f, v) for f, op, v in conds if op == '=='}


PATH_KEYS = {n: [tuple(tuple(c) for c in path) for path in paths] for n, paths in PATHWAY_DEFINITIONS.items()}
PATH_NOM = {n: [nominal(path) for path in paths] for n, paths in PATHWAY_DEFINITIONS.items()}
PATH_CLASS = {n: ('Negative' if n == 'C1' else 'Positive') for n in PATHWAY_DEFINITIONS}

rows, c45_vars, c45_roots, rip_conds, rip_vars, rip_counts = [], Counter(), Counter(), Counter(), Counter(), []
leaf_counts, pos_counts = [], []

p("=" * 90)
p("RULE STABILITY ACROSS TEN RANDOM PARTITIONS (C4.5 AND RIPPER)")
p("=" * 90)
p(f"Seeds: {SEEDS}; stratified 80/20 split; SMOTE-NC (k=5) on the training split only, as in step7_multiseed_stability.py\n")

for seed in SEEDS:
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y_raw, test_size=0.20, random_state=seed, stratify=y_raw)
    fmt = X_tr.copy()
    fmt['b19'] = fmt['b19'].astype(int)
    for col in nominal_features:
        fmt[col] = fmt[col].astype(int).astype(str)
    X_res, y_res = SMOTENC(categorical_features=cat_indices, random_state=seed, k_neighbors=5).fit_resample(fmt, y_tr)
    y_res = np.asarray(y_res).astype(int)

    # C4.5
    tree = fit_chefboost_isolated(X_res, y_res)
    n_reach = tree.leaf_counts(X_res, y_res)['n']
    reached = [l for l in tree.leaves if n_reach.loc[l.id] > 0]
    pos = [l for l in reached if l.label == 'Positive']
    leaf_counts.append(len(reached)); pos_counts.append(len(pos))
    c45_roots[tree.leaves[0].path[0][0]] += 1
    seen = set()
    for l in pos:
        for f, op, v in l.path:
            seen.add(f)
    c45_vars.update(seen)
    row = {'Seed': seed, 'C45_leaves_reached': len(reached), 'C45_positive_leaves_reached': len(pos),
           'C45_root_feature': tree.leaves[0].path[0][0]}
    for name in PATHWAY_DEFINITIONS:
        cls = PATH_CLASS[name]
        same = [l for l in reached if l.label == cls]
        row[f'{name}_exact'] = any(tuple(tuple(c) for c in l.path) in PATH_KEYS[name] for l in same)
        row[f'{name}_contains'] = (row[f'{name}_exact'] if name == 'C1'
                                   else any(s <= nominal(l.path) for l in same for s in PATH_NOM[name]))

    # RIPPER
    rip = lw.RIPPER(random_state=seed)
    rip.fit(to_ripper(X_res), y_res, pos_class=1)
    rules = [{(c.feature, str(c.val).replace(' ', '')) for c in r.conds} for r in rip.ruleset_.rules]  # age bins print as '14.0 - 17.0'
    rip_counts.append(len(rules))
    row['RIPPER_rules'] = len(rules)
    row['RIPPER_ruleset'] = str(rip.ruleset_)
    for r in rules:
        rip_conds.update(r)
    rip_vars.update({f for r in rules for f, _ in r})
    for name, target in RIPPER_RULES.items():
        nom = {c for c in target if c[0] != 'b19'}
        row[f'{name}_exact'] = any(target == r for r in rules)
        row[f'{name}_contains'] = any(nom <= r for r in rules)
    rows.append(row)
    p(f"Seed {seed:>3}: C4.5 reached leaves {len(reached):>2} (positive {len(pos):>2}), root {row['C45_root_feature']}; "
      f"RIPPER rules {len(rules):>2}: {rip.ruleset_}")

per = pd.DataFrame(rows)
per.to_csv(os.path.join("outputs", "rule_stability_per_seed.csv"), index=False)

N = len(SEEDS)
seed42 = per[per.Seed == 42].iloc[0]
summary = []
for name in list(PATHWAY_DEFINITIONS) + list(RIPPER_RULES):
    desc = PATHWAY_NAMES.get(name, ' AND '.join(f'{f}={v}' for f, v in sorted(RIPPER_RULES.get(name, []))))
    summary.append({'Rule': name, 'Description': desc, 'Seeds': N, 'Exact_recurrence': int(per[f'{name}_exact'].sum()),
                    'Contains_nominal_conditions': int(per[f'{name}_contains'].sum()), 'Present_at_seed_42_exact': bool(seed42[f'{name}_exact'])})
summary = pd.DataFrame(summary)
summary.to_csv(os.path.join("outputs", "rule_stability_summary.csv"), index=False)

p("\n" + "=" * 90)
p(f"RECURRENCE OF THE REPORTED RULES ACROSS {N} SEEDS")
p("=" * 90)
for _, r in summary.iterrows():
    p(f"{r.Rule}: exact {r.Exact_recurrence}/{N}; contains its nominal conditions {r.Contains_nominal_conditions}/{N}; "
      f"exact at seed 42: {r.Present_at_seed_42_exact}  [{r.Description}]")
p(f"\nC4.5 leaves reached by training records: mean {np.mean(leaf_counts):.1f} (range {min(leaf_counts)}-{max(leaf_counts)}); "
  f"positive leaves: mean {np.mean(pos_counts):.1f} (range {min(pos_counts)}-{max(pos_counts)})")
p(f"C4.5 root split feature: {dict(c45_roots)}")
p(f"Variables on at least one reached positive C4.5 leaf (seeds of {N}): {dict(c45_vars.most_common())}")
p(f"RIPPER rule count: mean {np.mean(rip_counts):.1f} (range {min(rip_counts)}-{max(rip_counts)}); seeds with no rule: {sum(c == 0 for c in rip_counts)}")
p(f"Variables in at least one RIPPER rule (seeds of {N}): {dict(rip_vars.most_common())}")
p(f"Most frequent RIPPER conditions (rules across all seeds): {rip_conds.most_common(12)}")


# ------------------------------------------------------------------ Wilson intervals
def wilson(k, n, z=1.959963984540054):
    """Same formula as run_bootstrap_ci_final.py."""
    if n == 0:
        return (np.nan, np.nan)
    ph = k / n
    denom = 1 + z ** 2 / n
    centre = (ph + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(ph * (1 - ph) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


w = []
for fname in ['c45_pathway_coverage_support.csv', 'ripper_rule_coverage_support.csv']:
    d = pd.read_csv(os.path.join('outputs', fname))
    d = d[~d['Dataset'].str.contains('resampled')]
    d = d[~d['Rule'].str.contains('Default Rule')]
    for _, r in d.iterrows():
        k, n = int(r.TP), int(r.Support_n)
        lo, hi = wilson(k, n)
        w.append({'Rule': r.Rule, 'Dataset': r.Dataset, 'n': n, 'Correct': k, 'Precision_%': (k / n * 100 if n else np.nan),
                  'Wilson_low_%': lo, 'Wilson_high_%': hi})
w = pd.DataFrame(w)
w.to_csv(os.path.join("outputs", "rule_precision_wilson.csv"), index=False)
p("\n" + "=" * 90)
p("WILSON 95% INTERVALS FOR RULE PRECISION (ORIGINAL TRAINING RECORDS AND TEST SET)")
p("=" * 90)
for _, r in w.iterrows():
    ci = 'n/a (no records)' if r.n == 0 else f"{r['Precision_%']:.2f}% [{r['Wilson_low_%']:.1f}, {r['Wilson_high_%']:.1f}]"
    p(f"{r.Rule[:60]:<60} | {r.Dataset[:36]:<36} | {r.Correct}/{r.n} | {ci}")

elapsed = time.time() - t0
with open(LOG_PATH, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("EXECUTION LOG: Diagnostic: Rule Stability Across Random Partitions and Wilson Intervals for Rule Precision\n")
    f.write("Script: rule_stability.py\n")
    f.write(f"Python Runtime: {sys.version.split()[0]} ({sys.executable})\n")
    f.write(f"Execution Duration: {elapsed:.2f} seconds\n")
    f.write("Exit Code: 0\n")
    f.write("=" * 80 + "\n\n")
    f.write('\n'.join(out) + '\n')
