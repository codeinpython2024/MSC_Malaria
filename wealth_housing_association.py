"""Association between wealth quintile and housing and water variables (diagnostic; not one of the eleven pipeline stages).

Measures how strongly household wealth quintile (v190) is associated with floor (v127), wall (v128) and roof (v129) material,
drinking-water source (v113) and toilet facility (v116) in the analytic cohort. It is unweighted, and it describes this cohort only.

Two units of analysis are reported. The child level uses all 243 children. Wealth and housing are household attributes, so children
in the same household repeat the same values; the household level therefore keeps one record per household.

For each pair the script reports Cramer's V (Cramer, 1946), the bias-corrected V (Bergsma, 2013), which removes the upward bias of V
in tables with many sparse cells, and a permutation p-value (10,000 permutations of v190, seed 42), which does not rely on the
chi-square approximation that sparse tables violate.

Writes outputs/wealth_housing_association.csv and outputs/logs/wealth_housing_association.log.
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
from scipy.stats import chi2_contingency

t0 = time.time()
LOG_PATH = os.path.join("outputs", "logs", "wealth_housing_association.log")
OUT_PATH = os.path.join("outputs", "wealth_housing_association.csv")
SEED, B = 42, 10000
VARS = {'v128': 'Wall material', 'v127': 'Floor material', 'v129': 'Roof material', 'v113': 'Drinking-water source', 'v116': 'Toilet facility'}
out = []


def p(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


def chi2(x, y):
    return chi2_contingency(pd.crosstab(x, y).values, correction=False)[0]


def cramers(x, y):
    """Cramer's V and the bias-corrected V of Bergsma (2013)."""
    table = pd.crosstab(x, y).values
    n, r, c = table.sum(), table.shape[0], table.shape[1]
    phi2 = chi2_contingency(table, correction=False)[0] / n
    v = np.sqrt(phi2 / min(r - 1, c - 1))
    phi2_c = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
    r_c, c_c = r - (r - 1) ** 2 / (n - 1), c - (c - 1) ** 2 / (n - 1)
    v_c = np.sqrt(phi2_c / min(r_c - 1, c_c - 1))
    return v, v_c, r, c, int(n)


def permutation_p(x, y, rng):
    observed = chi2(x, y)
    values = x.values.copy()
    hits = 0
    for _ in range(B):
        rng.shuffle(values)
        if chi2(pd.Series(values, index=x.index), y) >= observed - 1e-9:
            hits += 1
    return (hits + 1) / (B + 1)


df = pd.read_csv("final_malaria_sdoh_matrix.csv")
keys = ['hv001', 'hv002']
household_columns = ['v190'] + list(VARS)
varying = df.groupby(keys)[household_columns].nunique().max()
assert (varying == 1).all(), f"household variable differs within a household: {varying.to_dict()}"
households = df.drop_duplicates(keys)
assert len(df) == 243 and len(households) == 139, (len(df), len(households))

p("=" * 90)
p("ASSOCIATION BETWEEN WEALTH QUINTILE (v190) AND HOUSING AND WATER VARIABLES")
p("=" * 90)
p(f"Children: {len(df)}; households: {len(households)}. Unweighted. Each variable was constant within every household.")
p(f"Permutation p-values: {B} permutations of v190, seed {SEED}.\n")
p(f"{'Variable':<24} {'Unit':<10} {'n':>4} {'Table':>6} {'V':>7} {'Corrected V':>12} {'Perm. p':>9}")

rows = []
for level, data in (('Household', households), ('Child', df)):
    rng = np.random.default_rng(SEED)
    for var, label in VARS.items():
        x, y = data['v190'].astype(int), data[var].astype(int)
        v, v_c, r, c, n = cramers(x, y)
        perm = permutation_p(x, y, rng)
        rows.append({'Variable': var, 'Description': label, 'Unit': level, 'n': n, 'Rows': r, 'Columns': c, 'Cramers_V': v,
                     'Bias_corrected_V': v_c, 'Permutation_p': perm})
        p(f"{label:<24} {level:<10} {n:>4} {f'{r}x{c}':>6} {v:>7.3f} {v_c:>12.3f} {perm:>9.4f}")

pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

elapsed = time.time() - t0
with open(LOG_PATH, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("EXECUTION LOG: Diagnostic: Association Between Wealth Quintile and Housing and Water Variables\n")
    f.write("Script: wealth_housing_association.py\n")
    f.write(f"Python Runtime: {sys.version.split()[0]} ({sys.executable})\n")
    f.write(f"Execution Duration: {elapsed:.2f} seconds\n")
    f.write("Exit Code: 0\n")
    f.write("=" * 80 + "\n\n")
    f.write('\n'.join(out) + '\n')
