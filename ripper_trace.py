"""RIPPER induction trace (diagnostic; not one of the eleven pipeline stages).

Re-fits the RIPPER model of fourth_step.py (Wittgenstein, library defaults, random_state=42,
pos_class=1) on the same resampled training data with Wittgenstein's verbose output switched on,
and writes the trace to outputs/logs/ripper_trace.log.

It records which phase of RIPPER produced the final two-rule set:
  A. grow/prune rules one at a time until the description length (DL) exceeds the minimum
     DL so far by more than dl_allowance (64) bits, or no positives remain;
  B. optimisation passes (k = 2; stops early if a pass changes nothing);
  C. grow rules for any positives still uncovered (same stopping rule as A);
  D. delete rules that increase the total description length.
Wittgenstein's RIPPER has no "stop when a rule's error rate exceeds 50%" condition; that check
exists only in its separate IREP class (wittgenstein/irep.py), which this pipeline does not use.

The script asserts that the traced fit gives the same rule set as a silent fit, so the trace
describes the model reported in the thesis.
"""
import contextlib
import io
import os
import re
import sys
import time

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

import pandas as pd
import wittgenstein as lw

LOG_PATH = os.path.join("outputs", "logs", "ripper_trace.log")
EXPECTED = "[[v113=31^v127=34^v116=31^b4=2^v129=31^hml20=0] V [v128=21^v129=31^b19=14.0-17.0]]"
NOMINAL = ['b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128', 'v129', 'v113', 'v116', 'hml20']

t0 = time.time()

# Same data and dtype handling as fourth_step.py
X_train = pd.read_csv("X_train_resampled.csv")
Y_train = pd.read_csv("Y_train_resampled.csv")['hml32'].astype(int)
X_train['b19'] = X_train['b19'].astype(int)
for col in NOMINAL:
    X_train[col] = X_train[col].astype(int).astype(str).astype(object)

# Silent fit, exactly as in fourth_step.py
silent = lw.RIPPER(random_state=42)
silent.fit(X_train, Y_train, pos_class=1)

# Traced fit: identical apart from verbosity
buf = io.StringIO()
traced = lw.RIPPER(random_state=42, verbosity=3)
with contextlib.redirect_stdout(buf):
    traced.fit(X_train, Y_train, pos_class=1)
trace = buf.getvalue()

assert str(silent.ruleset_) == str(traced.ruleset_), "verbose fit differs from silent fit"
assert str(silent.ruleset_) == EXPECTED, f"rule set differs from the one reported: {silent.ruleset_}"


def block(marker):
    """Rule lines printed after a marker line such as 'GREW INITIAL RULESET:'."""
    lines = trace.splitlines()
    i = next(k for k, line in enumerate(lines) if line.strip().startswith(marker))
    out = []
    for line in lines[i + 1:]:
        s = line.strip()
        if not s.startswith('['):
            break
        out.append(s.rstrip(' V').strip('[]').replace(' ', ''))
    return out


initial = block("GREW INITIAL RULESET")
grown_final = block("GREW FINAL RULES")
final = [r.strip('[]') for r in str(silent.ruleset_).strip('[]').split('] V [')]
halts = re.findall(r"dl-halt: diff ([0-9.]+) exceeds allowance \(([0-9]+)\)", trace)
opt = re.search(r"Halting optimization at iteration k=(\d+)", trace)
pos, neg = int((Y_train == 1).sum()), int((Y_train == 0).sum())

summary = io.StringIO()
p = lambda *a: print(*a, file=summary)
p("=" * 90)
p("RIPPER INDUCTION TRACE (Wittgenstein %s, RIPPER(random_state=42), pos_class=1)" % lw.__version__)
p("=" * 90)
p(f"Training data: X_train_resampled.csv, {len(X_train)} records ({pos} positive, {neg} negative)")
p(f"Settings: k={silent.k}, prune_size={silent.prune_size}, dl_allowance={silent.dl_allowance}, "
  f"n_discretize_bins={silent.n_discretize_bins}")
p(f"Traced fit reproduces the silent fit and the reported rule set: True")
p("")
p(f"Phase A (grow initial rule set): {len(initial)} rules grown; stopped by the description-length "
  f"limit (diff {halts[0][0]} bits > {halts[0][1]})" if halts else
  f"Phase A (grow initial rule set): {len(initial)} rules grown")
p(f"Phase B (optimisation, k={silent.k}): " +
  (f"no changes; halted at iteration {opt.group(1)}" if opt else "see trace"))
distinct_final = list(dict.fromkeys(grown_final))
p(f"Phase C (cover remaining positives): rule list grew to {len(grown_final)} entries "
  f"({len(distinct_final)} distinct; new rules {'only repeat existing ones' if set(grown_final) <= set(initial) else 'include new rules'})" +
  (f"; stopped by the description-length limit (diff {halts[1][0]} bits > {halts[1][1]})" if len(halts) > 1 else ""))
p(f"Phase D (delete rules that increase total description length): {len(final)} rules kept, "
  f"{len(distinct_final) - len(final)} distinct rules deleted")
p("A 'rule error rate > 50%' stop was not applied (Wittgenstein's RIPPER has no such check).")
p("")
p("Rules grown in Phase A (K = kept in the final rule set):")
for k, r in enumerate(initial, 1):
    p(f"  {k:2d}. {'K' if r in final else ' '} [{r}]")
p("")
p(f"Final rule set: {silent.ruleset_}")
p("")
p("=" * 90)
p("FULL WITTGENSTEIN TRACE (verbosity=3)")
p("=" * 90)

body = summary.getvalue() + trace
elapsed = time.time() - t0
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
with open(LOG_PATH, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("EXECUTION LOG: Diagnostic: RIPPER Induction Trace (Wittgenstein verbose output)\n")
    f.write("Script: ripper_trace.py\n")
    f.write(f"Python Runtime: {sys.version.split()[0]} ({sys.executable})\n")
    f.write(f"Execution Duration: {elapsed:.2f} seconds\n")
    f.write("Exit Code: 0\n")
    f.write("=" * 80 + "\n\n")
    f.write(body)
print(summary.getvalue().split("FULL WITTGENSTEIN TRACE")[0].rstrip("= \n"))
print(f"\nTrace written to {LOG_PATH}")
