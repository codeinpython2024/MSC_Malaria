import os
import wittgenstein as lw
import pandas as pd

# Load data
X_train = pd.read_csv("X_train_resampled.csv")
Y_train = pd.read_csv("Y_train_resampled.csv")['hml32']
X_test = pd.read_csv("X_test.csv")

# Train RIPPER
ripper = lw.RIPPER(random_state=42)
ripper.fit(X_train, Y_train, pos_class=1)

print("RIPPER Ruleset:")
print(ripper.ruleset_)

print("\nRules count:")
print(len(ripper.ruleset_.rules))

print("\nRules detail:")
for i, rule in enumerate(ripper.ruleset_.rules):
    print(f"Rule {i}: {rule}")
    print(f"  Conjuncts count: {len(rule.conds)}")
    print(f"  Conjuncts: {rule.conds}")

# Try predict_proba
try:
    probs = ripper.predict_proba(X_test)
    print("\npredict_proba successful!")
    print(probs[:5])
except Exception as e:
    print(f"\npredict_proba failed: {e}")
