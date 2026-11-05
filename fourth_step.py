import os
import sys
import io

# 1. Agnostic Standard Output Stream Wrapping (Guarantees Windows UTF-8 Terminal Safety)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure the current working directory is at the front of sys.path for ChefBoost's dynamic imports
sys.path.insert(0, os.getcwd())

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# ==============================================================================
# STEP 5 & STEP 6: WHITE-BOX MODELING VIA C4.5 AND RIPPER INDUCTIONS
# ==============================================================================

print("="*80)
print("STAGE 5 & 6 PIPELINE: INDUCTION OF INTERPRETABLE WHITE-BOX MODELS")
print("="*80)

# ------------------------------------------------------------------------------
# 1. INGEST PRE-PARTITIONED AND OVERSAMPLED DATASETS
# ------------------------------------------------------------------------------
print("\n[1/4] Ingesting train and test splits...")
try:
    X_train = pd.read_csv("X_train_resampled.csv")
    Y_train = pd.read_csv("Y_train_resampled.csv")['hml32']
    X_test = pd.read_csv("X_test.csv")
    Y_test = pd.read_csv("Y_test.csv")['hml32']
    print(f"Successfully loaded datasets:")
    print(f" - Resampled Train instances: {len(X_train)}")
    print(f" - Stratified Test instances: {len(X_test)}")
except Exception as e:
    print(f"Error loading datasets: {e}")
    print("Ensure third_step.py was run successfully to generate the files.")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. DATA TYPE SANITIZATION (STABILIZING CONTINUOUS VS NOMINAL SPACES)
# ------------------------------------------------------------------------------
print("\n[2/4] Sanitizing nominal vs continuous feature datatypes...")
# Continuous feature: b19 (Age in months)
X_train['b19'] = X_train['b19'].astype(int)
X_test['b19'] = X_test['b19'].astype(int)

# Nominal features: cast to int first (to drop any trailing .0 floats), then str, then object
nominal_features = ['b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128', 'v129', 'v113', 'v116', 'hml20']
for col in nominal_features:
    X_train[col] = X_train[col].astype(int).astype(str).astype(object)
    X_test[col] = X_test[col].astype(int).astype(str).astype(object)

print(" - Datatypes successfully locked down.")

# ------------------------------------------------------------------------------
# 3. STEP 5: INDUCE C4.5 DECISION TREE MODEL VIA CHEFBOOST
# ------------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 5: C4.5 DECISION TREE INDUCTION (CHEFBOOST)")
print(" - Optimizing Claude Shannon's Information Entropy & Split Gain Ratio")
print("-"*80)

chef_model = None
chef_test_preds = []

try:
    from chefboost import Chefboost as chef
    
    # Configure C4.5 training dataframe
    # Chefboost requires the target to be named 'Decision' and placed as the final column
    chef_df = X_train.copy()
    chef_df['Decision'] = Y_train.map({0: 'Negative', 1: 'Positive'}).astype(object)
    
    # Fit C4.5 tree
    config = {'algorithm': 'C4.5'}
    chef_model = chef.fit(chef_df, config=config, target_label='Decision')
    print("\nChefBoost C4.5 model trained successfully!")
    
    # Read and print the generated C4.5 ruleset from rules.py
    rules_file_path = os.path.join("outputs", "rules", "rules.py")
    if os.path.exists(rules_file_path):
        print("\n=== GENERATED C4.5 RULES PATHS (outputs/rules/rules.py) ===")
        with open(rules_file_path, 'r', encoding='UTF-8') as f:
            print(f.read())
        print("="*60)
    
    # Predict on test set
    print("Running predictions on clinical test set...")
    chef_test_preds_str = []
    for idx, row in X_test.iterrows():
        features = list(row.values)
        pred = chef.predict(chef_model, features)
        chef_test_preds_str.append(pred)
        
    # Map predictions back to binary integers (0 = Negative, 1 = Positive)
    for p in chef_test_preds_str:
        if p == 'Positive':
            chef_test_preds.append(1)
        elif p == 'Negative':
            chef_test_preds.append(0)
        else:
            # Fallback for nulls or unexpected values
            chef_test_preds.append(0)
            
except Exception as e:
    print(f"Error executing ChefBoost modeling: {e}")
    import traceback
    traceback.print_exc()

# ------------------------------------------------------------------------------
# 4. STEP 6: INDUCE RIPPER SEQUENTIAL COVERING RULESET VIA WITTGENSTEIN
# ------------------------------------------------------------------------------
print("\n" + "-"*80)
print("STEP 6: RIPPER SEPARATE-AND-CONQUER RULES INDUCTION (WITTGENSTEIN)")
print(" - Optimizing FOIL Information Gain for Positive Class (1)")
print("-"*80)

ripper_model = None
ripper_test_preds = []

try:
    import wittgenstein as lw
    
    # Fit RIPPER ruleset
    ripper_model = lw.RIPPER(random_state=42)
    ripper_model.fit(X_train, Y_train, pos_class=1)
    
    print("RIPPER model trained successfully!")
    print("\n=== GENERATED RIPPER SEPARATE-AND-CONQUER RULESET ===")
    print(ripper_model.ruleset_)
    print("="*60)
    
    # Predict on test set
    print("Running predictions on clinical test set...")
    ripper_raw_preds = ripper_model.predict(X_test)
    ripper_test_preds = [1 if p else 0 for p in ripper_raw_preds]
    
except Exception as e:
    print(f"Error executing RIPPER modeling: {e}")
    import traceback
    traceback.print_exc()

# ------------------------------------------------------------------------------
# 5. PRELIMINARY PERFORMANCE METRICS COMPARATIVE REPORT
# ------------------------------------------------------------------------------
print("\n" + "="*80)
print("PRELIMINARY PERFORMANCE METRICS COMPARATIVE MATRIX")
print("="*80)

if chef_test_preds:
    print("\n[MODEL 1] ChefBoost C4.5 Decision Tree:")
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, chef_test_preds))
    print("\nClassification Report:")
    print(classification_report(Y_test, chef_test_preds, target_names=['Negative', 'Positive']))
else:
    print("\n[MODEL 1] ChefBoost C4.5 Decision Tree evaluation was skipped due to error.")

if ripper_test_preds:
    print("\n" + "-"*60)
    print("[MODEL 2] Wittgenstein RIPPER Ruleset:")
    print("Confusion Matrix:")
    print(confusion_matrix(Y_test, ripper_test_preds))
    print("\nClassification Report:")
    print(classification_report(Y_test, ripper_test_preds, target_names=['Negative', 'Positive']))
else:
    print("\n[MODEL 2] Wittgenstein RIPPER evaluation was skipped due to error.")

print("="*80)
print("Pipeline steps 5 and 6 executions completed successfully.")
print("="*80)

# ------------------------------------------------------------------------------
# 6. GENERATE FIGURE 4.5 DIRECTLY FROM outputs/rules/rules.py
# ------------------------------------------------------------------------------
# The diagram is drawn from the parsed rules file; node counts are computed from
# the training data. Nothing about the tree structure or its counts is typed in.
print("\n[6] Generating Figure 4.5 from outputs/rules/rules.py ...")
try:
    from c45_tree_utils import C45Tree, plot_tree

    tree = C45Tree.from_file(os.path.join("outputs", "rules", "rules.py"))
    X_train_raw = pd.read_csv("X_train_resampled.csv")
    Y_train_raw = pd.read_csv("Y_train_resampled.csv")['hml32'].astype(int).values
    n_original = len(pd.read_csv("Y_train_original.csv"))
    pathway_map = tree.resolve_pathways()

    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
    # Figure 4.5: named-pathway view (off-path subtrees collapsed, counts exact)
    omitted = plot_tree(
        tree, X_train_raw, Y_train_raw,
        out_paths=[os.path.join("outputs", "plots", "decision_tree_plot.png"),
                   os.path.join("outputs", "decision_tree_plot.png")],
        n_original=n_original, pathway_map=pathway_map, collapse_offpath=True,
        title="C4.5 (ChefBoost) tree from rules.py: named pathways C1-C5",
    )
    # Appendix figure: the complete tree, nothing collapsed
    plot_tree(
        tree, X_train_raw, Y_train_raw,
        out_paths=[os.path.join("outputs", "plots", "decision_tree_full.png")],
        n_original=n_original, pathway_map=pathway_map, collapse_offpath=False,
        title="C4.5 (ChefBoost) tree from rules.py: complete tree", font=5.0, row_h=0.29,
    )
    print(f" - Leaves in rules.py: {len(tree.leaves)} "
          f"({len(tree.leaves) - len(omitted)} reached by training records, "
          f"{len(omitted)} unseen-value fallback leaves not drawn)")
    print(" - Named pathways resolved to leaves:")
    for name, ids in pathway_map.items():
        for i in ids:
            print(f"     {name} -> Leaf {i} ({tree.leaves[i].label}): {tree.describe_path(tree.leaves[i])}")
    print(" - Figure 4.5 (pathways view) written to outputs/plots/decision_tree_plot.png")
    print(" - Complete tree written to outputs/plots/decision_tree_full.png")
except Exception as e:
    print(f"Error generating Figure 4.5 from rules.py: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
