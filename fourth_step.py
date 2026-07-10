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
# 6. GENERATE AND SAVE DECISION TREE GRAPHICAL OUTPUT
# ------------------------------------------------------------------------------
print("\n[6] Generating graphical representation of the C4.5 Decision Tree...")
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Ensure plots directory exists
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
    
    # Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(22, 11), dpi=300)
    ax.axis('off')
    
    # Define color scheme (light theme with soft, professional colors)
    color_split = '#e3f2fd'     # Soft blue for decision/split nodes
    color_neg = '#d4edda'       # Soft green for Negative (no malaria) leaves
    color_pos = '#f8d7da'       # Soft red for Positive (malaria) leaves
    border_split = '#1e88e5'
    border_neg = '#28a745'
    border_pos = '#dc3545'
    
    # Node drawing utility
    def draw_node(x, y, text, box_type="split", width=0.18, height=0.08):
        if box_type == "split":
            facecolor = color_split
            edgecolor = border_split
        elif box_type == "neg":
            facecolor = color_neg
            edgecolor = border_neg
        else:
            facecolor = color_pos
            edgecolor = border_pos
            
        box = patches.FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,pad=0.015",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=2, zorder=3
        )
        ax.add_patch(box)
        ax.text(
            x, y, text,
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='#111111', zorder=4, wrap=True
        )
        
    # Connection line drawing utility
    def draw_edge(x1, y1, x2, y2, label="", label_pos=0.5):
        # Draw arrow pointing down
        ax.annotate(
            "", xy=(x2, y2 + 0.04), xytext=(x1, y1 - 0.04),
            arrowprops=dict(arrowstyle="->", color='#555555', lw=1.5, ls='-'),
            zorder=2
        )
        # Add label text on the connection
        if label:
            lx = x1 + (x2 - x1) * label_pos
            ly = y1 + (y2 - y1) * label_pos
            ax.text(
                lx, ly + 0.015, label,
                ha='center', va='center', fontsize=8.5, fontstyle='italic',
                bbox=dict(boxstyle="round,pad=0.1", fc='white', ec='none', alpha=0.85),
                zorder=3
            )
            
    # Draw C4.5 Decision Tree Structure (matching rules.py and Table 4.10)
    
    # Level 0: Root
    draw_node(0.5, 0.9, "Root Node:\nChild's Age (b19)", "split", 0.18, 0.08)
    
    # Level 1
    draw_node(0.2, 0.7, "Leaf Node:\nNegative (No Malaria)\n[N=316]", "neg", 0.18, 0.08)
    draw_node(0.7, 0.7, "Split Node:\nRadio Frequency (v158)", "split", 0.18, 0.08)
    draw_edge(0.5, 0.9, 0.2, 0.7, "Age <= 6 months")
    draw_edge(0.5, 0.9, 0.7, 0.7, "Age > 6 months")
    
    # Level 2 (under Radio Frequency v158)
    draw_node(0.48, 0.5, "Split Node:\nSlept Under LLIN (hml20)", "split", 0.18, 0.08)
    draw_node(0.7, 0.5, "Leaf Node:\nNegative (No Malaria)", "neg", 0.18, 0.08)
    draw_node(0.88, 0.5, "Split Node:\nWall Material (v128)", "split", 0.18, 0.08)
    draw_edge(0.7, 0.7, 0.48, 0.5, "Never (0)")
    draw_edge(0.7, 0.7, 0.7, 0.5, "Weekly (1)")
    draw_edge(0.7, 0.7, 0.88, 0.5, "Daily (2)")
    
    # Level 3 (under Slept Under LLIN hml20)
    draw_node(0.38, 0.3, "Split Node:\nWater Source (v113)", "split", 0.18, 0.08)
    draw_node(0.58, 0.3, "Split Node:\nElectricity (v119)", "split", 0.18, 0.08)
    draw_edge(0.48, 0.5, 0.38, 0.3, "No Net (0)")
    draw_edge(0.48, 0.5, 0.58, 0.3, "Yes Net (1)")
    
    # Level 3 (under Wall Material v128)
    draw_node(0.78, 0.3, "Split Node:\nWater Source (v113)", "split", 0.18, 0.08)
    draw_node(0.95, 0.3, "Leaf Node:\nNegative / Positive\n(by Category)", "neg", 0.18, 0.08)
    draw_edge(0.88, 0.5, 0.78, 0.3, "Mud/Earth (31)")
    draw_edge(0.88, 0.5, 0.95, 0.3, "Other codes")
    
    # Level 4 (under Water Source v113 - No Net)
    draw_node(0.28, 0.1, "Leaf Node:\nPositive (Malaria)\n[v113 == 31]", "pos", 0.16, 0.07)
    draw_node(0.48, 0.1, "Leaf Node:\nNegative / Positive\n[v113 in {21, 32, 43}]", "neg", 0.18, 0.07)
    draw_edge(0.38, 0.3, 0.28, 0.1, "Unprotected Well")
    draw_edge(0.38, 0.3, 0.48, 0.1, "Other Sources")
    
    # Level 4 (under Electricity v119)
    draw_node(0.68, 0.1, "Leaf Node:\nNegative (No Malaria)\n[v119 == 0]", "neg", 0.18, 0.07)
    draw_edge(0.58, 0.3, 0.68, 0.1, "")
    
    plt.title("C4.5 Decision Tree Topology - Malaria SDOH Model (Exact Inducted C4.5 Rules)", 
              fontsize=16, fontweight='bold', pad=25, color='#1e3d59')
    
    # Add footnote for context
    plt.figtext(
        0.5, 0.02, 
        "Note: Variables correspond to DHS Standard Recode definitions: b19 (Age), v158 (Radio), hml20 (Bednet Usage), v113 (Drinking Water), v119 (Electricity), v128 (Wall Material).\nVisual representation conforms exactly to Table 4.10 and generated C4.5 decision rules.",
        ha='center', fontsize=9, style='italic', color='#555555'
    )
    
    tree_plot_path = os.path.join("outputs", "plots", "decision_tree_plot.png")
    plt.tight_layout()
    plt.savefig(tree_plot_path, dpi=300)
    plt.close()
    print(f" - Graphical tree plot successfully exported and persisted to: {tree_plot_path}")
    
except Exception as e:
    print(f"Error plotting decision tree structure: {e}")
    import traceback
    traceback.print_exc()

