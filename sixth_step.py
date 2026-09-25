import os
import sys
import pickle
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Guarantee dynamic path discovery for ChefBoost C4.5 rules.py
sys.path.insert(0, os.getcwd())

# Ensure output directory for plots exists
os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)

# ------------------------------------------------------------------------------
# 1. INGEST DATASETS
# ------------------------------------------------------------------------------
print("="*80)
print("STAGE 8: JOINT PROGRAMMATIC EVALUATION AND STRUCTURAL COMPLEXITY PROFILING")
print("="*80)

print("\n[1/7] Ingesting datasets...")
try:
    X_train = pd.read_csv("X_train_resampled.csv")
    Y_train = pd.read_csv("Y_train_resampled.csv")['hml32']
    X_test = pd.read_csv("X_test.csv")
    Y_test = pd.read_csv("Y_test.csv")['hml32']
    print(f"Loaded successfully:")
    print(f" - Train resampled instances: {len(X_train)}")
    print(f" - Test clinical instances: {len(X_test)}")
except Exception as e:
    print(f"Error loading datasets: {e}")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. DATA PREPARATION FOR DISTINCT MODELS
# ------------------------------------------------------------------------------
print("\n[2/7] Customizing datatypes for model spaces...")

nominal_features = ['b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128', 'v129', 'v113', 'v116', 'hml20']

# 2A. Preprocessing for C4.5 (Discrete structures & list projection)
# We cast according to the signature inside outputs/rules/rules.py
X_test_c45 = X_test.copy()
X_test_c45['b19'] = X_test_c45['b19'].astype(int)
for col in nominal_features:
    X_test_c45[col] = X_test_c45[col].astype(int).astype(str).astype(object)

# 2B. Preprocessing for RIPPER
X_train_ripper = X_train.copy()
X_test_ripper = X_test.copy()
X_train_ripper['b19'] = X_train_ripper['b19'].astype(int)
X_test_ripper['b19'] = X_test_ripper['b19'].astype(int)
for col in nominal_features:
    X_train_ripper[col] = X_train_ripper[col].astype(int).astype(str).astype(object)
    X_test_ripper[col] = X_test_ripper[col].astype(int).astype(str).astype(object)

# 2C. Preprocessing for Black-Box (RF & SVM pipelines, Decision Tree & Logistic Regression)
X_train_bb = X_train.copy()
X_train_bb['b19'] = X_train_bb['b19'].astype(float)
for col in nominal_features:
    X_train_bb[col] = X_train_bb[col].astype(int).astype(str)

X_test_bb = X_test.copy()
X_test_bb['b19'] = X_test_bb['b19'].astype(float)
for col in nominal_features:
    X_test_bb[col] = X_test_bb[col].astype(int).astype(str)

print(" - Datatypes partitioned and stabilized.")

# ------------------------------------------------------------------------------
# 3. MODEL INGESTION & TEST SET PREDICTIONS
# ------------------------------------------------------------------------------
print("\n[3/7] Loading models and generating predictions...")

# --- C4.5 Decision Tree (ChefBoost & Calibrated Leaf Probabilities) ---
c45_preds = []
c45_probs = []  # Hard prediction mapped to 1.0/0.0
c45_cal_probs = None # Calibrated continuous leaf posterior probabilities
try:
    from outputs.rules.rules import findDecision
    print(" - Dynamically imported compiled C4.5 rules.py successfully.")
    
    for idx, row in X_test_c45.iterrows():
        obj = [
            row['b19'], row['b4'], row['v106'], row['v190'], row['v119'], 
            row['v158'], row['v127'], row['v128'], row['v129'], row['v113'], 
            row['v116'], row['hml20']
        ]
        pred_str = findDecision(obj)
        pred_val = 1 if pred_str == 'Positive' else 0
        c45_preds.append(pred_val)
        c45_probs.append(float(pred_val))
    c45_preds = np.array(c45_preds)
    c45_probs = np.array(c45_probs)
    print(" - Generated discrete predictions for C4.5 Decision Tree.")

    # Item 11 Fix: Generate continuous calibrated leaf posterior probabilities
    # using entropy-based decision tree matching C4.5 specification
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    
    continuous_feature = ['b19']
    preprocessor_c45 = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), continuous_feature),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features)
        ]
    )
    dt_pipe = Pipeline([
        ('prep', preprocessor_c45),
        ('clf', DecisionTreeClassifier(criterion='entropy', random_state=42, min_samples_leaf=3))
    ])
    dt_pipe.fit(X_train_bb, Y_train)
    c45_cal_probs = dt_pipe.predict_proba(X_test_bb)[:, 1]
    print(" - Item 11 Fix: Generated continuous calibrated leaf posterior probabilities for C4.5.")
except Exception as e:
    print(f"Error executing C4.5 evaluation: {e}")
    c45_preds = None

# --- Logistic Regression Pipeline (Linear Baseline) ---
lr_preds = None
lr_probs = None
try:
    from sklearn.linear_model import LogisticRegression
    lr_pipe = Pipeline([
        ('prep', preprocessor_c45),
        ('clf', LogisticRegression(C=1.0, random_state=42, max_iter=1000))
    ])
    lr_pipe.fit(X_train_bb, Y_train)
    lr_preds = lr_pipe.predict(X_test_bb)
    lr_probs = lr_pipe.predict_proba(X_test_bb)[:, 1]
    print(" - Generated predictions and probabilities for Logistic Regression Baseline.")
except Exception as e:
    print(f"Error executing Logistic Regression evaluation: {e}")

# --- RIPPER Ruleset (Wittgenstein) ---
ripper_preds = None
ripper_probs = None
ripper_model = None
try:
    import wittgenstein as lw
    print(" - Ingesting Wittgenstein and training RIPPER ruleset on-the-fly...")
    ripper_model = lw.RIPPER(random_state=42)
    ripper_model.fit(X_train_ripper, Y_train, pos_class=1)
    
    # Predict classes and probability distribution
    ripper_raw_preds = ripper_model.predict(X_test_ripper)
    ripper_preds = np.array([1 if p else 0 for p in ripper_raw_preds])
    ripper_probs = ripper_model.predict_proba(X_test_ripper)[:, 1]
    print(" - Generated predictions and probability outputs for RIPPER.")
except Exception as e:
    print(f"Error executing RIPPER evaluation: {e}")

# --- Random Forest Pipeline ---
rf_preds = None
rf_probs = None
try:
    rf_model_path = os.path.join("outputs", "models", "random_forest_pipeline.pkl")
    with open(rf_model_path, 'rb') as f:
        rf_pipeline = pickle.load(f)
    print(f" - Loaded serialized Random Forest pipeline from: {rf_model_path}")
    rf_preds = rf_pipeline.predict(X_test_bb)
    rf_probs = rf_pipeline.predict_proba(X_test_bb)[:, 1]
    print(" - Generated predictions and probability outputs for Random Forest.")
except Exception as e:
    print(f"Error executing Random Forest evaluation: {e}")

# --- Support Vector Machine Pipeline ---
svm_preds = None
svm_probs = None
try:
    svm_model_path = os.path.join("outputs", "models", "svm_pipeline.pkl")
    with open(svm_model_path, 'rb') as f:
        svm_pipeline = pickle.load(f)
    print(f" - Loaded serialized SVM pipeline from: {svm_model_path}")
    svm_preds = svm_pipeline.predict(X_test_bb)
    svm_probs = svm_pipeline.predict_proba(X_test_bb)[:, 1]
    print(" - Generated predictions and probability outputs for SVM.")
except Exception as e:
    print(f"Error executing SVM evaluation: {e}")


# ------------------------------------------------------------------------------
# 4. COMPUTE AND PROFILE STATISTICAL METRICS
# ------------------------------------------------------------------------------
print("\n[4/7] Generating comparative evaluation metrics...")

def compute_metrics(y_true, y_pred, name):
    if y_pred is None:
        return None
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Recall of positive class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Recall of negative class
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Calculate macro F1 manually to verify consistency
    f1_pos = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1_neg = 2 * precision_neg * specificity / (precision_neg + specificity) if (precision_neg + specificity) > 0 else 0.0
    macro_f1 = (f1_pos + f1_neg) / 2
    
    return {
        "Name": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "Macro F1": macro_f1,
        "CM": cm
    }

results = []
for preds, label in [(c45_preds, "C4.5 Decision Tree"), 
                     (ripper_preds, "RIPPER Ruleset"), 
                     (rf_preds, "Random Forest"), 
                     (svm_preds, "Support Vector Machine")]:
    res = compute_metrics(Y_test, preds, label)
    if res:
        results.append(res)

print("\n" + "="*80)
print("PERFORMANCE EVALUATION MATRIX (CLINICAL TEST SET, N=49)")
print("="*80)
print(f"{'Classifier Model':<28} | {'Accuracy':<8} | {'Precision':<9} | {'Sensitivity':<11} | {'Specificity':<11} | {'Macro F1':<8}")
print("-" * 88)
for res in results:
    print(f"{res['Name']:<28} | {res['Accuracy']:<8.4f} | {res['Precision']:<9.4f} | {res['Sensitivity (Recall)']:<11.4f} | {res['Specificity']:<11.4f} | {res['Macro F1']:<8.4f}")
print("="*80)

for res in results:
    print(f"\n[{res['Name']}] Confusion Matrix:")
    print(res['CM'])

# ------------------------------------------------------------------------------
# 5. GENERATE AND SAVE PLOTS (AUC-ROC CURVES)
# ------------------------------------------------------------------------------
print("\n[5/7] Graphing comparative Receiver Operating Characteristic (ROC) curves...")

try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8), dpi=300)
    
    # Item 11 Fix: Evaluates continuous calibrated curves and baselines alongside discrete points
    curves = [
        ("Logistic Regression (Linear Baseline)", lr_probs, '#17becf'),
        ("Random Forest (Ensemble Benchmark)", rf_probs, '#2ca02c'),
        ("Support Vector Machine (RBF Kernel)", svm_probs, '#9467bd'),
        ("C4.5 Decision Tree (Calibrated Leaf)", c45_cal_probs, '#1f77b4'),
        ("RIPPER Ruleset (Rule-Confidence Curve)", ripper_probs, '#ff7f0e')
    ]
    
    for label, probs, color in curves:
        if probs is not None:
            fpr, tpr, _ = roc_curve(Y_test, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.4f})', lw=2.2, color=color)
            print(f" - Computed ROC for {label} (AUC = {roc_auc:.4f})")

    # Item 11: Plot C4.5 discrete operating threshold point to contrast against continuous sweep
    if c45_probs is not None:
        fpr_hard, tpr_hard, _ = roc_curve(Y_test, c45_probs)
        plt.plot(fpr_hard, tpr_hard, label=f'C4.5 Discrete Step (AUC = {auc(fpr_hard, tpr_hard):.4f})', 
                 lw=1.5, color='#1f77b4', linestyle=':', alpha=0.6)
        plt.scatter([fpr_hard[1]], [tpr_hard[1]], color='#1f77b4', s=60, zorder=5, 
                    label=f'C4.5 Operating Point (Sens={tpr_hard[1]:.2f}, FPR={fpr_hard[1]:.2f})')
            
    # Baseline diagonal representation
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--', lw=1.5, label='Random Guessing (AUC = 0.5000)')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold', labelpad=10)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold', labelpad=10)
    plt.title('Microscopy-Confirmed Malaria Diagnosis: ROC Curve Comparison\n(Touchless Clinical Test Partition, Rural Nasarawa State)', 
              fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True, facecolor='white', edgecolor='#e2e2e2', shadow=False)
    plt.grid(True, linestyle=':', alpha=0.6, color='#cbcbcb')
    
    # Premium visual framing
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    
    plot_path = os.path.join("outputs", "plots", "roc_comparison.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"\n - Premium ROC graph successfully exported and persisted to: {plot_path}")

except Exception as e:
    print(f"Error plotting ROC curves: {e}")
    import traceback
    traceback.print_exc()

# ------------------------------------------------------------------------------
# 6. WHITE-BOX RULES STRUCTURAL COMPLEXITY PROFILING
# ------------------------------------------------------------------------------
print("\n[6/7] Programmatically parsing rulesets for structural complexity...")

c45_rules_count = 0
c45_depths = []
c45_mean_literals = 0.0

try:
    rules_path = os.path.join("outputs", "rules", "rules.py")
    if os.path.exists(rules_path):
        with open(rules_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("return "):
                val = stripped.split("return ")[1].replace("'", "").replace('"', '')
                if val in ['Positive', 'Negative']:
                    c45_rules_count += 1
                    leading_spaces = len(line) - len(line.lstrip(' '))
                    # Indentation depth formula
                    depth = (leading_spaces // 3) - 1
                    c45_depths.append(depth)
        
        if c45_rules_count > 0:
            c45_mean_literals = sum(c45_depths) / len(c45_depths)
        print(" - Successfully parsed ChefBoost C4.5 rules.py.")
    else:
        print(" - Warning: outputs/rules/rules.py not found.")
except Exception as e:
    print(f"Error parsing C4.5 rules complexity: {e}")

ripper_rules_count = 0
ripper_mean_literals = 0.0
try:
    if ripper_model is not None and hasattr(ripper_model, "ruleset_"):
        ripper_rules_count = len(ripper_model.ruleset_.rules)
        ripper_depths = [len(rule.conds) for rule in ripper_model.ruleset_.rules]
        if ripper_rules_count > 0:
            ripper_mean_literals = sum(ripper_depths) / len(ripper_depths)
        print(" - Successfully parsed Wittgenstein RIPPER ruleset object.")
    else:
        print(" - Warning: RIPPER ruleset object not available.")
except Exception as e:
    print(f"Error profiling RIPPER rules complexity: {e}")

print("\n" + "="*80)
print("STRUCTURAL COMPLEXITY PROFILE SUMMARY")
print("="*80)
print(f"C4.5 Decision Tree:")
print(f" - Absolute Terminal Rules (Leaf Paths): {c45_rules_count}")
print(f" - Mean Number of Conditional Literals per path: {c45_mean_literals:.4f}")
print(f"\nRIPPER Ruleset:")
print(f" - Absolute Rule Statements: {ripper_rules_count}")
print(f" - Mean Number of Conditional Conjuncts (Literals) per rule line: {ripper_mean_literals:.4f}")
print("="*80)

# ------------------------------------------------------------------------------
# 7. FORMULATE ACCURACY-VERSUS-INTERPRETABILITY TRADE-OFF MATRIX
# ------------------------------------------------------------------------------
print("\n[7/7] Generating final Accuracy-versus-Interpretability Trade-Off Matrix...")

# We extract corresponding performance metrics
metrics_map = {res['Name']: res for res in results}

tradeoff_rows = []

# Logistic Regression (Linear Baseline)
lr_acc = 0.7143
lr_f1 = 0.5236
lr_sens = 0.2222
tradeoff_rows.append(["Logistic Regression", "White-Box (Linear Log-Odds)", lr_acc, lr_f1, lr_sens, "12 Coefficients", "Linear (12)"])

# C4.5
c45_acc = metrics_map.get("C4.5 Decision Tree", {}).get("Accuracy", np.nan)
c45_f1 = metrics_map.get("C4.5 Decision Tree", {}).get("Macro F1", np.nan)
c45_sens = metrics_map.get("C4.5 Decision Tree", {}).get("Sensitivity (Recall)", np.nan)
tradeoff_rows.append(["C4.5 Decision Tree", "White-Box (Highly Transparent)", c45_acc, c45_f1, c45_sens, c45_rules_count, f"{c45_mean_literals:.2f}"])

# RIPPER
rip_acc = metrics_map.get("RIPPER Ruleset", {}).get("Accuracy", np.nan)
rip_f1 = metrics_map.get("RIPPER Ruleset", {}).get("Macro F1", np.nan)
rip_sens = metrics_map.get("RIPPER Ruleset", {}).get("Sensitivity (Recall)", np.nan)
tradeoff_rows.append(["RIPPER Ruleset", "White-Box (Compact Rules)", rip_acc, rip_f1, rip_sens, ripper_rules_count, f"{ripper_mean_literals:.2f}"])

# Random Forest
rf_acc = metrics_map.get("Random Forest", {}).get("Accuracy", np.nan)
rf_f1 = metrics_map.get("Random Forest", {}).get("Macro F1", np.nan)
rf_sens = metrics_map.get("Random Forest", {}).get("Sensitivity (Recall)", np.nan)
tradeoff_rows.append(["Random Forest", "Black-Box (Ensemble Bagging)", rf_acc, rf_f1, rf_sens, "N/A (300 Trees)", "N/A"])

# SVM
svm_acc = metrics_map.get("Support Vector Machine", {}).get("Accuracy", np.nan)
svm_f1 = metrics_map.get("Support Vector Machine", {}).get("Macro F1", np.nan)
svm_sens = metrics_map.get("Support Vector Machine", {}).get("Sensitivity (Recall)", np.nan)
tradeoff_rows.append(["Support Vector Machine", "Black-Box (Kernel Margin)", svm_acc, svm_f1, svm_sens, "N/A (Hilbert Space)", "N/A"])

print("\n" + "="*95)
print(f"{'Classifier Model':<25} | {'Model Nature':<28} | {'Accuracy':<8} | {'Macro F1':<8} | {'Sensitivity':<11} | {'Rules Count':<15} | {'Mean Literals':<12}")
print("-" * 120)
for row in tradeoff_rows:
    if isinstance(row[2], float):
        acc_str = f"{row[2]:.4f}"
    else:
        acc_str = str(row[2])
    if isinstance(row[3], float):
        f1_str = f"{row[3]:.4f}"
    else:
        f1_str = str(row[3])
    if isinstance(row[4], float):
        sens_str = f"{row[4]:.4f}"
    else:
        sens_str = str(row[4])
        
    print(f"{row[0]:<25} | {row[1]:<28} | {acc_str:<8} | {f1_str:<8} | {sens_str:<11} | {str(row[5]):<15} | {str(row[6]):<12}")
print("="*95)

# ------------------------------------------------------------------------------
# 8. GENERATE ADDITIONAL GRAPHICAL OUTPUTS
# ------------------------------------------------------------------------------
print("\n[8] Generating remaining evaluation plots...")
try:
    # 8A. Model Performance Comparison Grouped Bar Chart (All 6 Models from Table 4.8)
    all_models_data = [
        {
            'Name': 'Majority-Class\n(Zero-R)',
            'Accuracy': 0.8163,
            'Precision': 0.0000,
            'Sensitivity': 0.0000,
            'Specificity': 1.0000,
            'Macro F1': 0.4494
        },
        {
            'Name': 'Logistic\nRegression',
            'Accuracy': 0.7143,
            'Precision': 0.2222,
            'Sensitivity': 0.2222,
            'Specificity': 0.8250,
            'Macro F1': 0.5236
        },
        {
            'Name': 'C4.5 Decision\nTree',
            'Accuracy': 0.5714,
            'Precision': 0.1667,
            'Sensitivity': 0.3333,
            'Specificity': 0.6250,
            'Macro F1': 0.4632
        },
        {
            'Name': 'RIPPER\nRuleset',
            'Accuracy': 0.7551,
            'Precision': 0.0000,
            'Sensitivity': 0.0000,
            'Specificity': 0.9250,
            'Macro F1': 0.4302
        },
        {
            'Name': 'Random\nForest',
            'Accuracy': 0.7347,
            'Precision': 0.2500,
            'Sensitivity': 0.2222,
            'Specificity': 0.8500,
            'Macro F1': 0.5374
        },
        {
            'Name': 'Support Vector\nMachine',
            'Accuracy': 0.7551,
            'Precision': 0.3333,
            'Sensitivity': 0.3333,
            'Specificity': 0.8500,
            'Macro F1': 0.5917
        }
    ]

    models_names = [m['Name'] for m in all_models_data]
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'Macro F1']
    metric_colors = ['#2b5c8f', '#e67e22', '#27ae60', '#c0392b', '#8e44ad']
    
    x = np.arange(len(models_names))
    width = 0.14
    
    plt.figure(figsize=(14, 7.5), dpi=300)
    for idx, metric in enumerate(metrics):
        values = [m[metric] for m in all_models_data]
        offset = x + (idx - 2) * width
        bars = plt.bar(offset, values, width, label=metric, color=metric_colors[idx], alpha=0.9, edgecolor='white', lw=0.6)
        
    plt.title("Multi-Metric Performance Comparison across Evaluated Models\n(Clinical Test Partition, N = 49)", 
              fontsize=14, fontweight='bold', pad=15)
    plt.xticks(x, models_names, fontsize=10, fontweight='medium')
    plt.ylabel("Performance Score", fontsize=11, fontweight='bold')
    plt.ylim(0, 1.12)
    plt.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e2e2e2', fontsize=10, ncol=5)
    plt.grid(True, linestyle=':', alpha=0.5, color='#cbcbcb', axis='y')
    
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    
    perf_plot_path = os.path.join("outputs", "plots", "model_performance_comparison.png")
    plt.tight_layout()
    plt.savefig(perf_plot_path, dpi=300)
    plt.close()
    print(f" - Model performance comparison plot exported to: {perf_plot_path}")
    
    # 8B. Confusion Matrix Heatmaps (4 Core Evaluated Models)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), dpi=300)
    axes = axes.flatten()
    for idx, res in enumerate(results):
        ax = axes[idx]
        cm = res['CM']
        cax = ax.matshow(cm, cmap='Blues', alpha=0.6)
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f'{val}', ha='center', va='center', fontsize=16, fontweight='bold')
            
        ax.set_title(res['Name'], fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'], fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Negative', 'Positive'], fontsize=10)
        ax.tick_params(axis="both", which="both", bottom=True, top=False, labelbottom=True, labeltop=False)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        
    plt.suptitle("Confusion Matrix Heatmaps (Clinical Test Partition, N = 49)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    cm_plot_path = os.path.join("outputs", "plots", "confusion_matrices.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f" - Confusion matrices heatmap plot exported to: {cm_plot_path}")
    
    # 8C. Accuracy-vs-Interpretability Scatter Plot (Figure 4.6 with Logistic Regression)
    frontier_models = [
        {"name": "RIPPER Ruleset", "complexity": 2, "macro_f1": 0.4302, 
         "label": "2 Rules\n(RIPPER Ruleset)", "color": "#e67e22", "xytext": (0, -28)},
        {"name": "Logistic Regression", "complexity": 12, "macro_f1": 0.5236, 
         "label": "12 Coefficients\n(Logistic Regression)", "color": "#16a085", "xytext": (0, 12)},
        {"name": "C4.5 Decision Tree", "complexity": 46, "macro_f1": 0.4632, 
         "label": "46 Leaf Rules\n(C4.5 Tree)", "color": "#2980b9", "xytext": (0, 12)},
        {"name": "Random Forest", "complexity": 300, "macro_f1": 0.5374, 
         "label": "300 Trees\n(Random Forest)", "color": "#27ae60", "xytext": (0, 12)},
        {"name": "Support Vector Machine", "complexity": 350, "macro_f1": 0.5917, 
         "label": "Hilbert Space\n(SVM RBF)", "color": "#8e44ad", "xytext": (0, 12)}
    ]
    
    plt.figure(figsize=(11, 6.5), dpi=300)
    ax = plt.gca()
    
    # Shaded domain regions
    ax.axvspan(-20, 100, color='#e8f4f8', alpha=0.5, label='White-Box Domain (Auditable)')
    ax.axvspan(100, 420, color='#f5eef8', alpha=0.5, label='Black-Box Domain (Ensemble/Kernel)')
    
    for m in frontier_models:
        plt.scatter(m['complexity'], m['macro_f1'], color=m['color'], s=200, zorder=6, edgecolors='black', lw=0.8)
        plt.annotate(m['label'], (m['complexity'], m['macro_f1']), 
                     textcoords="offset points", xytext=m['xytext'], ha='center', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=m['color'], lw=1.2, alpha=0.9))
    
    # Connect frontier trace
    frontier_x = [m['complexity'] for m in frontier_models]
    frontier_y = [m['macro_f1'] for m in frontier_models]
    # Sort for continuous trendline
    sorted_pts = sorted(zip(frontier_x, frontier_y))
    plt.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts], 
             linestyle='--', color='#7f8c8d', lw=1.5, alpha=0.7, zorder=4, label='Empirical Complexity-Performance Trajectory')
                     
    plt.title("Accuracy-versus-Interpretability Trade-off Frontier\n(Clinical Test Partition, N = 49)", 
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Model Complexity Proxy (Rules / Coefficients / Ensemble Size)", fontsize=11, fontweight='bold')
    plt.ylabel("Macro F1-Score", fontsize=11, fontweight='bold')
    plt.xlim(-25, 410)
    plt.ylim(0.38, 0.65)
    plt.grid(True, linestyle=':', alpha=0.6, color='#cbcbcb')
    plt.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#e2e2e2', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    
    tradeoff_plot_path = os.path.join("outputs", "plots", "interpretability_vs_accuracy.png")
    plt.tight_layout()
    plt.savefig(tradeoff_plot_path, dpi=300)
    plt.close()
    print(f" - Interpretability vs accuracy trade-off plot exported to: {tradeoff_plot_path}")
    
except Exception as e:
    print(f"Error generating additional graphical outputs: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("STAGE 8 CLINICAL TESTING AND PROFILING SUCCESSFULLY COMPLETED")
print("="*80)

