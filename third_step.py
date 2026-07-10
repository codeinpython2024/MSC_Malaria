import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.model_selection import train_test_split
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTENC

# ==============================================================================
# STEP 3: ESTABLISH THE VALIDATION PARTITION VIA STRATIFIED TRAIN-TEST SPLITTING
# ==============================================================================
print("Executing Step 3: Establish the Validation Partition via Stratified Splitting...")

# 1. Load the type-stabilized ML matrix from Step 1
input_csv = "final_malaria_sdoh_matrix.csv"
try:
    df = pd.read_csv(input_csv)
    print(f"Successfully loaded dataset: {input_csv} ({len(df)} records)")
except FileNotFoundError:
    print(f"Error: {input_csv} not found! Please run first_stage.py first.")
    exit(1)

# 2. Extract predictors (X) and target (Y)
# Exclude administrative/weight tracking metadata (hv001, hv002, hvidx, v005)
target_col = 'hml32'
feature_cols = [
    'b19',   # Child's Age (Continuous)
    'b4',    # Child's Sex (Nominal)
    'v106',  # Maternal Education (Nominal)
    'v190',  # Wealth Quintile (Ordinal/Nominal)
    'v119',  # Electricity (Nominal)
    'v158',  # Radio (Nominal)
    'v127',  # Floor Material (Nominal)
    'v128',  # Wall Material (Nominal)
    'v129',  # Roof Material (Nominal)
    'v113',  # Water Source (Nominal)
    'v116',  # Toilet Type (Nominal)
    'hml20'  # Slept Under LLIN (Nominal)
]

X = df[feature_cols].copy()
Y = df[target_col].copy()

# 3. Restabilize data types on ingestion to block any auto-parsing float decay
X['b19'] = pd.to_numeric(X['b19']).astype(int)
nominal_features = ['b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128', 'v129', 'v113', 'v116', 'hml20']
for col in nominal_features:
    X[col] = pd.to_numeric(X[col]).astype(int).astype(str)

Y = pd.to_numeric(Y).astype(int)

# 4. Perform Stratified Train-Test Split (80% train, 20% test)
# Stratify=Y guarantees the minority class ratio (~18.5%) is perfectly mirrored.
# random_state=42 ensures absolute reproducibility.
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, 
    test_size=0.20, 
    stratify=Y, 
    random_state=42
)

print(f" - Train set size: {len(X_train)} records")
print(f" - Test set size: {len(X_test)} records")

# ==============================================================================
# STEP 4: EXECUTE DATA-LEVEL RESAMPLING VIA SMOTE-NC
# ==============================================================================
print("\nExecuting Step 4: Execute Data-Level Resampling via SMOTE-NC...")

# 1. Identify index positions of all nominal features in X matrix
# In X, b19 is at index 0 (continuous). Indices 1 to 11 are the 11 nominal variables.
categorical_indices = [X.columns.get_loc(col) for col in nominal_features]
print(f" - Tracked Categorical Column Indices for SMOTE-NC: {categorical_indices}")

# 2. Instantiate and run SMOTE-NC over the training partition only
# This prevents target leakage into the isolated evaluation validation partition.
smotenc = SMOTENC(
    categorical_features=categorical_indices, 
    random_state=42, 
    k_neighbors=5
)

X_train_resampled, Y_train_resampled = smotenc.fit_resample(X_train, Y_train)

# ==============================================================================
# PARTITION & PREVALENCE DIAGNOSTIC REPORT
# ==============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE SPLIT & RESAMPLING DIAGNOSTIC MATRIX")
print("="*80)

def compute_distribution(label, y_series):
    counts = y_series.value_counts()
    pcts = y_series.value_counts(normalize=True) * 100
    neg_count = counts.get(0, 0)
    pos_count = counts.get(1, 0)
    neg_pct = pcts.get(0, 0.0)
    pos_pct = pcts.get(1, 0.0)
    ratio = neg_count / pos_count if pos_count > 0 else 0
    return {
        'Dataset Partition': label,
        'Total N': len(y_series),
        'Negative N (0)': neg_count,
        'Negative %': f"{neg_pct:.2f}%",
        'Positive N (1)': pos_count,
        'Positive %': f"{pos_pct:.2f}%",
        'Imbalance Ratio': f"{ratio:.2f}:1"
    }

metrics = [
    compute_distribution("Original Full Dataset", Y),
    compute_distribution("Isolated Test Split (Step 3)", Y_test),
    compute_distribution("Original Train Split (Step 3)", Y_train),
    compute_distribution("Balanced Train Split (Step 4 SMOTE-NC)", Y_train_resampled)
]

report_df = pd.DataFrame(metrics)
print(report_df.to_string(index=False))
print("="*80)

# Verify that there are no fractional codes in the synthetic nominal variables
any_fractional = False
for col in nominal_features:
    unique_vals = X_train_resampled[col].unique()
    for val in unique_vals:
        # Check if the string cannot be parsed as a perfect integer
        try:
            int(val)
        except ValueError:
            any_fractional = True
            print(f"Warning: Fractional or invalid value detected in nominal variable {col}: {val}")

if not any_fractional:
    print("Verification Success: 100% of nominal categories in SMOTE-NC resampled training")
    print("set remain structurally valid integers (no fractional decimals generated).")
print("="*80 + "\n")

# Save partitions for downstream modeling (Step 5, 6, 7, 8)
# We will save:
# - X_train_resampled, Y_train_resampled (balanced training set)
# - X_test, Y_test (isolated testing/validation set)
X_train_resampled.to_csv("X_train_resampled.csv", index=False)
Y_train_resampled.to_csv("Y_train_resampled.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
Y_test.to_csv("Y_test.csv", index=False)

print("Saved files successfully:")
print(" - X_train_resampled.csv")
print(" - Y_train_resampled.csv")
print(" - X_test.csv")
print(" - Y_test.csv")
print("\nStep 3 and Step 4 Pipeline Stage Complete.")
