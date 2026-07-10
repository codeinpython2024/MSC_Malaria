import pandas as pd
from pre_process import final_ml_matrix

# ==============================================================================
# STEP 1: CATEGORICAL MAPPING, VALUE STABILIZATION, AND INDEX TRACKING
# ==============================================================================
print("Executing Step 1: Categorical Mapping, Value Stabilization, and Index Tracking...")

# 1. Continuous Vector Space Constraint
# Retain b19 (Child's Age in Months) strictly as a continuous integer array
# This preserves exact variance for thresholds (theta) in C4.5 split calculations
final_ml_matrix['b19'] = pd.to_numeric(final_ml_matrix['b19']).astype(int)

# 2. Ordinal Constraint Verification (v190)
# Preserve v190 (Wealth Quintile) strictly as its pre-calculated 1–5 categories
# without running manual PCA or altering its underlying demographic distribution.
if final_ml_matrix['v190'].dtype.name == 'category':
    # If loaded with Stata factor labels, sanitize string categories to prevent parsing anomalies
    final_ml_matrix['v190'] = final_ml_matrix['v190'].astype(str).str.strip()
else:
    # If loaded as raw integers, ensure strict integer type matching
    final_ml_matrix['v190'] = final_ml_matrix['v190'].astype(int)

# 3. Value Stabilization for Nominal Predictor Columns
# Convert nominal features into explicit, clean string representations. 
# This blocks white-box models (C4.5/RIPPER) from treating system codes as continuous values.
# Note: Redundant 'v717' and 'hv253' have been removed. New variables 'v119' and 'v158' have been included.
nominal_features = ['b4', 'v106', 'v127', 'v128', 'v129', 'v113', 'v116', 'v119', 'v158', 'hml20']

for col in nominal_features:
    final_ml_matrix[col] = final_ml_matrix[col].astype(str).str.strip()

# 4. Define Predictor Feature Matrix Structure (X)
# Establishes the exact sequence of columns for modeling, excluding tracking metadata and target (Y)
X_feature_names = [
    'b19',   # Continuous (Age)
    'b4',    # Nominal (Sex)
    'v106',  # Nominal (Maternal Education)
    'v190',  # Ordinal (Wealth Quintile - treated as non-continuous to avoid fractional artifacts)
    'v119',  # Nominal/Binary (Household Has Electricity)
    'v158',  # Nominal/Ordinal (Frequency of Listening to Radio)
    'v127',  # Nominal (Floor Material)
    'v128',  # Nominal (Wall Material)
    'v129',  # Nominal (Roof Material)
    'v113',  # Nominal (Water Source)
    'v116',  # Nominal (Toilet Type)
    'hml20'  # Nominal/Binary (LLIN Adherence)
]

# 5. Crucial Sub-step: Categorical Index Tracking Array
# Identifies and isolates every categorical/nominal feature name within the X matrix
categorical_features_for_smotenc = [
    'b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128', 'v129', 'v113', 'v116', 'hml20'
]

# Map names directly to their exact integer index positions relative to the X matrix
# This explicit array index is an absolute requirement for the SMOTENC constructor parameters
categorical_indices = [X_feature_names.index(col) for col in categorical_features_for_smotenc]

print("==============================================================================")
print(f"STEP 1 COMPLETE. Verified Feature Subspaces:")
print(f" - Continuous Features: ['b19']")
print(f" - Nominal/Categorical Features: {categorical_features_for_smotenc}")
print(f" - Tracked Categorical Indices for SMOTENC: {categorical_indices}")

# 6. Save the final stabilized ML matrix to CSV, overwriting the unstabilized version
output_csv = "final_malaria_sdoh_matrix.csv"
final_ml_matrix.to_csv(output_csv, index=False)
print(f" - Stabilized dataset successfully written to {output_csv}")
print("==============================================================================")
