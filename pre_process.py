import pandas as pd
import os

# ==============================================================================
# PHASE 1: DATA INGESTION
# ==============================================================================
kr_path = os.path.join("NGKR81DT", "NGKR81FL.DTA")
pr_path = os.path.join("NGPR81DT", "NGPR81FL.DTA")

print(f"Loading files: {kr_path} and {pr_path}...")
kr_df = pd.read_stata(kr_path, convert_categoricals=False)  # Child Recode (Baseline)
pr_df = pd.read_stata(pr_path, convert_categoricals=False)  # Household Member Recode

# ==============================================================================
# PHASE 2: SUBSETTING PROTOCOL (DEMOGRAPHIC & RESIDENCY)
# ==============================================================================
print("Executing demographic and residency subsetting...")

# 1. Biological Survival Filter: Keep only children who are currently alive (b5 == 1)
kr_cohort = kr_df[kr_df['b5'] == 1].copy()

# 2. Household Residency Filter: Keep only children physically living in the surveyed dwelling
kr_cohort = kr_cohort.dropna(subset=['b16'])
kr_cohort = kr_cohort[kr_cohort['b16'] > 0]

# 3. Biological Age Parameter: Restrict to children 6 to 59 months old
kr_cohort = kr_cohort[(kr_cohort['b19'] >= 6) & (kr_cohort['b19'] <= 59)]

# ==============================================================================
# PHASE 3: SUBSETTING PROTOCOL (GEOPOLITICAL)
# ==============================================================================
print("Applying geopolitical spatial constraints...")

# 1. State Filter: Isolate Nasarawa State (v024 == 15 represents Nasarawa in 2021 MIS)
kr_cohort = kr_cohort[kr_cohort['v024'] == 15]

# 2. Topographical Filter: Isolate Rural communities (v025 == 2)
kr_cohort = kr_cohort[kr_cohort['v025'] == 2]

print(f"Total target cohort isolated: {len(kr_cohort)} rural Nasarawa children.")

# ==============================================================================
# PHASE 4: RELATIONAL DATA MERGE
# ==============================================================================
print("Mapping relational bridging keys and executing Left Join...")

# 1. Standardize composite tracking keys in the KR dataframe
kr_cohort = kr_cohort.rename(columns={
    'v001': 'hv001',   # Cluster number
    'v002': 'hv002',   # Household number
    'b16': 'hvidx'     # Child's roster line index mapping
})

# Cast merge keys to matching integer types to guarantee flawless left-join mapping
kr_cohort['hv001'] = kr_cohort['hv001'].astype(int)
kr_cohort['hv002'] = kr_cohort['hv002'].astype(int)
kr_cohort['hvidx'] = kr_cohort['hvidx'].astype(int)

# 2. Extract targeted malaria and vector features simultaneously from the PR dataframe
pr_cols = ['hv001', 'hv002', 'hvidx', 'hml20', 'hml32']
pr_malaria_vector = pr_df[pr_cols].copy()

pr_malaria_vector['hv001'] = pr_malaria_vector['hv001'].astype(int)
pr_malaria_vector['hv002'] = pr_malaria_vector['hv002'].astype(int)
pr_malaria_vector['hvidx'] = pr_malaria_vector['hvidx'].astype(int)

# 3. Execute Left Join to build the core matrix
merged_matrix = pd.merge(
    kr_cohort, 
    pr_malaria_vector, 
    on=['hv001', 'hv002', 'hvidx'], 
    how='left'
)

# ==============================================================================
# PHASE 5: DATA CLEANSING FOR C4.5 / RIPPER
# ==============================================================================
print("Cleaning target class and algorithmic predictors...")

# 1. Clean Target Class (Y) - hml32 (Malaria Microscopy)
# Retains definitive Negative (0) and Positive (1) test results. Drops untested/indeterminate.
merged_matrix = merged_matrix[merged_matrix['hml32'].isin([0, 1])]

# 2. Clean Predictor (X) - hml20 (Child Slept Under LLIN)
merged_matrix = merged_matrix[merged_matrix['hml20'].isin([0, 1])]

# ==============================================================================
# PHASE 6: FINAL SDOH FEATURE EXTRACTION
# ==============================================================================
print("Extracting final SDOH theoretical feature space...")

# Define the exact columns operationalized in Chapter 3 (Table 3.1)
sdoh_features = [
    # Administrative & Weights
    'hv001', 'hv002', 'hvidx', 'v005', 
    
    # Target Variable (Y)
    'hml32', 
    
    # Biological Profile
    'b19',   # Child's Age in Months
    'b4',    # Child's Sex
    
    # Maternal Attributes
    'v106',  # Highest Educational Level
    
    # Socioeconomic Status
    'v190',  # Household Wealth Quintile
    'v119',  # Household Has Electricity
    'v158',  # Frequency of Listening to Radio
    
    # Dwelling Architecture
    'v127',  # Main Floor Material
    'v128',  # Main Wall Material
    'v129',  # Main Roof Material
    
    # Sanitation & Environment
    'v113',  # Source of Drinking Water
    'v116',  # Type of Toilet Facility
    
    # Preventative Vector Control Adherence (Merged from PR)
    'hml20'  # Child Slept Under LLIN
]

# Isolate the final dataset ready for algorithmic processing
final_ml_matrix = merged_matrix[sdoh_features].copy()

# Reset index for a clean dataframe
final_ml_matrix = final_ml_matrix.reset_index(drop=True)

# Save the final cleaned ML matrix to CSV
pre_process_csv = "pre_processed_malaria_sdoh_matrix.csv"
final_ml_matrix.to_csv(pre_process_csv, index=False)

print("==============================================================================")
print(f"PIPELINE COMPLETE. Final ML Matrix shape: {final_ml_matrix.shape}")
print(f"Successfully saved final ML matrix to {pre_process_csv}")
print("Ready for categorical discretization and rule-based induction.")
print("==============================================================================")
