import pandas as pd
# Flawlessly import stabilized structures from first_stage in-memory pipeline
from first_stage import final_ml_matrix, X_feature_names

# ==============================================================================
# STEP 2: SURVEY WEIGHT NORMALIZATION & BASELINE COMPLEX SURVEY DIAGNOSTICS
# ==============================================================================
print("Executing Step 2: Survey Weight Normalization & Baseline Complex Survey Diagnostics...")

# 1. Normalize the raw sample weight variable (v005)
# In DHS, analytical weights are stored with a 6-decimal implicit precision factor.
# Dividing by 1,000,000 isolates true fractional analytical weights.
final_ml_matrix['analytical_weight'] = final_ml_matrix['v005'] / 1000000.0

total_analytical_weight = final_ml_matrix['analytical_weight'].sum()
cohort_size = len(final_ml_matrix)

print(f"Total Cohort Size: {cohort_size} children.")
print(f"Sum of Analytical Weights: {total_analytical_weight:.4f}")

# Map of raw feature codes to human-readable names for presentation/publication clarity
feature_labels = {
    'b19': "Child's Age (Months)",
    'b4': "Child's Sex",
    'v106': "Maternal Education Level",
    'v190': "Wealth Quintile",
    'v119': "Household Has Electricity",
    'v158': "Frequency of Listening to Radio",
    'v127': "Main Floor Material",
    'v128': "Main Wall Material",
    'v129': "Main Roof Material",
    'v113': "Source of Drinking Water",
    'v116': "Type of Toilet Facility",
    'hml20': "Child Slept Under LLIN"
}

# Value mappings for cleaner diagnostic summaries (derived from MIS standard coding structures)
value_mappings = {
    'b4': { '1': 'Male', '2': 'Female' },
    'v106': { '0': 'No Education', '1': 'Primary', '2': 'Secondary', '3': 'Higher' },
    'v190': { '1': 'Poorest', '2': 'Poorer', '3': 'Middle', '4': 'Richer', '5': 'Richest' },
    'v119': { '0': 'No', '1': 'Yes' },
    'v158': { '0': 'Not at all', '1': 'Less than once a week', '2': 'At least once a week', '3': 'Almost every day' },
    'hml20': { '0': 'No', '1': 'Yes' },
    'v127': {
        '11': 'Earth/sand',
        '12': 'Dung',
        '21': 'Wood planks',
        '22': 'Palm/bamboo',
        '33': 'Ceramic tiles',
        '34': 'Cement',
        '96': 'Other'
    },
    'v128': {
        '11': 'No walls',
        '13': 'Dirt',
        '21': 'Bamboo with mud',
        '22': 'Stone with mud',
        '31': 'Cement',
        '33': 'Bricks',
        '34': 'Cement blocks',
        '96': 'Other'
    },
    'v129': {
        '12': 'Thatch/palm leaf',
        '13': 'Grass',
        '22': 'Palm/bamboo',
        '31': 'Metal/Zinc'
    },
    'v113': {
        '14': 'Public tap/standpipe',
        '21': 'Tube well/borehole',
        '31': 'Protected well',
        '32': 'Unprotected well',
        '43': 'Surface water (River/stream/canal)'
    },
    'v116': {
        '11': 'Flush to sewer',
        '12': 'Flush to septic tank',
        '21': 'VIP latrine',
        '22': 'Pit latrine with slab',
        '23': 'Pit latrine without slab/open pit',
        '31': 'No facility/bush/field'
    }
}

def get_readable_category(feature, cat):
    cat_str = str(cat)
    if feature in value_mappings and cat_str in value_mappings[feature]:
        return f"{cat_str} ({value_mappings[feature][cat_str]})"
    return cat_str

# ==============================================================================
# DELIVERABLE 1: CLASS DISTRIBUTION DIAGNOSTIC
# ==============================================================================
print("\n" + "="*80)
print("DELIVERABLE 1: TARGET CLASS DISTRIBUTION DIAGNOSTIC (hml32 - Malaria Microscopy)")
print("="*80)

# Calculate raw unweighted counts and percentages
raw_counts = final_ml_matrix['hml32'].value_counts()
raw_pcts = final_ml_matrix['hml32'].value_counts(normalize=True) * 100

# Calculate weighted counts (sum of weights) and percentages
weighted_sums = final_ml_matrix.groupby('hml32')['analytical_weight'].sum()
weighted_pcts = (weighted_sums / total_analytical_weight) * 100

target_diag = pd.DataFrame({
    'Raw Count': raw_counts,
    'Raw Pct (%)': raw_pcts.round(2),
    'Weighted Sum': weighted_sums.round(4),
    'Weighted Pct (%)': weighted_pcts.round(2)
})
target_diag.index = target_diag.index.map(lambda x: "Positive (1.0)" if x == 1.0 else "Negative (0.0)")
print(target_diag.to_string())
print("-"*80)
print(f"Mathematical Class Imbalance Ratio (Weighted): "
      f"{weighted_pcts[0.0]/weighted_pcts[1.0]:.2f}:1 (Negative to Positive)")
print("="*80 + "\n")

# ==============================================================================
# DELIVERABLE 2: WEIGHTED UNIVARIATE & BIVARIATE ANALYSIS
# ==============================================================================
print("="*80)
print("DELIVERABLE 2: WEIGHTED UNIVARIATE & BIVARIATE COMPREHENSIVE SDOH ANALYSIS")
print("="*80)

# Prepare continuous age 'b19' binning exclusively for clean reporting
# Group by developmental brackets: 6-11, 12-23, 24-35, 36-47, 48-59 months
age_bins = [5, 11, 23, 35, 47, 59]
age_labels = ['6-11 months', '12-23 months', '24-35 months', '36-47 months', '48-59 months']
report_matrix = final_ml_matrix.copy()
report_matrix['b19_binned'] = pd.cut(report_matrix['b19'], bins=age_bins, labels=age_labels)

# Iterate through every feature in X_feature_names for structured profiling
for feature in X_feature_names:
    print(f"\nFeature: {feature} - {feature_labels[feature]}")
    print("-"*60)
    
    # Define target column for grouping (use binned column for b19)
    group_col = 'b19_binned' if feature == 'b19' else feature
    
    # Calculate group-level raw counts and sums of weights
    group_raw_counts = report_matrix[group_col].value_counts()
    group_weighted_sums = report_matrix.groupby(group_col)['analytical_weight'].sum()
    group_weighted_pcts = (group_weighted_sums / total_analytical_weight) * 100
    
    # Calculate malaria prevalence per group category (hml32 == 1.0 positive rate)
    group_malaria_positive_weight = report_matrix[report_matrix['hml32'] == 1.0].groupby(group_col)['analytical_weight'].sum()
    # Fill missing values where a category might have zero positives
    group_malaria_positive_weight = group_malaria_positive_weight.reindex(group_weighted_sums.index, fill_value=0.0)
    
    # Weighted Prevalence calculation
    group_malaria_prevalence = (group_malaria_positive_weight / group_weighted_sums) * 100
    
    # Combine results into a single clean summary dataframe
    summary_df = pd.DataFrame({
        'Category Name': [get_readable_category(feature, cat) for cat in group_weighted_sums.index],
        'Raw Count': [group_raw_counts.get(cat, 0) for cat in group_weighted_sums.index],
        'Weighted Pct (%)': group_weighted_pcts.values.round(2),
        'Weighted Malaria Prevalence (%)': group_malaria_prevalence.values.round(2)
    })
    
    print(summary_df.to_string(index=False))
    print("-"*60)

print("\n" + "="*80)
print("STEP 2 DIAGNOSTICS COMPLETE. All baseline characteristics and complex survey design")
print("weighted statistics successfully mapped.")
print("="*80)

# ------------------------------------------------------------------------------
# DELIVERABLE 3: GRAPHICAL REPRESENTATION OF SDOH PREVALENCE TABLES
# ------------------------------------------------------------------------------
print("\n[3] Generating graphical representations of SDOH prevalence...")
try:
    import matplotlib.pyplot as plt
    import os
    
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)
    
    # We will plot 4 key variables: v190, v106, hml20, v127
    plot_features = ['v190', 'v106', 'hml20', 'v127']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    axes = axes.flatten()
    
    for idx, feature in enumerate(plot_features):
        ax = axes[idx]
        
        # Calculate weighted prevalence
        group_weighted_sums = final_ml_matrix.groupby(feature)['analytical_weight'].sum()
        group_malaria_positive_weight = final_ml_matrix[final_ml_matrix['hml32'] == 1.0].groupby(feature)['analytical_weight'].sum()
        group_malaria_positive_weight = group_malaria_positive_weight.reindex(group_weighted_sums.index, fill_value=0.0)
        group_malaria_prevalence = (group_malaria_positive_weight / group_weighted_sums) * 100
        
        categories = [value_mappings[feature].get(str(cat), str(cat)) for cat in group_weighted_sums.index]
        prevalences = group_malaria_prevalence.values
        
        bars = ax.bar(categories, prevalences, color='#1f77b4', edgecolor='none', alpha=0.85, width=0.5)
        
        # Style the subplots
        ax.set_title(f"Malaria Prevalence by {feature_labels[feature]}", fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel("Weighted Prevalence (%)", fontsize=10)
        ax.set_ylim(0, max(prevalences) * 1.15 if len(prevalences) > 0 and max(prevalences) > 0 else 10)
        ax.grid(True, linestyle=':', alpha=0.6, color='#cbcbcb', axis='y')
        
        # Add value labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')
                        
        # Rotate x labels if they are long
        if feature in ['v127', 'v106']:
            ax.set_xticklabels(categories, rotation=20, ha='right', fontsize=9)
        else:
            ax.set_xticklabels(categories, fontsize=9)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#888888')
        ax.spines['bottom'].set_color('#888888')
        
    plt.suptitle("Malaria Microscopy Prevalence by Social Determinants of Health (SDOH)\n(Rural Nasarawa State, Weighted Analysis)", 
                 fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot_path = os.path.join("outputs", "plots", "sdoh_prevalence.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f" - SDOH Prevalence graph successfully exported to: {plot_path}")
except Exception as e:
    print(f"Error plotting SDOH prevalence: {e}")
    import traceback
    traceback.print_exc()

