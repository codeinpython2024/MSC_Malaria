import os
import sys
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

# Ensure output directories exist
os.makedirs(os.path.join("outputs", "models"), exist_ok=True)

# ==============================================================================
# STEP 7: CONSTRUCT BLACK-BOX BENCHMARKING BASELINES
# ==============================================================================

print("="*80)
print("STAGE 7 PIPELINE: HYPERPARAMETER OPTIMIZATION OF BLACK-BOX BASELINES")
print("="*80)

# ------------------------------------------------------------------------------
# 1. INGEST PRE-PARTITIONED AND OVERSAMPLED DATASETS
# ------------------------------------------------------------------------------
print("\n[1/5] Ingesting resampled train and validation test splits...")
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
# 2. DEFINE PREPROCESSING PIPELINE (SCALING & ONE-HOT ENCODING)
# ------------------------------------------------------------------------------
print("\n[2/5] Setting up column preprocessing pipelines...")
continuous_feature = ['b19']  # Child's Age (continuous)
nominal_features = ['b4', 'v106', 'v190', 'v119', 'v158', 'v127', 'v128', 'v129', 'v113', 'v116', 'hml20']

# Ensure variables are cast appropriately: continuous as float, nominals as str/object
X_train['b19'] = X_train['b19'].astype(float)
X_test['b19'] = X_test['b19'].astype(float)

for col in nominal_features:
    X_train[col] = X_train[col].astype(int).astype(str)
    X_test[col] = X_test[col].astype(int).astype(str)

# Preprocessor transformer definition
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), continuous_feature),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), nominal_features)
    ]
)
print(" - Preprocessing pipeline configured: Continuous scaled, Nominal one-hot encoded.")

# ------------------------------------------------------------------------------
# 3. CONSTRUCT & GRID-SEARCH OPTIMIZED RANDOM FOREST BASELINE
# ------------------------------------------------------------------------------
print("\n" + "-"*80)
print("OPTIMIZING BLACK-BOX MODEL 1: RANDOM FOREST ENSEMBLE CLASSIFIER")
print("-"*80)

# Build Random Forest pipeline
rf_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ]
)

# Define Random Forest hyperparameter grid
rf_param_grid = {
    'classifier__n_estimators': [50, 100, 200, 300],
    'classifier__max_depth': [None, 5, 10, 15],
    'classifier__min_samples_split': [2, 5, 10]
}

print("Running 5-fold Stratified GridSearchCV for Random Forest...")
rf_grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    cv=5,
    scoring='f1_macro',  # Optimize macro F1 due to target clinical imbalance
    n_jobs=-1,
    verbose=1
)

rf_grid_search.fit(X_train, Y_train)
print(f"\nRandom Forest Optimization Complete.")
print(f"Best cross-validated parameters: {rf_grid_search.best_params_}")
print(f"Best cross-validated score (Macro F1): {rf_grid_search.best_score_:.4f}")

# Evaluate on validation test set
best_rf = rf_grid_search.best_estimator_
rf_preds = best_rf.predict(X_test)

print("\nRandom Forest Test Set Evaluation:")
print("Confusion Matrix:")
print(confusion_matrix(Y_test, rf_preds))
print("\nClassification Report:")
print(classification_report(Y_test, rf_preds, target_names=['Negative', 'Positive']))

# Serialize optimal Random Forest pipeline
rf_model_path = os.path.join("outputs", "models", "random_forest_pipeline.pkl")
try:
    with open(rf_model_path, 'wb') as f:
        pickle.dump(best_rf, f)
    print(f" - Persistent pipeline successfully saved to: {rf_model_path}")
except Exception as e:
    print(f"Error serializing Random Forest model: {e}")

# ------------------------------------------------------------------------------
# 4. CONSTRUCT & GRID-SEARCH OPTIMIZED SUPPORT VECTOR MACHINE BASELINE
# ------------------------------------------------------------------------------
print("\n" + "-"*80)
print("OPTIMIZING BLACK-BOX MODEL 2: SUPPORT VECTOR MACHINE (SVC)")
print("-"*80)

# Build SVM pipeline
svm_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True, random_state=42))  # probability=True for downstream ROC calculations
    ]
)

# Define SVM hyperparameter grid
svm_param_grid = {
    'classifier__C': [0.1, 1, 10, 100],
    'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],
    'classifier__kernel': ['rbf', 'linear', 'poly']
}

print("Running 5-fold Stratified GridSearchCV for Support Vector Machine...")
svm_grid_search = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=svm_param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

svm_grid_search.fit(X_train, Y_train)
print(f"\nSupport Vector Machine Optimization Complete.")
print(f"Best cross-validated parameters: {svm_grid_search.best_params_}")
print(f"Best cross-validated score (Macro F1): {svm_grid_search.best_score_:.4f}")

# Evaluate on validation test set
best_svm = svm_grid_search.best_estimator_
svm_preds = best_svm.predict(X_test)

print("\nSupport Vector Machine Test Set Evaluation:")
print("Confusion Matrix:")
print(confusion_matrix(Y_test, svm_preds))
print("\nClassification Report:")
print(classification_report(Y_test, svm_preds, target_names=['Negative', 'Positive']))

# Serialize optimal SVM pipeline
svm_model_path = os.path.join("outputs", "models", "svm_pipeline.pkl")
try:
    with open(svm_model_path, 'wb') as f:
        pickle.dump(best_svm, f)
    print(f" - Persistent pipeline successfully saved to: {svm_model_path}")
except Exception as e:
    print(f"Error serializing SVM model: {e}")

print("\n" + "="*80)
print("STAGE 7 BASELINE CLASSIFIER COMPLETE AND PERSISTED")
print("="*80)
