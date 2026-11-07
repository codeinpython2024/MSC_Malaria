import os
import sys
import subprocess
import time

python_exe = sys.executable

stages = [
    ("outputs/logs/step0_data_ingestion.log", "pre_process.py", "Stage 0: Data Ingestion & Geopolitical Filtering"),
    ("outputs/logs/step1_categorical_mapping.log", "first_stage.py", "Stage 1: Categorical Mapping & Value Stabilization"),
    ("outputs/logs/step2_baseline_diagnostics.log", "second_step.py", "Stage 2: Survey Weight Normalization & Baseline Diagnostics"),
    ("outputs/logs/step3_partition.log", "third_step.py", "Stage 3 & 4: Stratified Train-Test Partition & SMOTE-NC Resampling"),
    ("outputs/logs/step4_rule_induction.log", "fourth_step.py", "Stage 5: White-Box Rule Induction (C4.5 & RIPPER)"),
    ("outputs/logs/step5_black_box_baselines.log", "fifth_step.py", "Stage 6: Black-Box Benchmark Tuning (RF & SVM, SMOTE-NC inside CV folds)"),
    ("outputs/logs/step6_comparative_evaluation.log", "sixth_step.py", "Stage 7: Comparative Evaluation, ROC (C4.5 leaf scores) & Trade-off Matrix"),
    ("outputs/logs/step7_multiseed_stability.log", "step7_multiseed_stability.py", "Stage 8: Multi-Seed Stability Analysis (10 Seeds, all six models)"),
    ("outputs/logs/step8_grouped_cluster_sensitivity.log", "step8_grouped_cluster_sensitivity.py", "Stage 9: Grouped Household & Cluster Sensitivity (5-Fold CV, incl. C4.5 & RIPPER)"),
    ("outputs/logs/step9_rule_coverage_support.log", "step9_rule_coverage_support.py", "Stage 10: Rule Support & Coverage Audit (C4.5 leaves, RIPPER rules)"),
    ("outputs/logs/step10_bootstrap_ci.log", "run_bootstrap_ci_final.py", "Stage 11: Non-Parametric Bootstrap Evaluation (B=1000)"),
]

os.makedirs("outputs/logs", exist_ok=True)

# Optional: pass script names to run only those stages, e.g.
#   python generate_execution_logs.py pre_process.py
selected = sys.argv[1:]
unknown = [s for s in selected if s not in {script for _, script, _ in stages}]
if unknown:
    sys.exit(f"Unknown stage script(s): {unknown}")
run_stages = [s for s in stages if not selected or s[1] in selected]

print(f"Beginning Master Execution and Log Generation using {python_exe}...")
start_all = time.time()

for log_path, script_name, desc in run_stages:
    print(f"\n---> Running {desc} ({script_name})...")
    t0 = time.time()
    
    # Run the script
    result = subprocess.run(
        [python_exe, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    elapsed = time.time() - t0
    status_str = "SUCCESS" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"     Status: {status_str} in {elapsed:.2f}s | Writing to {log_path} ({len(result.stdout)} chars)")
    
    # Write execution log
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"================================================================================\n")
        f.write(f"EXECUTION LOG: {desc}\n")
        f.write(f"Script: {script_name}\n")
        f.write(f"Python Runtime: {sys.version.split()[0]} ({python_exe})\n")
        f.write(f"Execution Duration: {elapsed:.2f} seconds\n")
        f.write(f"Exit Code: {result.returncode}\n")
        f.write(f"================================================================================\n\n")
        f.write(result.stdout)
        
    if result.returncode != 0:
        print(f"ERROR: Execution of {script_name} failed! Aborting log generation.")
        sys.exit(1)

total_elapsed = time.time() - start_all
print(f"\n================================================================================")
print(f"{len(run_stages)} OF {len(stages)} STAGES EXECUTED AND LOGGED SUCCESSFULLY in {total_elapsed:.2f}s!")
print(f"Log files for these stages in outputs/logs/ are updated.")
print(f"================================================================================")
