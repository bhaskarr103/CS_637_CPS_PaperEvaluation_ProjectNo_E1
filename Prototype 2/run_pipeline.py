"""
run_pipeline.py - Master Pipeline Orchestrator for Prototype 2 CBF Research
Executes all pipeline stages in sequential order, verifying output integrity at each step.
"""

import sys
import subprocess
import time
from pathlib import Path

PYTHON_EXE = sys.executable
BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    ("Stage 0: Configuration Check", "00_project_config.py"),
    ("Stage 1: Dataset Validation & Classification", "01_data_validation.py"),
    ("Stage 2: Baseline ACC Simulation", "02_baseline_acc.py"),
    ("Stage 3: CBF Controller Unit Testing", "03_cbf_controller.py"),
    ("Stage 4: ACC + CBF Simulation", "04_cbf_simulation.py"),
    ("Stage 5: Scientific Comparative Evaluation", "05_cbf_evaluation.py"),
    ("Stage 6: Deep-Dive Stress Case Forensics", "06_stress_cases.py"),
    ("Stage 7: Parametric Sensitivity Sweep", "09_sensitivity_analysis.py"),
    ("Stage 8: Publication Figure Generation", "07_paper_figures.py"),
    ("Stage 9: Final Report Compilation", "08_final_report.py"),
]

def run_master_pipeline():
    print("=" * 70)
    print("STARTING COMPLETE PROTOTYPE 2 CBF RESEARCH PIPELINE (AUDITED)")
    print("=" * 70)
    t_start = time.time()

    for stage_name, script_name in SCRIPTS:
        script_path = BASE_DIR / script_name
        print(f"\n>>> Running {stage_name} ({script_name})...")
        t0 = time.time()
        res = subprocess.run([PYTHON_EXE, str(script_path)], cwd=BASE_DIR)
        elapsed = time.time() - t0

        if res.returncode != 0:
            print(f"\n[ERROR] Pipeline failed at {stage_name} (Exit code: {res.returncode})")
            sys.exit(res.returncode)
        print(f"--- Completed {stage_name} in {elapsed:.2f}s ---")

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"MASTER PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f}s!")
    print("All results, metrics, figures, and reports generated in Prototype 2/")
    print("=" * 70)

if __name__ == "__main__":
    run_master_pipeline()
