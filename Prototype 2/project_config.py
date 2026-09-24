"""
project_config.py - Central Configuration for Prototype 2 CBF Pipeline
Research Paper: "Can Control Barrier Functions Keep Automated Vehicles Safe in Live Freeway Traffic?"
Authors: George Gunter, Matthew Nice, Matt Bunting, Jonathan Sprinkle, Daniel B. Work (ICCPS 2025)
"""

import os
from pathlib import Path

# ============================================================
# DIRECTORY & FILE PATHS
# ============================================================
PROTOTYPE2_DIR = Path(r"E:\CPS Project\Prototype 2")
DATA_FILE = PROTOTYPE2_DIR / "Vicolungo.csv"
RESULTS_DIR = PROTOTYPE2_DIR / "results"
FIGURES_DIR = PROTOTYPE2_DIR / "figures"

# Create directories if they do not exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Output files in results/
DATASET_VALIDATION_SUMMARY_CSV = RESULTS_DIR / "dataset_validation_summary.csv"
TRAJECTORY_CLASSIFICATION_CSV = RESULTS_DIR / "trajectory_classification.csv"

BASELINE_RESULTS_CSV = RESULTS_DIR / "baseline_results.csv"
BASELINE_SUMMARY_CSV = RESULTS_DIR / "baseline_summary.csv"

CBF_RESULTS_CSV = RESULTS_DIR / "cbf_results.csv"
CBF_SUMMARY_CSV = RESULTS_DIR / "cbf_summary.csv"

COMPARATIVE_METRICS_CSV = RESULTS_DIR / "comparative_metrics.csv"
STRESS_CASE_ANALYSIS_CSV = RESULTS_DIR / "stress_case_analysis.csv"

FINAL_REPORT_MD = PROTOTYPE2_DIR / "FINAL_RESEARCH_REPORT.md"

# ============================================================
# SIMULATION PARAMETERS
# ============================================================
DT = 0.1  # 10 Hz sampling frequency (0.1 s time step)

# ============================================================
# NOMINAL ACC CONTROLLER PARAMETERS
# ============================================================
D_DES = 5.0    # Standstill / minimum desired distance (m)
T_DES = 1.5    # Desired time gap (s)
K_GAP = 0.20   # Position/gap error feedback gain
K_REL = 0.80   # Relative velocity feedback gain

# ============================================================
# CBF PARAMETERS (EXACT FROM THE PAPER)
# ============================================================
# Safety function: h = gap - (T_MIN * ego_speed + D_MIN)
# Barrier condition: u_CBF <= (k * h + delta_v) / T_MIN
T_MIN = 2.0    # Minimum headway time parameter (s)
D_MIN = 15.0   # Minimum distance parameter (m)
K_CBF = 0.1    # Class-K comparison gain parameter (1/s)

# ============================================================
# VEHICLE PHYSICAL & ACTUATOR CONSTRAINTS
# ============================================================
A_MAX = 2.0    # Maximum longitudinal acceleration (m/s^2)
A_MIN = -4.0   # Maximum braking deceleration (m/s^2)
JERK_MAX = 1.5 # Maximum allowed jerk (|da/dt|) (m/s^3)

# ============================================================
# SAFETY & EVALUATION THRESHOLDS
# ============================================================
STRICT_COLLISION_GAP = 0.0  # Physical bump-to-bump contact (m)
NEAR_MISS_GAP = 0.5         # Critical near-miss threshold (m)
RECOVERY_EPSILONS = [0.1, 1.0, 2.0] # Proximity to safe set boundary h >= -eps (m)

# ============================================================
# STRESS & CASE STUDY TRAJECTORIES
# ============================================================
CRITICAL_TRAJECTORIES = [6, 16, 20, 51, 55, 56]

if __name__ == "__main__":
    print("Project configuration loaded successfully.")
    print(f"Data file: {DATA_FILE} (Exists: {DATA_FILE.exists()})")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Figures dir: {FIGURES_DIR}")
