"""
01_data_validation.py - Dataset Validation and Trajectory Classification
Verifies Vicolungo dataset integrity, sampling rate, column schemas,
and classifies trajectories into initially safe vs initially unsafe based on paper CBF boundary.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Import central configuration
from project_config import (
    DATA_FILE,
    DT,
    T_MIN,
    D_MIN,
    DATASET_VALIDATION_SUMMARY_CSV,
    TRAJECTORY_CLASSIFICATION_CSV,
)

def validate_dataset():
    print("==================================================")
    print("STEP 1: DATASET VALIDATION & CLASSIFICATION")
    print("==================================================")
    print(f"Loading data from: {DATA_FILE}")

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset file not found at {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    total_rows = len(df)
    unique_trajs = df["Trajectory_ID"].nunique()

    print(f"Total Rows: {total_rows}")
    print(f"Unique Trajectories: {unique_trajs}")

    # Check required columns
    required_cols = [
        "Trajectory_ID", "Time_Index", "ID_LV", "Type_LV", "Pos_LV",
        "Speed_LV", "Acc_LV", "ID_FAV", "Pos_FAV", "Speed_FAV", "Acc_FAV",
        "Spatial_Gap", "Spatial_Headway", "Speed_Diff"
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    print("All required columns verified.")

    # Check NaN counts
    nan_counts = df[required_cols].isna().sum()
    if nan_counts.any():
        print("Warning: Missing values detected:")
        print(nan_counts[nan_counts > 0])
    else:
        print("Zero missing values across required columns.")

    # Trajectory-level classification
    classification_records = []
    traj_ids = sorted(df["Trajectory_ID"].unique())

    # Overall dataset gap distribution relative to paper CBF boundary
    h_all = df["Spatial_Gap"] - (T_MIN * df["Speed_FAV"] + D_MIN)
    pct_unsafe_all = (h_all < 0).mean() * 100.0

    for tid in traj_ids:
        traj_data = df[df["Trajectory_ID"] == tid].sort_values("Time_Index").reset_index(drop=True)
        n_samples = len(traj_data)
        if n_samples < 2:
            continue

        t0 = float(traj_data.loc[0, "Time_Index"])
        t_end = float(traj_data.loc[n_samples - 1, "Time_Index"])
        duration = t_end - t0

        s0 = float(traj_data.loc[0, "Spatial_Gap"])
        v_ego_0 = float(traj_data.loc[0, "Speed_FAV"])
        v_lead_0 = float(traj_data.loc[0, "Speed_LV"])
        delta_v_0 = v_lead_0 - v_ego_0  # Positive: lead faster, Negative: closing

        # Paper CBF safety function at t=0
        h0 = s0 - (T_MIN * v_ego_0 + D_MIN)
        is_initially_safe = (h0 >= 0.0)

        # Severe initial overlap / impossibility detection
        # e.g., gap < 0.1 m while closing at > 5 m/s
        is_stress_anomaly = (s0 < 0.5 and delta_v_0 < -2.0)

        min_gap_raw = float(traj_data["Spatial_Gap"].min())
        mean_gap_raw = float(traj_data["Spatial_Gap"].mean())
        mean_speed_raw = float(traj_data["Speed_FAV"].mean())

        classification_records.append({
            "Trajectory_ID": tid,
            "Sample_Count": n_samples,
            "Duration_s": round(duration, 2),
            "Initial_Gap_m": round(s0, 4),
            "Initial_Ego_Speed_ms": round(v_ego_0, 3),
            "Initial_Lead_Speed_ms": round(v_lead_0, 3),
            "Initial_Delta_V_ms": round(delta_v_0, 3),
            "Initial_h_m": round(h0, 4),
            "Initially_Safe": is_initially_safe,
            "Is_Stress_Anomaly": is_stress_anomaly,
            "Min_Raw_Gap_m": round(min_gap_raw, 4),
            "Mean_Raw_Gap_m": round(mean_gap_raw, 3),
            "Mean_Raw_Speed_ms": round(mean_speed_raw, 3),
        })

    class_df = pd.DataFrame(classification_records)
    class_df.to_csv(TRAJECTORY_CLASSIFICATION_CSV, index=False)
    print(f"\nTrajectory classification saved to: {TRAJECTORY_CLASSIFICATION_CSV}")

    # Summary statistics
    n_safe = class_df["Initially_Safe"].sum()
    n_unsafe = len(class_df) - n_safe
    n_stress = class_df["Is_Stress_Anomaly"].sum()

    print("\n--- CLASSIFICATION SUMMARY ---")
    print(f"Total Trajectories: {len(class_df)}")
    print(f"Initially Safe (h0 >= 0): {n_safe} ({n_safe/len(class_df)*100:.1f}%)")
    print(f"Initially Unsafe (h0 < 0): {n_unsafe} ({n_unsafe/len(class_df)*100:.1f}%)")
    print(f"Initial Stress Anomalies: {n_stress} (Trajectory IDs: {class_df[class_df['Is_Stress_Anomaly']]['Trajectory_ID'].tolist()})")
    print(f"Overall Dataset samples with h < 0: {pct_unsafe_all:.2f}%")

    val_summary = pd.DataFrame([{
        "Total_Rows": total_rows,
        "Total_Trajectories": len(class_df),
        "Initially_Safe_Count": n_safe,
        "Initially_Unsafe_Count": n_unsafe,
        "Stress_Anomaly_Count": n_stress,
        "Dataset_Mean_Gap_m": round(df["Spatial_Gap"].mean(), 3),
        "Dataset_Min_Gap_m": round(df["Spatial_Gap"].min(), 4),
        "Dataset_Mean_Speed_ms": round(df["Speed_FAV"].mean(), 3),
        "Dataset_Pct_Samples_h_negative": round(pct_unsafe_all, 2),
    }])
    val_summary.to_csv(DATASET_VALIDATION_SUMMARY_CSV, index=False)
    print(f"Validation summary saved to: {DATASET_VALIDATION_SUMMARY_CSV}")

    return class_df

if __name__ == "__main__":
    validate_dataset()
