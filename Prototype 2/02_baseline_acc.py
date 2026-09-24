"""
02_baseline_acc.py - Baseline Adaptive Cruise Control (ACC) Simulation
Simulates the baseline ACC controller across all 61 trajectories of Vicolungo.
Features:
- Time-gap spacing policy
- Relative velocity damping
- Acceleration saturation [-4.0, +2.0] m/s^2
- Actuator jerk rate limiting (1.5 m/s^3)
- Non-negative speed constraint
- Clean single-pass state integration
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

from project_config import (
    DATA_FILE,
    DT,
    D_DES,
    T_DES,
    K_GAP,
    K_REL,
    A_MAX,
    A_MIN,
    JERK_MAX,
    STRICT_COLLISION_GAP,
    NEAR_MISS_GAP,
    BASELINE_RESULTS_CSV,
    BASELINE_SUMMARY_CSV,
)

def run_baseline_acc():
    print("==================================================")
    print("STEP 2: RUNNING PROTOTYPE 2 BASELINE ACC SIMULATION")
    print("==================================================")
    print(f"Loading data from: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    traj_ids = sorted(df["Trajectory_ID"].unique())
    print(f"Processing {len(traj_ids)} trajectories...")

    all_timeseries = []
    trajectory_summaries = []

    for tid in traj_ids:
        data = df[df["Trajectory_ID"] == tid].sort_values("Time_Index").reset_index(drop=True)
        if len(data) < 2:
            continue

        ego_speed = float(data.loc[0, "Speed_FAV"])
        gap = float(data.loc[0, "Spatial_Gap"])
        initial_gap = gap
        initial_ego_speed = ego_speed
        prev_acc = 0.0

        ts_time = []
        ts_ego_speed = []
        ts_lead_speed = []
        ts_gap = []
        ts_desired_gap = []
        ts_rel_speed = []
        ts_acc = []
        ts_ttc = []

        max_acc_change = JERK_MAX * DT

        for i in range(len(data)):
            row = data.iloc[i]
            t = float(row["Time_Index"])
            lead_speed = float(row["Speed_LV"])

            # Relative velocity (Positive: ego faster than lead -> closing)
            closing_speed = ego_speed - lead_speed
            # Relative speed delta_v (lead - ego)
            delta_v = lead_speed - ego_speed

            # Desired spacing
            s_des = D_DES + T_DES * ego_speed
            gap_error = gap - s_des

            # Nominal ACC command
            # Accelerate when gap > s_des, brake when closing fast
            u_nom = K_GAP * gap_error + K_REL * delta_v

            # Acceleration saturation
            u_sat = np.clip(u_nom, A_MIN, A_MAX)

            # Jerk limitation
            acc_change = np.clip(u_sat - prev_acc, -max_acc_change, max_acc_change)
            acc = prev_acc + acc_change
            prev_acc = acc

            # Time to Collision (TTC) - only defined when closing on lead and gap > 0
            if closing_speed > 0.0 and gap > 0.0:
                ttc = gap / closing_speed
            else:
                ttc = np.inf

            # Store step
            ts_time.append(t)
            ts_ego_speed.append(ego_speed)
            ts_lead_speed.append(lead_speed)
            ts_gap.append(gap)
            ts_desired_gap.append(s_des)
            ts_rel_speed.append(delta_v)
            ts_acc.append(acc)
            ts_ttc.append(ttc)

            # Vehicle state update (forward Euler)
            ego_speed_next = max(0.0, ego_speed + acc * DT)
            gap_next = gap + (lead_speed - ego_speed) * DT

            ego_speed = ego_speed_next
            gap = gap_next

        # Build trajectory dataframe
        traj_df = pd.DataFrame({
            "Trajectory_ID": tid,
            "Time": ts_time,
            "Lead_speed": ts_lead_speed,
            "Ego_speed": ts_ego_speed,
            "Gap": ts_gap,
            "Desired_gap": ts_desired_gap,
            "Relative_speed": ts_rel_speed,
            "Acceleration": ts_acc,
            "TTC": ts_ttc,
        })
        all_timeseries.append(traj_df)

        # Compute summary metrics
        min_gap = float(traj_df["Gap"].min())
        mean_gap = float(traj_df["Gap"].mean())
        mean_speed = float(traj_df["Ego_speed"].mean())
        mean_speed_error = float(np.mean(np.abs(traj_df["Ego_speed"] - traj_df["Lead_speed"])))
        max_braking = float(traj_df["Acceleration"].min())

        jerk_series = np.diff(traj_df["Acceleration"]) / DT
        max_jerk = float(np.max(np.abs(jerk_series))) if len(jerk_series) > 0 else 0.0

        finite_ttc = traj_df[np.isfinite(traj_df["TTC"])]["TTC"]
        min_ttc = float(finite_ttc.min()) if len(finite_ttc) > 0 else np.inf

        strict_collision = (min_gap <= STRICT_COLLISION_GAP)
        near_miss = (min_gap <= NEAR_MISS_GAP)

        trajectory_summaries.append({
            "Trajectory_ID": tid,
            "Initial_gap": round(initial_gap, 4),
            "Initial_speed": round(initial_ego_speed, 3),
            "Minimum_gap": round(min_gap, 4),
            "Mean_gap": round(mean_gap, 3),
            "Minimum_TTC": round(min_ttc, 4) if np.isfinite(min_ttc) else np.inf,
            "Mean_speed": round(mean_speed, 3),
            "Mean_speed_error": round(mean_speed_error, 4),
            "Maximum_braking": round(max_braking, 3),
            "Maximum_abs_jerk": round(max_jerk, 3),
            "Collision": strict_collision,
            "Near_miss": near_miss,
        })

    # Combine and save
    full_results = pd.concat(all_timeseries, ignore_index=True)
    summary_df = pd.DataFrame(trajectory_summaries)

    full_results.to_csv(BASELINE_RESULTS_CSV, index=False)
    summary_df.to_csv(BASELINE_SUMMARY_CSV, index=False)

    print(f"\nDetailed time-series saved to: {BASELINE_RESULTS_CSV}")
    print(f"Summary metrics saved to: {BASELINE_SUMMARY_CSV}")

    # Print summary results
    print("\n--- BASELINE ACC OVERALL RESULTS ---")
    print(f"Trajectories: {len(summary_df)}")
    print(f"Strict Collisions (gap <= 0m): {summary_df['Collision'].sum()}")
    print(f"Near-misses (gap <= 0.5m): {summary_df['Near_miss'].sum()}")
    print(f"Overall Minimum Gap: {summary_df['Minimum_gap'].min():.4f} m (Trajectory {summary_df.loc[summary_df['Minimum_gap'].idxmin(), 'Trajectory_ID']})")
    
    finite_summary_ttc = summary_df[np.isfinite(summary_df["Minimum_TTC"])]["Minimum_TTC"]
    min_ttc_overall = finite_summary_ttc.min() if len(finite_summary_ttc) > 0 else np.inf
    print(f"Overall Minimum TTC: {min_ttc_overall:.4f} s")
    print(f"Average Gap: {summary_df['Mean_gap'].mean():.2f} m")
    print(f"Average Ego Speed: {summary_df['Mean_speed'].mean():.2f} m/s")
    print(f"Average Speed Error: {summary_df['Mean_speed_error'].mean():.4f} m/s")
    print(f"Maximum Braking: {summary_df['Maximum_braking'].min():.2f} m/s^2")
    print(f"Maximum Jerk: {summary_df['Maximum_abs_jerk'].max():.2f} m/s^3")
    print(f"Collision Rate: {summary_df['Collision'].mean() * 100:.3f}%")

    return full_results, summary_df

if __name__ == "__main__":
    run_baseline_acc()
