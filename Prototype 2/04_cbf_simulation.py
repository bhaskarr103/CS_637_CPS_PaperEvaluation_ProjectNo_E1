"""
04_cbf_simulation.py - Full ACC + Control Barrier Function (CBF) Simulation
Simulates the nominal ACC with CBF safety filter across all 61 trajectories of Vicolungo.
Features:
- Exact CBF safety boundary and linear comparison function: h = s - (2v + 15), u_CBF <= (0.1h + dv)/2
- Supervisor min-filter: u_cmd = min(u_nom, u_CBF)
- Actuator acceleration saturation [-4.0, +2.0] m/s^2
- Actuator jerk rate limiting (1.5 m/s^3)
- Tracking of intervention and infeasibility samples
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
    STRICT_COLLISION_GAP,
    NEAR_MISS_GAP,
    CBF_RESULTS_CSV,
    CBF_SUMMARY_CSV,
)
from cbf_controller import CBFSafetyFilter, compute_h

def run_cbf_simulation():
    print("==================================================")
    print("STEP 4: RUNNING ACC + CBF SIMULATION")
    print("==================================================")
    print(f"Loading data from: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    traj_ids = sorted(df["Trajectory_ID"].unique())
    print(f"Processing {len(traj_ids)} trajectories...")

    cbf_filter = CBFSafetyFilter()

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
        initial_h = compute_h(gap, ego_speed)
        is_init_safe = (initial_h >= 0.0)

        cbf_filter.reset(initial_acc=0.0)

        ts_time = []
        ts_ego_speed = []
        ts_lead_speed = []
        ts_gap = []
        ts_desired_gap = []
        ts_rel_speed = []
        ts_h = []
        ts_u_nom = []
        ts_u_cbf = []
        ts_u_raw = []
        ts_acc = []
        ts_ttc = []
        ts_intervention = []
        ts_infeasible = []

        for i in range(len(data)):
            row = data.iloc[i]
            t = float(row["Time_Index"])
            lead_speed = float(row["Speed_LV"])

            closing_speed = ego_speed - lead_speed
            delta_v = lead_speed - ego_speed

            # 1. Nominal ACC controller
            s_des = D_DES + T_DES * ego_speed
            gap_error = gap - s_des
            u_nom = K_GAP * gap_error + K_REL * delta_v
            u_nom_sat = np.clip(u_nom, A_MIN, A_MAX)

            # 2. CBF Safety Filter & Actuator Limiting
            filter_res = cbf_filter.filter_control(
                u_nom=u_nom_sat,
                gap=gap,
                ego_speed=ego_speed,
                lead_speed=lead_speed,
            )

            applied_acc = filter_res["applied_acc"]
            h = filter_res["h"]
            u_cbf = filter_res["u_cbf"]
            u_raw = filter_res["u_raw"]
            intervention = filter_res["intervention"]
            infeasible = filter_res["infeasible"]

            # 3. Time to Collision (TTC)
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
            ts_h.append(h)
            ts_u_nom.append(u_nom_sat)
            ts_u_cbf.append(u_cbf)
            ts_u_raw.append(u_raw)
            ts_acc.append(applied_acc)
            ts_ttc.append(ttc)
            ts_intervention.append(intervention)
            ts_infeasible.append(infeasible)

            # 4. Vehicle state update (forward Euler)
            ego_speed_next = max(0.0, ego_speed + applied_acc * DT)
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
            "h": ts_h,
            "u_nom": ts_u_nom,
            "u_cbf": ts_u_cbf,
            "u_raw": ts_u_raw,
            "Acceleration": ts_acc,
            "TTC": ts_ttc,
            "CBF_intervention": ts_intervention,
            "CBF_infeasible": ts_infeasible,
        })
        all_timeseries.append(traj_df)

        # Compute summary metrics
        min_gap = float(traj_df["Gap"].min())
        mean_gap = float(traj_df["Gap"].mean())
        min_h = float(traj_df["h"].min())
        max_violation = float(max(0.0, -min_h))
        mean_speed = float(traj_df["Ego_speed"].mean())
        mean_speed_error = float(np.mean(np.abs(traj_df["Ego_speed"] - traj_df["Lead_speed"])))
        max_braking = float(traj_df["Acceleration"].min())

        jerk_series = np.diff(traj_df["Acceleration"]) / DT
        max_jerk = float(np.max(np.abs(jerk_series))) if len(jerk_series) > 0 else 0.0

        finite_ttc = traj_df[np.isfinite(traj_df["TTC"])]["TTC"]
        min_ttc = float(finite_ttc.min()) if len(finite_ttc) > 0 else np.inf

        intervention_pct = float(traj_df["CBF_intervention"].mean() * 100.0)
        unsafe_samples = int((traj_df["h"] < 0.0).sum())
        infeasible_samples = int(traj_df["CBF_infeasible"].sum())

        strict_collision = (min_gap <= STRICT_COLLISION_GAP)
        near_miss = (min_gap <= NEAR_MISS_GAP)

        trajectory_summaries.append({
            "Trajectory_ID": tid,
            "Initial_gap": round(initial_gap, 4),
            "Initial_speed": round(initial_ego_speed, 3),
            "Initial_h": round(initial_h, 4),
            "Initially_Safe": is_init_safe,
            "Minimum_gap": round(min_gap, 4),
            "Mean_gap": round(mean_gap, 3),
            "Minimum_h": round(min_h, 4),
            "Max_Safety_Violation": round(max_violation, 4),
            "Minimum_TTC": round(min_ttc, 4) if np.isfinite(min_ttc) else np.inf,
            "Mean_speed": round(mean_speed, 3),
            "Mean_speed_error": round(mean_speed_error, 4),
            "Maximum_braking": round(max_braking, 3),
            "Maximum_abs_jerk": round(max_jerk, 3),
            "CBF_intervention_percent": round(intervention_pct, 2),
            "CBF_unsafe_samples": unsafe_samples,
            "CBF_infeasible_samples": infeasible_samples,
            "Collision": strict_collision,
            "Near_miss": near_miss,
        })

    # Combine and save
    full_results = pd.concat(all_timeseries, ignore_index=True)
    summary_df = pd.DataFrame(trajectory_summaries)

    full_results.to_csv(CBF_RESULTS_CSV, index=False)
    summary_df.to_csv(CBF_SUMMARY_CSV, index=False)

    print(f"\nDetailed time-series saved to: {CBF_RESULTS_CSV}")
    print(f"Summary metrics saved to: {CBF_SUMMARY_CSV}")

    # Print summary results
    print("\n--- ACC + CBF OVERALL RESULTS ---")
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
    print(f"Average CBF Intervention: {summary_df['CBF_intervention_percent'].mean():.2f}%")
    print(f"Total CBF Unsafe Samples: {summary_df['CBF_unsafe_samples'].sum()}")
    print(f"Total Infeasible CBF Samples: {summary_df['CBF_infeasible_samples'].sum()}")

    return full_results, summary_df

if __name__ == "__main__":
    run_cbf_simulation()
