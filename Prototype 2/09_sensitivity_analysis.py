"""
09_sensitivity_analysis.py - Parametric Sensitivity Analysis of T_min
Evaluates the CBF safety supervisor across T_min in [1.0, 1.5, 2.0, 2.5] s.
Holds all other parameters fixed (D_min = 15m, k = 0.1, ACC T_des = 1.5s, D_des = 5m).
Saves results to results/sensitivity_summary.csv.
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
    D_MIN,
    RESULTS_DIR,
)
from cbf_controller import CBFSafetyFilter, compute_h

SENSITIVITY_CSV = RESULTS_DIR / "sensitivity_summary.csv"

def run_sensitivity_analysis():
    print("==================================================")
    print("STEP 9: PARAMETRIC SENSITIVITY ANALYSIS (T_MIN)")
    print("==================================================")

    df = pd.read_csv(DATA_FILE)
    traj_ids = sorted(df["Trajectory_ID"].unique())

    t_min_values = [1.0, 1.5, 2.0, 2.5]
    results_records = []

    for t_min in t_min_values:
        print(f"Simulating pipeline for T_min = {t_min:.1f} s...")
        cbf_filter = CBFSafetyFilter(t_min=t_min)
        traj_summaries = []

        for tid in traj_ids:
            data = df[df["Trajectory_ID"] == tid].sort_values("Time_Index").reset_index(drop=True)
            if len(data) < 2:
                continue

            ego_speed = float(data.loc[0, "Speed_FAV"])
            gap = float(data.loc[0, "Spatial_Gap"])
            h0 = compute_h(gap, ego_speed, t_min=t_min, d_min=D_MIN)
            is_safe0 = (h0 >= 0.0)

            cbf_filter.reset(0.0)

            gaps = []
            ego_speeds = []
            lead_speeds = []
            interventions = []
            infeasibles = []
            h_vals = []

            for i in range(len(data)):
                row = data.iloc[i]
                lead_speed = float(row["Speed_LV"])
                delta_v = lead_speed - ego_speed

                s_des = D_DES + T_DES * ego_speed
                gap_error = gap - s_des
                u_nom = np.clip(K_GAP * gap_error + K_REL * delta_v, A_MIN, A_MAX)

                res = cbf_filter.filter_control(u_nom, gap, ego_speed, lead_speed)
                acc = res["applied_acc"]

                gaps.append(gap)
                ego_speeds.append(ego_speed)
                lead_speeds.append(lead_speed)
                interventions.append(res["intervention"])
                infeasibles.append(res["infeasible"])
                h_vals.append(res["h"])

                ego_speed_next = max(0.0, ego_speed + acc * DT)
                gap_next = gap + (lead_speed - ego_speed) * DT
                ego_speed = ego_speed_next
                gap = gap_next

            min_gap = min(gaps)
            mean_gap = np.mean(gaps)
            mean_speed = np.mean(ego_speeds)
            speed_err = np.mean(np.abs(np.array(ego_speeds) - np.array(lead_speeds)))
            interv_pct = np.mean(interventions) * 100.0
            inf_count = sum(infeasibles)
            max_h = max(h_vals)
            recovered = (max_h >= -0.1) if not is_safe0 else True
            collision = (min_gap <= 0.0)

            traj_summaries.append({
                "tid": tid,
                "is_safe0": is_safe0,
                "min_gap": min_gap,
                "mean_gap": mean_gap,
                "mean_speed": mean_speed,
                "speed_err": speed_err,
                "interv_pct": interv_pct,
                "inf_count": inf_count,
                "recovered": recovered,
                "collision": collision,
            })

        tdf = pd.DataFrame(traj_summaries)
        clean_tdf = tdf[tdf["tid"] != 6]

        init_safe_count = int(tdf["is_safe0"].sum())
        init_unsafe_count = len(tdf) - init_safe_count
        unsafe_tdf = tdf[~tdf["is_safe0"]]
        rec_rate = float(unsafe_tdf["recovered"].mean() * 100.0) if len(unsafe_tdf) > 0 else 100.0

        results_records.append({
            "T_min_s": t_min,
            "Initially_Safe_Count": init_safe_count,
            "Initially_Safe_Pct": round(init_safe_count / 61 * 100.0, 1),
            "Intervention_Rate_Pct": round(float(tdf["interv_pct"].mean()), 2),
            "Mean_Gap_m": round(float(tdf["mean_gap"].mean()), 2),
            "Clean_Min_Gap_m": round(float(clean_tdf["min_gap"].min()), 2),
            "Mean_Speed_ms": round(float(tdf["mean_speed"].mean()), 2),
            "Speed_Tracking_Error_ms": round(float(tdf["speed_err"].mean()), 4),
            "Infeasible_Samples": int(tdf["inf_count"].sum()),
            "Recovery_Rate_01m_Pct": round(rec_rate, 1),
            "Collisions_All_61": int(tdf["collision"].sum()),
            "Collisions_Valid_60": int(clean_tdf["collision"].sum()),
        })

    summary_df = pd.DataFrame(results_records)
    summary_df.to_csv(SENSITIVITY_CSV, index=False)
    print(f"\nSensitivity summary saved to: {SENSITIVITY_CSV}")

    print("\n--- SENSITIVITY ANALYSIS RESULTS ---")
    print(summary_df.to_string(index=False))

    return summary_df

if __name__ == "__main__":
    run_sensitivity_analysis()
