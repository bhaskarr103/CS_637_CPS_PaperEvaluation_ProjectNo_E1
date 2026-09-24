"""
07_paper_figures.py - Publication-Quality Figure Generation
Generates publication-ready figures for the research report:
  - Fig 1: Critical trajectory time-series comparisons (Baseline vs CBF)
  - Fig 2: Forward invariance demonstration across initially safe trajectories
  - Fig 3: Recovery dynamics across initially unsafe trajectories
  - Fig 4: Trajectory 6 forensic analysis (initial condition & actuator saturation)
  - Fig 5: Statistical distributions (Minimum Gap, TTC, Maximum Braking)
  - Fig 6: Safety boundary vs ACC nominal setpoint vs human gap distribution
  - Fig 7: Paired trajectory delta comparison (Safety margin vs Speed impact)
  - Fig 8: Parametric sensitivity curves across T_min in [1.0, 1.5, 2.0, 2.5] s
"""

import sys
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to prevent GUI blocking
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from project_config import (
    DATA_FILE,
    BASELINE_RESULTS_CSV,
    CBF_RESULTS_CSV,
    COMPARATIVE_METRICS_CSV,
    FIGURES_DIR,
    RESULTS_DIR,
    T_MIN,
    D_MIN,
    T_DES,
    D_DES,
    A_MIN,
    A_MAX,
)

SENSITIVITY_CSV = RESULTS_DIR / "sensitivity_summary.csv"
PAIRED_CLASSIFICATION_CSV = RESULTS_DIR / "paired_trajectory_classification.csv"

# Styling for publication quality
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.3,
})

def generate_all_figures():
    print("==================================================")
    print("STEP 7: GENERATING PUBLICATION FIGURES (AUDITED)")
    print("==================================================")

    base_ts = pd.read_csv(BASELINE_RESULTS_CSV)
    cbf_ts = pd.read_csv(CBF_RESULTS_CSV)
    comp_metrics = pd.read_csv(COMPARATIVE_METRICS_CSV)
    raw_df = pd.read_csv(DATA_FILE)

    # ------------------------------------------------------------
    # FIGURE 1: CRITICAL TRAJECTORY TIME-SERIES (Traj 16 & 55)
    # ------------------------------------------------------------
    print("Generating Figure 1: Critical Trajectories Time Series...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    t16_b = base_ts[base_ts["Trajectory_ID"] == 16].sort_values("Time")
    t16_c = cbf_ts[cbf_ts["Trajectory_ID"] == 16].sort_values("Time")

    axes[0, 0].plot(t16_b["Time"], t16_b["Gap"], "r--", label="Baseline ACC Gap")
    axes[0, 0].plot(t16_c["Time"], t16_c["Gap"], "b-", label="ACC + CBF Gap")
    axes[0, 0].axhline(0, color="black", linestyle=":", label="Collision Boundary")
    axes[0, 0].set_title("Trajectory 16 (Close Following): Longitudinal Gap")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Gap (m)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(t16_c["Time"], t16_c["u_nom"], "g--", alpha=0.7, label="Nominal ACC Command")
    axes[0, 1].plot(t16_c["Time"], t16_c["u_cbf"], "m:", label="Raw CBF Limit")
    axes[0, 1].plot(t16_c["Time"], t16_c["Acceleration"], "b-", label="Applied Acceleration")
    axes[0, 1].axhline(A_MIN, color="gray", linestyle="--", label="Max Braking (-4 m/s²)")
    axes[0, 1].set_title("Trajectory 16: Control Acceleration & Actuator Saturation")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Acceleration (m/s²)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    t55_b = base_ts[base_ts["Trajectory_ID"] == 55].sort_values("Time")
    t55_c = cbf_ts[cbf_ts["Trajectory_ID"] == 55].sort_values("Time")

    axes[1, 0].plot(t55_b["Time"], t55_b["Gap"], "r--", label="Baseline ACC Gap")
    axes[1, 0].plot(t55_c["Time"], t55_c["Gap"], "b-", label="ACC + CBF Gap")
    axes[1, 0].set_title("Trajectory 55 (Initially Safe): Longitudinal Gap Expansion")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Gap (m)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(t55_c["Time"], t55_c["h"], "b-", label="CBF Barrier Value h(t)")
    axes[1, 1].axhline(0, color="red", linestyle="--", label="Safe Boundary (h = 0)")
    axes[1, 1].set_title("Trajectory 55: Forward Invariance Verification (h >= 0)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Safety Function h (m)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    fig1_path = FIGURES_DIR / "fig1_critical_trajectories.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # ------------------------------------------------------------
    # FIGURE 2: FORWARD INVARIANCE (All 9 Initially Safe Trajectories)
    # ------------------------------------------------------------
    print("Generating Figure 2: Forward Invariance Demonstration...")
    plt.figure(figsize=(10, 6))
    safe_ids = comp_metrics[comp_metrics["Initially_Safe"]]["Trajectory_ID"].tolist()

    for tid in safe_ids:
        t_data = cbf_ts[cbf_ts["Trajectory_ID"] == tid].sort_values("Time")
        t_rel = t_data["Time"] - t_data["Time"].iloc[0]
        plt.plot(t_rel, t_data["h"], label=f"Traj {tid} (h0={t_data['h'].iloc[0]:.1f}m)")

    plt.axhline(0, color="red", linestyle="--", linewidth=2.0, label="Safety Boundary h = 0")
    plt.title("Empirical Forward Invariance: Safety Barrier Evolution for Initially Safe Trajectories (N=9)")
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("CBF Safety Function h(t) = s - (2v + 15) (m)")
    plt.legend(ncol=2)
    plt.grid(True)
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "fig2_forward_invariance.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # ------------------------------------------------------------
    # FIGURE 3: RECOVERY DYNAMICS (Initially Unsafe Trajectories)
    # ------------------------------------------------------------
    print("Generating Figure 3: Recovery Analysis...")
    plt.figure(figsize=(10, 6))
    sample_unsafe_ids = [0, 1, 2, 4, 9, 12, 16, 20, 24, 38, 51]

    for tid in sample_unsafe_ids:
        t_data = cbf_ts[cbf_ts["Trajectory_ID"] == tid].sort_values("Time")
        t_rel = t_data["Time"] - t_data["Time"].iloc[0]
        plt.plot(t_rel, t_data["h"], label=f"Traj {tid} (h0={t_data['h'].iloc[0]:.1f}m)")

    plt.axhline(0, color="black", linestyle="--", linewidth=2.0, label="Safe Set Boundary (h = 0)")
    plt.axhline(-1.0, color="gray", linestyle=":", label="Tolerance Boundary (h = -1m)")
    plt.title("Recovery Dynamics: Asymptotic Convergence Toward Safe Set from Unsafe States (N=52)")
    plt.xlabel("Elapsed Time (s)")
    plt.ylabel("CBF Safety Function h(t) (m)")
    plt.legend(ncol=3, loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    fig3_path = FIGURES_DIR / "fig3_recovery_analysis.png"
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # ------------------------------------------------------------
    # FIGURE 4: TRAJECTORY 6 FORENSIC ANALYSIS
    # ------------------------------------------------------------
    print("Generating Figure 4: Trajectory 6 Forensics...")
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    t6_b = base_ts[base_ts["Trajectory_ID"] == 6].sort_values("Time")
    t6_c = cbf_ts[cbf_ts["Trajectory_ID"] == 6].sort_values("Time")

    axes[0].plot(t6_c["Time"], t6_c["Ego_speed"], "b-", label="Ego Speed (35.8 m/s initial)")
    axes[0].plot(t6_c["Time"], t6_c["Lead_speed"], "k--", label="Lead Speed (27.1 m/s initial)")
    axes[0].set_title("Trajectory 6: Extreme Initial Speed Differential (Closing at 8.68 m/s)")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(t6_b["Time"], t6_b["Gap"], "r--", label="Baseline ACC Gap")
    axes[1].plot(t6_c["Time"], t6_c["Gap"], "b-", label="ACC + CBF Gap")
    axes[1].axhline(0, color="black", linestyle=":", linewidth=1.5, label="Collision Boundary (s = 0)")
    axes[1].set_title("Initial Longitudinal Gap: 0.032 m (Immediate Inevitable Collision at t=0.1s)")
    axes[1].set_ylabel("Gap (m)")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(t6_c["Time"], t6_c["u_cbf"], "m:", label="Theoretical CBF Command (~ -8.8 m/s²)")
    axes[2].plot(t6_c["Time"], t6_c["Acceleration"], "b-", label="Applied Acceleration (Saturated at -4 m/s²)")
    axes[2].axhline(A_MIN, color="gray", linestyle="--", label="Physical Actuator Saturation Limit")
    axes[2].set_title("Actuator Infeasibility: Physically Impossible to Halt 8.7 m/s Closing in 3.2 cm")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Acceleration (m/s²)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    fig4_path = FIGURES_DIR / "fig4_trajectory6_forensics.png"
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")

    # ------------------------------------------------------------
    # FIGURE 5: METRICS DISTRIBUTION (Baseline vs CBF)
    # ------------------------------------------------------------
    print("Generating Figure 5: Metrics Distributions...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    clean_metrics = comp_metrics[comp_metrics["Trajectory_ID"] != 6]

    axes[0].hist(clean_metrics["Minimum_gap_Baseline"], bins=15, alpha=0.6, color="red", label="Baseline ACC")
    axes[0].hist(clean_metrics["Minimum_gap_CBF"], bins=15, alpha=0.6, color="blue", label="ACC + CBF")
    axes[0].set_title("Minimum Spacing Distribution (Valid N=60)")
    axes[0].set_xlabel("Minimum Gap (m)")
    axes[0].set_ylabel("Trajectory Count")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].hist(clean_metrics["Mean_gap_Baseline"], bins=15, alpha=0.6, color="red", label="Baseline ACC")
    axes[1].hist(clean_metrics["Mean_gap_CBF"], bins=15, alpha=0.6, color="blue", label="ACC + CBF")
    axes[1].set_title("Mean Spacing Distribution")
    axes[1].set_xlabel("Mean Gap (m)")
    axes[1].set_ylabel("Trajectory Count")
    axes[1].legend()
    axes[1].grid(True)

    axes[2].hist(comp_metrics["Maximum_braking_Baseline"], bins=10, alpha=0.6, color="red", label="Baseline ACC")
    axes[2].hist(comp_metrics["Maximum_braking_CBF"], bins=10, alpha=0.6, color="blue", label="ACC + CBF")
    axes[2].set_title("Peak Deceleration Distribution")
    axes[2].set_xlabel("Max Braking (m/s²)")
    axes[2].set_ylabel("Trajectory Count")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    fig5_path = FIGURES_DIR / "fig5_metrics_distribution.png"
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved: {fig5_path}")

    # ------------------------------------------------------------
    # FIGURE 6: THE ROOT CAUSE OF 94.5% INTERVENTION
    # ------------------------------------------------------------
    print("Generating Figure 6: Safety vs Nominal Trade-off...")
    plt.figure(figsize=(10, 6))

    v_grid = np.linspace(10, 40, 300)
    gap_cbf = T_MIN * v_grid + D_MIN
    gap_acc = T_DES * v_grid + D_DES

    raw_sub = raw_df.sample(min(10000, len(raw_df)), random_state=42)
    plt.scatter(raw_sub["Speed_FAV"], raw_sub["Spatial_Gap"], c="lightgray", s=3, alpha=0.4, label="Vicolungo Human Driving Samples")

    plt.plot(v_grid, gap_cbf, "b-", linewidth=2.5, label=f"Paper CBF Safe Boundary: s = {T_MIN}v + {D_MIN}")
    plt.plot(v_grid, gap_acc, "r--", linewidth=2.5, label=f"Nominal ACC Equilibrium: s = {T_DES}v + {D_DES}")

    plt.fill_between(v_grid, 0, gap_cbf, color="red", alpha=0.12, label="CBF Unsafe Zone (h < 0) -> CBF Forces Braking")
    plt.fill_between(v_grid, gap_cbf, 130, color="blue", alpha=0.08, label="CBF Safe Zone (h >= 0)")

    plt.title("Root Cause of 94.54% Intervention: Conflict Between ACC Target & Paper CBF Boundary")
    plt.xlabel("Vehicle Speed (m/s)")
    plt.ylabel("Longitudinal Spacing / Gap (m)")
    plt.xlim(10, 40)
    plt.ylim(0, 130)
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    fig6_path = FIGURES_DIR / "fig6_safety_vs_nominal_tradeoff.png"
    plt.savefig(fig6_path, dpi=300)
    plt.close()
    print(f"Saved: {fig6_path}")

    # ------------------------------------------------------------
    # FIGURE 7: PAIRED TRAJECTORY DELTA SCATTER
    # ------------------------------------------------------------
    print("Generating Figure 7: Paired Trajectory Safety vs Speed Deltas...")
    if PAIRED_CLASSIFICATION_CSV.exists():
        paired_df = pd.read_csv(PAIRED_CLASSIFICATION_CSV)
        # Exclude Trajectory 6 for clean visual scaling
        clean_paired = paired_df[paired_df["Trajectory_ID"] != 6]

        plt.figure(figsize=(10, 6))
        category_colors = {
            "Safety Improvement (Headway Expanded)": "blue",
            "No Meaningful Minimum Gap Change": "gray",
            "Safety Improvement with Speed Penalty": "orange",
            "Actuator Saturation with Safe Recovery": "purple",
        }

        for cat, color in category_colors.items():
            subset = clean_paired[clean_paired["Paired_Classification"] == cat]
            plt.scatter(
                subset["delta_v_mean"],
                subset["delta_s_min"],
                c=color,
                s=50,
                alpha=0.8,
                label=f"{cat} (N={len(subset)})"
            )

        plt.axhline(0, color="black", linestyle="--", linewidth=1.0)
        plt.axvline(0, color="black", linestyle="--", linewidth=1.0)
        plt.title("Paired Trajectory Analysis: Safety Margin Gain vs Speed Impact (N=60 Valid)")
        plt.xlabel("Change in Mean Speed: Δv_mean = v_CBF - v_ACC (m/s)")
        plt.ylabel("Change in Minimum Gap: Δs_min = s_min_CBF - s_min_ACC (m)")
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        fig7_path = FIGURES_DIR / "fig7_paired_trajectory_deltas.png"
        plt.savefig(fig7_path, dpi=300)
        plt.close()
        print(f"Saved: {fig7_path}")

    # ------------------------------------------------------------
    # FIGURE 8: T_MIN SENSITIVITY CURVES
    # ------------------------------------------------------------
    print("Generating Figure 8: Parametric Sensitivity Curves...")
    if SENSITIVITY_CSV.exists():
        sens_df = pd.read_csv(SENSITIVITY_CSV)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Intervention Rate vs T_min
        axes[0].plot(sens_df["T_min_s"], sens_df["Intervention_Rate_Pct"], "bo-", linewidth=2.5, markersize=8)
        axes[0].axvline(1.5, color="red", linestyle="--", label="Nominal ACC Headway (T_des = 1.5s)")
        axes[0].set_title("CBF Intervention Rate vs Headway Parameter T_min")
        axes[0].set_xlabel("Minimum Time Gap Parameter T_min (s)")
        axes[0].set_ylabel("CBF Intervention Rate (%)")
        axes[0].legend()
        axes[0].grid(True)

        # Panel 2: Mean Gap & Speed Error vs T_min
        color_gap = "tab:blue"
        axes[1].set_xlabel("Minimum Time Gap Parameter T_min (s)")
        axes[1].set_ylabel("Mean Following Gap (m)", color=color_gap)
        axes[1].plot(sens_df["T_min_s"], sens_df["Mean_Gap_m"], "s-", color=color_gap, linewidth=2.2, label="Mean Gap")
        axes[1].tick_params(axis="y", labelcolor=color_gap)
        axes[1].grid(True)

        ax2 = axes[1].twinx()
        color_err = "tab:red"
        ax2.set_ylabel("Speed Tracking Error (m/s)", color=color_err)
        ax2.plot(sens_df["T_min_s"], sens_df["Speed_Tracking_Error_ms"], "^--", color=color_err, linewidth=2.2, label="Speed Error")
        ax2.tick_params(axis="y", labelcolor=color_err)
        axes[1].set_title("Traffic Flow Trade-off: Headway Spacing vs Speed Error")

        plt.tight_layout()
        fig8_path = FIGURES_DIR / "fig8_tmin_sensitivity.png"
        plt.savefig(fig8_path, dpi=300)
        plt.close()
        print(f"Saved: {fig8_path}")

    print("All publication figures successfully created in figures/ directory.")

if __name__ == "__main__":
    generate_all_figures()
