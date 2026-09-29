"""Phase 2 Diagnostic Plots Generator for Prototype 3 (CBF-RL).

Generates the 8 basic diagnostic plots requested to assess PPO pipeline behavior:
1. Episode return vs training steps (4 variants)
2. Minimum h vs training steps (4 variants with h=0 boundary)
3. Safety rate (% h >= 0) vs training steps
4. CBF intervention rate vs training steps
5. Mean CBF correction magnitude vs training steps
6. Policy action distribution / acceleration telemetry vs training
7. Vicolungo evaluation: min gap and min h for 5 trajectories (Filter ON vs OFF)
8. Representative trajectory time series: u_policy vs u_safe (Filter Only vs Dual)

Saves all plots to:
- E:\\CPS Project\\Prototype 3\\figures\\diagnostic_plots\\
- Copies to conversation artifact directory for UI display
"""

import sys
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure Prototype 3 directory is in Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from env import CarFollowingEnv
from ppo import PPOAgent
from pilot_train import train_variant


def set_plot_style():
    """Sets clean, legible style for diagnostic plots."""
    plt.rcParams.update({
        "figure.autolayout": True,
        "figure.titlesize": 13,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
    })


# Color scheme for the 4 variants
COLORS = {
    "Nominal": "#4a5568",      # Slate Gray
    "Filter_Only": "#e53e3e",  # Red / Coral
    "Reward_Only": "#3182ce",  # Blue
    "Dual_CBF_RL": "#2b6cb0"   # Dark Green / Teal
}
COLORS["Dual_CBF_RL"] = "#2f855a"  # Forest Green


def plot_1_episode_return(df_train: pd.DataFrame, out_path: Path):
    """Plot 1: Episode return vs training steps (4 variants)."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    for v, col in COLORS.items():
        sub = df_train[df_train["variant"] == v].sort_values("global_step")
        if len(sub) == 0:
            continue
        # Raw points + rolling smoothed curve
        ax.plot(sub["global_step"], sub["ep_return"], color=col, alpha=0.25, linewidth=1.0)
        roll = sub["ep_return"].rolling(window=5, min_periods=1).mean()
        ax.plot(sub["global_step"], roll, label=v.replace("_", " "), color=col, linewidth=2.0)

    ax.set_title("Diagnostic 1: Episode Return vs. Training Steps")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Return")
    ax.grid(True)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_2_min_h(df_train: pd.DataFrame, out_path: Path):
    """Plot 2: Minimum h vs training steps (4 variants with h=0 boundary)."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    for v, col in COLORS.items():
        sub = df_train[df_train["variant"] == v].sort_values("global_step")
        if len(sub) == 0:
            continue
        ax.plot(sub["global_step"], sub["min_h"], color=col, alpha=0.2, linewidth=1.0)
        roll = sub["min_h"].rolling(window=5, min_periods=1).mean()
        ax.plot(sub["global_step"], roll, label=v.replace("_", " "), color=col, linewidth=2.0)

    # Reference safety boundary at h = 0
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="Safety Boundary (h=0)")
    ax.set_title("Diagnostic 2: Minimum Safety Margin (min h) vs. Training Steps")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Min Safety Margin h (m)")
    ax.grid(True)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_3_safety_rate(df_train: pd.DataFrame, out_path: Path):
    """Plot 3: Safety rate (% h >= 0) vs training steps."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    for v, col in COLORS.items():
        sub = df_train[df_train["variant"] == v].sort_values("global_step")
        if len(sub) == 0:
            continue
        ax.plot(sub["global_step"], sub["pct_h_safe"], color=col, alpha=0.2, linewidth=1.0)
        roll = sub["pct_h_safe"].rolling(window=5, min_periods=1).mean()
        ax.plot(sub["global_step"], roll, label=v.replace("_", " "), color=col, linewidth=2.0)

    ax.set_title("Diagnostic 3: Percentage of Safe Steps (% h >= 0) vs. Training Steps")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Safe Steps Percentage (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_4_intervention_rate(df_train: pd.DataFrame, out_path: Path):
    """Plot 4: CBF intervention rate vs training steps."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    for v, col in COLORS.items():
        sub = df_train[df_train["variant"] == v].sort_values("global_step")
        if len(sub) == 0:
            continue
        ax.plot(sub["global_step"], sub["intervention_rate"], color=col, alpha=0.2, linewidth=1.0)
        roll = sub["intervention_rate"].rolling(window=5, min_periods=1).mean()
        ax.plot(sub["global_step"], roll, label=v.replace("_", " "), color=col, linewidth=2.0)

    ax.set_title("Diagnostic 4: CBF Intervention Rate (%) vs. Training Steps")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Intervention Rate (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_5_correction_magnitude(df_train: pd.DataFrame, out_path: Path):
    """Plot 5: Mean CBF correction magnitude vs training steps."""
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    for v, col in COLORS.items():
        sub = df_train[df_train["variant"] == v].sort_values("global_step")
        if len(sub) == 0:
            continue
        ax.plot(sub["global_step"], sub["mean_correction"], color=col, alpha=0.2, linewidth=1.0)
        roll = sub["mean_correction"].rolling(window=5, min_periods=1).mean()
        ax.plot(sub["global_step"], roll, label=v.replace("_", " "), color=col, linewidth=2.0)

    ax.set_title("Diagnostic 5: Mean CBF Correction Magnitude |u_pol - u_safe| vs. Training Steps")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Correction Magnitude (m/s²)")
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_6_action_telemetry(df_train: pd.DataFrame, out_path: Path):
    """Plot 6: Policy action telemetry: mean u_policy vs u_safe across training."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=150, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, (v, col) in enumerate(COLORS.items()):
        ax = axes_flat[idx]
        sub = df_train[df_train["variant"] == v].sort_values("global_step")
        if len(sub) == 0:
            continue

        roll_pol = sub["mean_u_policy"].rolling(window=5, min_periods=1).mean()
        roll_safe = sub["mean_u_safe"].rolling(window=5, min_periods=1).mean()

        ax.plot(sub["global_step"], roll_pol, label="Mean u_policy", color="#e53e3e", linewidth=1.8)
        ax.plot(sub["global_step"], roll_safe, label="Mean u_safe", color="#2b6cb0", linestyle="--", linewidth=1.8)

        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(f"{v.replace('_', ' ')}")
        ax.set_ylabel("Acceleration (m/s²)")
        ax.grid(True)
        ax.legend(loc="lower right")

    axes_flat[2].set_xlabel("Environment Steps")
    axes_flat[3].set_xlabel("Environment Steps")
    fig.suptitle("Diagnostic 6: Mean Policy Action vs. Safe Action During Training", fontsize=12)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def plot_7_vicolungo_eval(df_eval: pd.DataFrame, out_path: Path):
    """Plot 7: Vicolungo evaluation: min gap and min h for 5 trajectories (Filter ON vs OFF)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Group by variant and deploy_mode
    trajs = sorted(df_eval["traj_id"].unique())
    x = np.arange(len(trajs))
    width = 0.18

    # Min Headway Gap Plot
    modes = [
        ("Nominal", "Filter_OFF", "#718096", "-"),
        ("Filter_Only", "Filter_OFF", "#fc8181", "--"),
        ("Filter_Only", "Filter_ON", "#e53e3e", "-"),
        ("Dual_CBF_RL", "Filter_OFF", "#68d391", "--"),
        ("Dual_CBF_RL", "Filter_ON", "#2f855a", "-"),
    ]

    for idx, (var, dep, col, ls) in enumerate(modes):
        sub = df_eval[(df_eval["variant"] == var) & (df_eval["deploy_mode"] == dep)].sort_values("traj_id")
        if len(sub) == 0:
            continue
        label = f"{var.replace('_', ' ')} ({dep.replace('_', ' ')})"
        ax1.plot(sub["traj_id"], sub["min_gap"], marker="o", color=col, linestyle=ls, label=label, linewidth=1.8)
        ax2.plot(sub["traj_id"], sub["min_h"], marker="s", color=col, linestyle=ls, label=label, linewidth=1.8)

    ax1.set_title("Vicolungo: Minimum Gap (m) per Trajectory")
    ax1.set_xlabel("Trajectory ID")
    ax1.set_ylabel("Minimum Gap (m)")
    ax1.set_xticks(trajs)
    ax1.grid(True)
    ax1.legend(fontsize=8, loc="upper right")

    ax2.set_title("Vicolungo: Minimum Safety Margin (min h) per Trajectory")
    ax2.set_xlabel("Trajectory ID")
    ax2.set_ylabel("Min h (m)")
    ax2.set_xticks(trajs)
    ax2.axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="Boundary (h=0)")
    ax2.grid(True)
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle("Diagnostic 7: Closed-Loop Performance on 5 Vicolungo Trajectories", fontsize=12)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def generate_plot_8_representative_timeseries(cfg: Config, out_path: Path):
    """Plot 8: Step-by-step u_policy vs u_safe on representative Vicolungo Trajectory 1 for Filter Only & Dual."""
    print("Generating representative Trajectory 1 time series for Filter Only & Dual...")
    # Train lightweight instances of Filter Only and Dual
    agent_filt, _ = train_variant("Filter_Only", use_filter=True, use_reward=False, total_steps=15000, seed=42, config=cfg)
    agent_dual, _ = train_variant("Dual_CBF_RL", use_filter=True, use_reward=True, total_steps=15000, seed=42, config=cfg)

    # Rollout on Trajectory 1
    def rollout_traj1(agent, use_filter):
        env = CarFollowingEnv(config=cfg, mode="vicolungo", use_cbf_filter=use_filter, use_cbf_reward=False)
        obs, info = env.reset(seed=1, options={"traj_id": 1})
        records = []
        done = False
        step = 0
        while not done and step < 200:
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, _, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            records.append({
                "time": step * cfg.dt,
                "u_policy": step_info["u_policy"],
                "u_safe": step_info["u_safe"],
                "u_applied": step_info["u_applied"],
                "u_cbf_max": step_info["u_cbf_max"],
                "h": step_info["h"],
                "s": step_info["s"],
                "v_ego": step_info["v_ego"],
                "v_lead": step_info["v_lead"],
            })
            step += 1
        return pd.DataFrame(records)

    df_filt = rollout_traj1(agent_filt, use_filter=True)
    df_dual = rollout_traj1(agent_dual, use_filter=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), dpi=150, sharex=True)

    # Subplot 1: Filter Only
    ax1.plot(df_filt["time"], df_filt["u_policy"], label="Policy Proposed u_policy", color="#e53e3e", linewidth=1.8)
    ax1.plot(df_filt["time"], df_filt["u_safe"], label="CBF Filtered u_safe", color="#2b6cb0", linestyle="--", linewidth=1.8)
    ax1.plot(df_filt["time"], df_filt["u_cbf_max"], label="CBF Upper Bound u_cbf_max", color="black", linestyle=":", linewidth=1.2)
    ax1.set_title("Filter Only (No CBF Reward) on Vicolungo Trajectory 1")
    ax1.set_ylabel("Acceleration (m/s²)")
    ax1.grid(True)
    ax1.legend(loc="lower right")

    # Subplot 2: Dual CBF-RL
    ax2.plot(df_dual["time"], df_dual["u_policy"], label="Policy Proposed u_policy", color="#e53e3e", linewidth=1.8)
    ax2.plot(df_dual["time"], df_dual["u_safe"], label="CBF Filtered u_safe", color="#2f855a", linestyle="--", linewidth=1.8)
    ax2.plot(df_dual["time"], df_dual["u_cbf_max"], label="CBF Upper Bound u_cbf_max", color="black", linestyle=":", linewidth=1.2)
    ax2.set_title("Dual CBF-RL (Active Filter + CBF Reward) on Vicolungo Trajectory 1")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Acceleration (m/s²)")
    ax2.grid(True)
    ax2.legend(loc="lower right")

    fig.suptitle("Diagnostic 8: Representative Acceleration Profile u_policy vs. u_safe", fontsize=12)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path.name}")


def main():
    set_plot_style()
    cfg = Config()

    out_dir = cfg.figures_dir / "diagnostic_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path(r"C:\Users\rajau\.gemini\antigravity\brain\6809cadf-3f47-4ce9-9f71-18861ed7d483\diagnostic_plots")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Load existing logs
    train_log = cfg.results_dir / "pilot_training_log.csv"
    eval_log = cfg.results_dir / "pilot_vicolungo_eval.csv"

    if not train_log.exists() or not eval_log.exists():
        print(f"Error: Missing log files in {cfg.results_dir}")
        return

    df_train = pd.read_csv(train_log)
    df_eval = pd.read_csv(eval_log)

    print(f"Loaded training records: {len(df_train)}")
    print(f"Loaded evaluation records: {len(df_eval)}")

    # Generate plots 1-7 from logs
    p1 = out_dir / "diag1_episode_return.png"
    plot_1_episode_return(df_train, p1)

    p2 = out_dir / "diag2_min_h.png"
    plot_2_min_h(df_train, p2)

    p3 = out_dir / "diag3_safety_rate.png"
    plot_3_safety_rate(df_train, p3)

    p4 = out_dir / "diag4_intervention_rate.png"
    plot_4_intervention_rate(df_train, p4)

    p5 = out_dir / "diag5_correction_magnitude.png"
    plot_5_correction_magnitude(df_train, p5)

    p6 = out_dir / "diag6_action_telemetry.png"
    plot_6_action_telemetry(df_train, p6)

    p7 = out_dir / "diag7_vicolungo_eval.png"
    plot_7_vicolungo_eval(df_eval, p7)

    # Generate plot 8 from fresh trajectory rollout
    p8 = out_dir / "diag8_trajectory_timeseries.png"
    generate_plot_8_representative_timeseries(cfg, p8)

    # Copy all to artifact directory for UI display
    for p in [p1, p2, p3, p4, p5, p6, p7, p8]:
        shutil.copy(p, artifact_dir / p.name)

    print(f"\nAll 8 diagnostic plots generated and copied to artifact directory:\n{artifact_dir}")


if __name__ == "__main__":
    main()
