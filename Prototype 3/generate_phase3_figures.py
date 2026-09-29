"""Phase 3 Publication-Quality Figure Generation.

Generates 9 figures strictly categorized into:
- Category A: [TRAINING — SYNTHETIC] (Figs 1, 2, 3, 4)
- Category B: [VICOLUNGO — REAL-DATA LEAD REPLAY] (Figs 5, 6, 7, 8)
- Category C: [SYNTHETIC DEMONSTRATION] (Fig 9)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import Config

# Styling configuration for publication-grade figures

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

# Distinct color palette for 4 ablations
COLORS = {
    "Nominal": "#7f7f7f",       # Neutral gray
    "Filter_Only": "#1f77b4",   # Deep blue
    "Reward_Only": "#ff7f0e",   # Muted orange
    "Dual_CBF_RL": "#2ca02c",   # Forest green
}

LABELS = {
    "Nominal": "Nominal RL (Unshielded)",
    "Filter_Only": "Filter Only (Shielded, No CBF Rew)",
    "Reward_Only": "Reward Only (Unshielded, CBF Rew)",
    "Dual_CBF_RL": "Dual CBF-RL (Shielded + CBF Rew)",
}


def smooth_series(vals: np.ndarray, window: int = 15) -> np.ndarray:
    """Moving average smoothing for noisy telemetry."""
    if len(vals) < window:
        return vals
    weights = np.ones(window) / window
    return np.convolve(vals, weights, mode="valid")


def plot_fig1_training_curves(df_train: pd.DataFrame, out_dir: Path):
    """Fig 1: [TRAINING — SYNTHETIC] Episode Return vs Training Steps."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    for variant in ["Nominal", "Filter_Only", "Reward_Only", "Dual_CBF_RL"]:
        v_df = df_train[df_train["variant"] == variant]
        # Aggregate across seeds by binning steps
        bins = np.linspace(0, 150000, 150)
        v_df = v_df.copy()
        v_df["step_bin"] = pd.cut(v_df["global_step"], bins=bins)
        grouped = v_df.groupby("step_bin", observed=False)["ep_return"].agg(["mean", "std", "count"])
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        means = grouped["mean"].to_numpy()
        stds = grouped["std"].fillna(0).to_numpy()
        valid = ~np.isnan(means)

        color = COLORS[variant]
        label = LABELS[variant]
        ax.plot(bin_centers[valid], means[valid], label=label, color=color, lw=2.0)
        ax.fill_between(
            bin_centers[valid],
            means[valid] - stds[valid],
            means[valid] + stds[valid],
            color=color,
            alpha=0.15,
        )

    ax.set_title("[TRAINING — SYNTHETIC] Episode Return vs. Environment Steps (Mean ± 1 SD across 3 seeds)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Return")
    ax.set_xlim(0, 150000)
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()
    fig_path = out_dir / "phase3_fig1_training_curves.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path.name}")


def plot_fig2_safety_training(df_train: pd.DataFrame, out_dir: Path):
    """Fig 2: [TRAINING — SYNTHETIC] Safety Margin & Rate during Training."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    bins = np.linspace(0, 150000, 150)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for variant in ["Nominal", "Filter_Only", "Reward_Only", "Dual_CBF_RL"]:
        v_df = df_train[df_train["variant"] == variant].copy()
        v_df["step_bin"] = pd.cut(v_df["global_step"], bins=bins)
        color = COLORS[variant]
        label = LABELS[variant]

        # Left: Minimum h
        grp_h = v_df.groupby("step_bin", observed=False)["min_h"].mean().to_numpy()
        valid_h = ~np.isnan(grp_h)
        ax1.plot(bin_centers[valid_h], grp_h[valid_h], label=label, color=color, lw=1.8)

        # Right: Pct Safe Steps (% h >= 0)
        grp_pct = v_df.groupby("step_bin", observed=False)["pct_h_safe"].mean().to_numpy()
        valid_pct = ~np.isnan(grp_pct)
        ax2.plot(bin_centers[valid_pct], grp_pct[valid_pct], label=label, color=color, lw=1.8)

    ax1.axhline(0.0, color="black", linestyle=":", lw=1.2, label="Safety Boundary (h=0)")
    ax1.set_title("[TRAINING — SYNTHETIC] Minimum Safety Margin (min h)")
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Safety Margin h (m)")
    ax1.set_xlim(0, 150000)
    ax1.legend(loc="lower right")

    ax2.axhline(100.0, color="black", linestyle=":", lw=1.0)
    ax2.set_title("[TRAINING — SYNTHETIC] Step Safety Rate (% h ≥ 0)")
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Safe Steps (%)")
    ax2.set_xlim(0, 150000)
    ax2.set_ylim(-5, 105)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    fig_path = out_dir / "phase3_fig2_safety_training.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path.name}")


def plot_fig3_cbf_interventions(df_train: pd.DataFrame, out_dir: Path):
    """Fig 3: [TRAINING — SYNTHETIC] Intervention Rate and Correction Magnitude."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    bins = np.linspace(0, 150000, 150)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    for variant in ["Filter_Only", "Dual_CBF_RL"]:
        v_df = df_train[df_train["variant"] == variant].copy()
        v_df["step_bin"] = pd.cut(v_df["global_step"], bins=bins)
        color = COLORS[variant]
        label = LABELS[variant]

        # Intervention rate
        grp_rate = v_df.groupby("step_bin", observed=False)["intervention_rate"].mean().to_numpy()
        valid = ~np.isnan(grp_rate)
        ax1.plot(bin_centers[valid], grp_rate[valid], label=label, color=color, lw=2.0)

        # Correction magnitude
        grp_corr = v_df.groupby("step_bin", observed=False)["mean_correction"].mean().to_numpy()
        valid_c = ~np.isnan(grp_corr)
        ax2.plot(bin_centers[valid_c], grp_corr[valid_c], label=label, color=color, lw=2.0)

    ax1.set_title("[TRAINING — SYNTHETIC] CBF Intervention Rate During Training")
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Intervention Rate (%)")
    ax1.set_xlim(0, 150000)
    ax1.legend(loc="upper right")

    ax2.set_title("[TRAINING — SYNTHETIC] Mean Action Correction ||u_policy - u_safe||")
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Correction Magnitude (m/s²)")
    ax2.set_xlim(0, 150000)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig_path = out_dir / "phase3_fig3_cbf_interventions.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path.name}")


def plot_fig4_action_alignment(df_train: pd.DataFrame, out_dir: Path):
    """Fig 4: [TRAINING — SYNTHETIC] Policy Action Internalization & Divergence."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    bins = np.linspace(0, 150000, 150)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Left: Filter Only divergence
    fo_df = df_train[df_train["variant"] == "Filter_Only"].copy()
    fo_df["step_bin"] = pd.cut(fo_df["global_step"], bins=bins)
    u_pol_fo = fo_df.groupby("step_bin", observed=False)["mean_u_policy"].mean().to_numpy()
    u_safe_fo = fo_df.groupby("step_bin", observed=False)["mean_u_safe"].mean().to_numpy()
    valid_fo = ~np.isnan(u_pol_fo)

    ax1.plot(bin_centers[valid_fo], u_pol_fo[valid_fo], color="#d62728", lw=1.8, label="u_policy (Proposed)")
    ax1.plot(bin_centers[valid_fo], u_safe_fo[valid_fo], color="#2ca02c", lw=1.8, linestyle="--", label="u_safe (Applied)")
    ax1.set_title("[TRAINING — SYNTHETIC] Filter Only: Action Divergence")
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Acceleration (m/s²)")
    ax1.set_xlim(0, 150000)
    ax1.legend(loc="center right")

    # Right: Dual CBF-RL alignment
    dual_df = df_train[df_train["variant"] == "Dual_CBF_RL"].copy()
    dual_df["step_bin"] = pd.cut(dual_df["global_step"], bins=bins)
    u_pol_dual = dual_df.groupby("step_bin", observed=False)["mean_u_policy"].mean().to_numpy()
    u_safe_dual = dual_df.groupby("step_bin", observed=False)["mean_u_safe"].mean().to_numpy()
    valid_dual = ~np.isnan(u_pol_dual)

    ax2.plot(bin_centers[valid_dual], u_pol_dual[valid_dual], color="#d62728", lw=1.8, label="u_policy (Proposed)")
    ax2.plot(bin_centers[valid_dual], u_safe_dual[valid_dual], color="#2ca02c", lw=1.8, linestyle="--", label="u_safe (Applied)")
    ax2.set_title("[TRAINING — SYNTHETIC] Dual CBF-RL: Action Alignment")
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Acceleration (m/s²)")
    ax2.set_xlim(0, 150000)
    ax2.legend(loc="center right")

    plt.tight_layout()
    fig_path = out_dir / "phase3_fig4_action_alignment.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path.name}")


def plot_fig5_vicolungo_overall(df_eval: pd.DataFrame, out_dir: Path):
    """Fig 5: [VICOLUNGO — REAL-DATA LEAD REPLAY] 60-Trajectory Safety Benchmark."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    variants = ["Nominal", "Filter_Only", "Reward_Only", "Dual_CBF_RL"]
    modes = ["Filter_ON", "Filter_OFF"]

    # Left: Minimum Gap Boxplots
    plot_data = []
    plot_labels = []
    plot_colors = []

    for var in variants:
        for mode in modes:
            sub = df_eval[(df_eval["variant"] == var) & (df_eval["deploy_mode"] == mode)]
            plot_data.append(sub["min_gap"].values)
            m_short = "ON" if mode == "Filter_ON" else "OFF"
            plot_labels.append(f"{var[:6]}\n({m_short})")
            plot_colors.append(COLORS[var])

    bplot = ax1.boxplot(plot_data, patch_artist=True, tick_labels=plot_labels, showmeans=True)
    for patch, c in zip(bplot['boxes'], plot_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax1.axhline(0.0, color="red", linestyle=":", lw=1.5, label="Collision Threshold (s=0)")
    ax1.set_title("[VICOLUNGO — REAL DATA] Minimum Spatial Gap Across 60 Trajectories")
    ax1.set_ylabel("Minimum Gap s_min (m)")
    ax1.legend(loc="lower right")

    # Right: Collision Counts
    collision_counts_on = []
    collision_counts_off = []
    for var in variants:
        c_on = df_eval[(df_eval["variant"] == var) & (df_eval["deploy_mode"] == "Filter_ON")]["collision"].sum()
        c_off = df_eval[(df_eval["variant"] == var) & (df_eval["deploy_mode"] == "Filter_OFF")]["collision"].sum()
        collision_counts_on.append(c_on)
        collision_counts_off.append(c_off)

    x = np.arange(len(variants))
    width = 0.35
    ax2.bar(x - width/2, collision_counts_on, width, label="Filter ON", color="#1f77b4", alpha=0.85)
    ax2.bar(x + width/2, collision_counts_off, width, label="Filter OFF (Internalization)", color="#d62728", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([v.replace("_", " ") for v in variants], rotation=15)
    ax2.set_title("[VICOLUNGO — REAL DATA] Total Collisions (Across 3 seeds × 60 trajs = 180 runs)")
    ax2.set_ylabel("Total Collision Events")
    ax2.legend()

    plt.tight_layout()
    fig_path = out_dir / "phase3_fig5_vicolungo_overall.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path.name}")


def plot_fig6_subset_a_forward_invariance(df_eval: pd.DataFrame, out_dir: Path):
    """Fig 6: [VICOLUNGO — REAL-DATA LEAD REPLAY] Subset A: Forward Invariance (N=9, h0 >= 0)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    sub_a = df_eval[df_eval["is_initially_safe"] == True]
    variants = ["Nominal", "Filter_Only", "Reward_Only", "Dual_CBF_RL"]

    # Filter ON vs Filter OFF safety rate
    for mode, ax, title_suffix in [("Filter_ON", ax1, "(Filter ON)"), ("Filter_OFF", ax2, "(Filter OFF - Internalization)")]:
        sub_mode = sub_a[sub_a["deploy_mode"] == mode]
        rates = []
        errs = []
        for var in variants:
            v_rates = sub_mode[sub_mode["variant"] == var]["pct_h_safe"]
            rates.append(v_rates.mean())
            errs.append(v_rates.std() / np.sqrt(max(1, len(v_rates))))

        x = np.arange(len(variants))
        bars = ax.bar(x, rates, yerr=errs, capsize=4, color=[COLORS[v] for v in variants], alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([v.replace("_", " ") for v in variants], rotation=15)
        ax.set_ylim(0, 105)
        ax.axhline(100.0, color="black", linestyle=":", lw=1.2, label="100% Invariance")
        ax.set_title(f"[SUBSET A: INITIALLY SAFE, N=9] Safe Steps {title_suffix}")
        ax.set_ylabel("% Steps with h ≥ 0")
        ax.legend(loc="lower left")

        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h/2),
                        xytext=(0, 0), textcoords="offset points", ha='center', va='center',
                        color='white', fontweight='bold')

    plt.tight_layout()
    fig_path = out_dir / "phase3_fig6_subset_a_forward_invariance.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path.name}")


def plot_fig7_subset_b_safe_recovery(df_eval: pd.DataFrame, out_dir: Path):
    """Fig 7: [VICOLUNGO — REAL-DATA LEAD REPLAY] Subset B: Safe Recovery (N=51, h0 < 0)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=300)

    sub_b = df_eval[df_eval["is_initially_safe"] == False]
    variants = ["Nominal", "Filter_Only", "Reward_Only", "Dual_CBF_RL"]

    # Filter OFF performance (internalization test on recovery)
    sub_off = sub_b[sub_b["deploy_mode"] == "Filter_OFF"]

    # Minimum gap under Filter OFF
    gaps_mean = [sub_off[sub_off["variant"] == v]["min_gap"].mean() for v in variants]
    gaps_std = [sub_off[sub_off["variant"] == v]["min_gap"].std() for v in variants]

    x = np.arange(len(variants))
    ax1.bar(x, gaps_mean, yerr=gaps_std, capsize=4, color=[COLORS[v] for v in variants], alpha=0.85)
    ax1.axhline(0.0, color="red", linestyle="--", lw=1.5, label="Collision Threshold (s=0)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([v.replace("_", " ") for v in variants], rotation=15)
    ax1.set_title("[SUBSET B: PENETRATED, N=51] Mean Minimum Gap (Filter OFF)")
    ax1.set_ylabel("Minimum Gap s_min (m)")
    ax1.legend(loc="lower right")

    # Recovery Time (seconds to reach h >= 0) under Filter ON
    sub_on = sub_b[sub_b["deploy_mode"] == "Filter_ON"]
    recov_mean = [sub_on[sub_on["variant"] == v]["recovery_time_s"].dropna().mean() for v in variants]
    recov_std = [sub_on[sub_on["variant"] == v]["recovery_time_s"].dropna().std() for v in variants]

    ax2.bar(x, recov_mean, yerr=recov_std, capsize=4, color=[COLORS[v] for v in variants], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([v.replace("_", " ") for v in variants], rotation=15)
    ax2.set_title("[SUBSET B: PENETRATED, N=51] Mean Time to Recover h ≥ 0 (Filter ON)")
    ax2.set_ylabel("Recovery Time (s)")

    plt.tight_layout()
    fig_path = out_dir / "phase3_fig7_subset_b_safe_recovery.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path.name}")


def plot_fig8_representative_trajectories(out_dir: Path):
    """Fig 8: [VICOLUNGO — REAL-DATA LEAD REPLAY] Representative Trajectories (Traj 3 & Traj 1)."""
    # Load checkpoints and run clean single-episode rollouts on Traj 3 and Traj 1
    # to plot high-resolution acceleration and margin timeseries
    from config import Config
    from env import CarFollowingEnv
    from ppo import PPOAgent
    import torch

    cfg = Config()
    ckpt_dir = cfg.results_dir / "checkpoints"

    # We use Seed 42 checkpoints
    agents = {}
    for var in ["Filter_Only", "Dual_CBF_RL"]:
        agent = PPOAgent(obs_dim=4, act_dim=1, device="cpu")
        ckpt_path = ckpt_dir / f"{var}_seed42.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            agent.ac.load_state_dict(ckpt["model_state_dict"])
            agents[var] = agent

    if len(agents) < 2:
        print("Skipping Fig 8: Checkpoints not yet available.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=300)

    # Top row: Trajectory 3 (Initially Safe, Subset A)
    # Bottom row: Trajectory 1 (Initially Unsafe, Subset B)
    trajs = [(3, axes[0], "Trajectory 3 (Initially Safe, h0 = +51.3m)"),
             (1, axes[1], "Trajectory 1 (Initially Penetrated, h0 = -34.1m)")]

    for tid, row_axes, row_title in trajs:
        for col_idx, (var, agent) in enumerate(agents.items()):
            ax = row_axes[col_idx]
            env = CarFollowingEnv(config=cfg, mode="vicolungo", use_cbf_filter=True, use_cbf_reward=False)
            obs, info = env.reset(seed=42, options={"traj_id": tid})

            times = [0.0]
            u_pols = []
            u_safes = []
            h_vals = [info["h"]]

            done = False
            step = 0
            while not done:
                action, _, _ = agent.select_action(obs, deterministic=True)
                obs, r, term, trunc, step_info = env.step(action)
                done = term or trunc
                step += 1
                times.append(step * cfg.dt)
                u_pols.append(step_info["u_policy"])
                u_safes.append(step_info["u_safe"])
                h_vals.append(step_info["h"])

            time_arr = np.array(times[1:])
            ax.plot(time_arr, u_pols, color="#d62728", lw=1.8, label="u_policy (Proposed)")
            ax.plot(time_arr, u_safes, color="#2ca02c", lw=1.8, linestyle="--", label="u_safe (Applied)")
            ax.axhline(cfg.u_min, color="black", linestyle=":", lw=1.0, label="u_min (-4.0)")
            ax.set_title(f"[VICOLUNGO] {row_title} — {LABELS[var][:15]}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Acceleration (m/s²)")
            ax.legend(loc="upper right")

    plt.tight_layout()
    fig_path = out_dir / "phase3_fig8_representative_trajectories.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path.name}")


def plot_fig9_platoon_demo(out_dir: Path):
    """Fig 9: [SYNTHETIC DEMONSTRATION] Multi-Car Synthetic Platoon Demonstration."""
    from simulate_platoon import run_platoon_simulation, plot_platoon_results

    fig_path = out_dir / "phase3_fig9_platoon_demonstration.png"
    df_platoon = run_platoon_simulation(num_followers=4, duration_s=22.0)
    plot_platoon_results(df_platoon, fig_path)
    print(f"Saved: {fig_path.name}")



def generate_all_figures():
    """Generates all 9 Phase 3 figures from saved experimental logs."""
    cfg = Config()
    train_log = cfg.results_dir / "phase3_training_log.csv"
    eval_log = cfg.results_dir / "phase3_vicolungo_eval.csv"

    if not train_log.exists() or not eval_log.exists():
        print(f"Error: Missing training or eval logs in {cfg.results_dir}")
        return

    out_dir = cfg.figures_dir / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Phase 3 experiment logs...")
    df_train = pd.read_csv(train_log)
    df_eval = pd.read_csv(eval_log)

    print("\nGenerating Category A: [TRAINING — SYNTHETIC] Figures...")
    plot_fig1_training_curves(df_train, out_dir)
    plot_fig2_safety_training(df_train, out_dir)
    plot_fig3_cbf_interventions(df_train, out_dir)
    plot_fig4_action_alignment(df_train, out_dir)

    print("\nGenerating Category B: [VICOLUNGO — REAL-DATA LEAD REPLAY] Figures...")
    plot_fig5_vicolungo_overall(df_eval, out_dir)
    plot_fig6_subset_a_forward_invariance(df_eval, out_dir)
    plot_fig7_subset_b_safe_recovery(df_eval, out_dir)
    plot_fig8_representative_trajectories(out_dir)

    print("\nGenerating Category C: [SYNTHETIC DEMONSTRATION] Figures...")
    plot_fig9_platoon_demo(out_dir)

    print("\nAll 9 Phase 3 figures successfully generated in:", out_dir)


if __name__ == "__main__":
    generate_all_figures()
