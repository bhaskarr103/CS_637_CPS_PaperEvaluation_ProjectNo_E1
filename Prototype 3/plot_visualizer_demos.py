"""Generates demonstration figures and plots from the visualizer dataset.

Saves figures to figures/demo/
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).parent
DATA_FILE = ROOT_DIR / "visualizer_data.json"
OUT_DIR = ROOT_DIR / "figures" / "demo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "lines.linewidth": 1.8,
})


def plot_emergency_braking_comparison(data: dict):
    """Visualizes Scenario 3 (Emergency Braking): Filter OFF Comparison."""
    sc = data["scenarios"]["emergency_braking"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=200, sharex=True)

    colors = {
        "Nominal": "#7f7f7f",
        "Filter_Only": "#1f77b4",
        "Dual_CBF_RL": "#2ca02c",
    }
    labels = {
        "Nominal": "Nominal RL (Crashes s=0m)",
        "Filter_Only": "Filter Only (Crashes s=0m)",
        "Dual_CBF_RL": "Dual CBF-RL (Safe s_min=24.5m)",
    }

    # Plot 1: Headway Gap
    for var in ["Nominal", "Filter_Only", "Dual_CBF_RL"]:
        telem = sc["runs"][f"{var}__filter_off"]["telemetry"]
        ax1.plot(telem["t"], telem["s"], label=labels[var], color=colors[var], lw=2.0)

    ax1.axhline(0.0, color="red", linestyle=":", lw=1.5, label="Collision Threshold (s=0)")
    ax1.axhline(15.0, color="black", linestyle="--", lw=1.0, label="D_min (15m)")
    ax1.set_title("[SYNTHETIC DEMO] Scenario 3: Emergency Hard Braking (Deployed with Filter OFF)", fontweight="bold")
    ax1.set_ylabel("Headway Gap s(t) [m]")
    ax1.legend(loc="upper right", framealpha=0.9)

    # Plot 2: Safety Margin h(t)
    for var in ["Nominal", "Filter_Only", "Dual_CBF_RL"]:
        telem = sc["runs"][f"{var}__filter_off"]["telemetry"]
        ax2.plot(telem["t"], telem["h"], label=var.replace("_", " "), color=colors[var], lw=2.0)

    ax2.axhline(0.0, color="black", linestyle="--", lw=1.2, label="Safety Boundary h=0")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Safety Margin h(t) [m]")
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig_path = OUT_DIR / "demo_scenario3_emergency_braking.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path.name}")


def plot_cutin_recovery(data: dict):
    """Visualizes Scenario 4 (Close Cut-In): Safe Recovery & Actuator Saturation."""
    sc = data["scenarios"]["close_cutin"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=200, sharex=True)

    telem = sc["runs"]["Dual_CBF_RL__filter_on"]["telemetry"]

    # Plot 1: Margin h(t) and Gap s(t)
    ax1.plot(telem["t"], telem["h"], color="#10b981", lw=2.2, label="Safety Margin h(t)")
    ax1.plot(telem["t"], telem["s"], color="#3b82f6", lw=1.8, linestyle="--", label="Headway Gap s(t)")
    ax1.axhline(0.0, color="black", linestyle="--", lw=1.2, label="Safety Boundary (h=0)")
    ax1.set_title("[SYNTHETIC DEMO] Scenario 4: Close Cut-In (h0 = -33.0m) Safe Recovery", fontweight="bold")
    ax1.set_ylabel("Margin / Gap (m)")
    ax1.legend(loc="lower right")

    # Plot 2: Accelerations & Actuator Clamping
    ax2.plot(telem["t"], telem["u_applied"], color="#10b981", lw=2.0, label="u_applied (Max Braking)")
    ax2.plot(telem["t"], telem["u_cbf"], color="#f59e0b", lw=1.5, linestyle=":", label="u_cbf_max (Infeasible < -4.0)")
    ax2.axhline(-4.0, color="red", linestyle="--", lw=1.2, label="Physical Limit u_min (-4.0 m/s²)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Acceleration (m/s²)")
    ax2.set_ylim(-6.0, 3.0)
    ax2.legend(loc="lower right")

    plt.tight_layout()
    fig_path = OUT_DIR / "demo_scenario4_cutin_recovery.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path.name}")


def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    plot_emergency_braking_comparison(data)
    plot_cutin_recovery(data)
    print("\nVisualizer demo plots generated in:", OUT_DIR)


if __name__ == "__main__":
    main()
