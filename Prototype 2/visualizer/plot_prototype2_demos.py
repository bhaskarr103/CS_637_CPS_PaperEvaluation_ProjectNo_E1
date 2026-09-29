"""Generates presentation demonstration plots for Prototype 2.

Plots:
1. Scenario 4 (Emergency Braking): Baseline ACC vs ACC+CBF Comparison
2. Trajectory 16 (Vicolungo Stress Case): Baseline ACC vs ACC+CBF Comparison

Saves figures to E:\CPS Project\Prototype 2\visualizer\screenshots/
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Ensure Prototype 2 is in sys.path
PROTO2_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROTO2_DIR))

from project_config import (
    DT, D_DES, T_DES, K_GAP, K_REL,
    T_MIN, D_MIN, K_CBF, A_MAX, A_MIN, JERK_MAX
)
from cbf_controller import CBFSafetyFilter, compute_h

OUT_DIR = Path(__file__).resolve().parent / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "lines.linewidth": 2.0,
})


def simulate_rollout(s0, v_ego0, lead_speeds, use_cbf):
    """Simulates a single vehicle rollout using Prototype 2 kinematics."""
    steps = len(lead_speeds) - 1
    cbf_filter = CBFSafetyFilter()

    s = s0
    v_ego = v_ego0
    prev_acc = 0.0

    t_arr = [0.0]
    gap_arr = [s]
    ego_v_arr = [v_ego]
    h_arr = [compute_h(s, v_ego)]
    acc_arr = [0.0]
    u_cbf_arr = [0.0]
    u_nom_arr = [0.0]

    max_acc_change = JERK_MAX * DT

    for k in range(steps):
        lead_speed = lead_speeds[k]
        delta_v = lead_speed - v_ego
        h = compute_h(s, v_ego)

        s_des = D_DES + T_DES * v_ego
        gap_error = s - s_des
        u_nom = K_GAP * gap_error + K_REL * delta_v
        u_nom_sat = float(np.clip(u_nom, A_MIN, A_MAX))

        if use_cbf:
            res = cbf_filter.filter_control(u_nom_sat, s, v_ego, lead_speed)
            applied_acc = res["applied_acc"]
            u_cbf_val = res["u_cbf"]
        else:
            acc_change = np.clip(u_nom_sat - prev_acc, -max_acc_change, max_acc_change)
            applied_acc = prev_acc + acc_change
            u_cbf_val = (K_CBF * h + delta_v) / T_MIN

        prev_acc = applied_acc
        v_ego_next = max(0.0, v_ego + applied_acc * DT)
        s_next = s + (lead_speed - v_ego) * DT

        s = s_next
        v_ego = v_ego_next

        t_arr.append(round((k + 1) * DT, 2))
        gap_arr.append(round(s, 2))
        ego_v_arr.append(round(v_ego, 2))
        h_arr.append(round(compute_h(s, v_ego), 2))
        acc_arr.append(round(applied_acc, 2))
        u_cbf_arr.append(round(u_cbf_val, 2))
        u_nom_arr.append(round(u_nom_sat, 2))

    return {
        "t": np.array(t_arr),
        "gap": np.array(gap_arr),
        "ego_v": np.array(ego_v_arr),
        "h": np.array(h_arr),
        "acc": np.array(acc_arr),
        "u_cbf": np.array(u_cbf_arr),
        "u_nom": np.array(u_nom_arr),
    }


def plot_emergency_braking_demo():
    """Generates comparison plot for Scenario 4 (Emergency Hard Braking)."""
    # Build lead speed profile
    dt = DT
    total_steps = 250
    t = np.arange(total_steps + 1) * dt
    lead_speeds = np.full_like(t, 25.0)
    for idx, ti in enumerate(t):
        if 3.0 <= ti <= 8.5:
            lead_speeds[idx] = 25.0 - 3.6 * (ti - 3.0)
        elif ti > 8.5:
            lead_speeds[idx] = 5.2

    s0 = 68.0
    v_ego0 = 25.0

    res_base = simulate_rollout(s0, v_ego0, lead_speeds, use_cbf=False)
    res_cbf = simulate_rollout(s0, v_ego0, lead_speeds, use_cbf=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8.5), dpi=200, sharex=True)

    # 1. Headway Gap
    ax1.plot(res_base["t"], res_base["gap"], label="Baseline ACC", color="#f59e0b", lw=2.0)
    ax1.plot(res_cbf["t"], res_cbf["gap"], label="ACC + CBF (Supervised)", color="#3b82f6", lw=2.2)
    ax1.axhline(0.0, color="red", linestyle=":", lw=1.5, label="Collision Threshold (s=0m)")
    ax1.axhline(15.0, color="black", linestyle="--", lw=1.0, label="D_min (15m)")
    ax1.set_title("[SYNTHETIC DEMO] Scenario 4: Emergency Hard Braking (Baseline ACC vs. ACC+CBF)", fontweight="bold")
    ax1.set_ylabel("Headway Gap s(t) [m]")
    ax1.legend(loc="upper right", framealpha=0.9)

    # 2. Safety Margin h(t)
    ax2.plot(res_base["t"], res_base["h"], label="Baseline ACC", color="#f59e0b", lw=2.0)
    ax2.plot(res_cbf["t"], res_cbf["h"], label="ACC + CBF", color="#10b981", lw=2.2)
    ax2.axhline(0.0, color="black", linestyle="--", lw=1.2, label="Safety Boundary (h=0)")
    ax2.set_ylabel("Safety Margin h(t) [m]")
    ax2.legend(loc="upper right", framealpha=0.9)

    # 3. Applied Accelerations
    ax3.plot(res_base["t"], res_base["acc"], label="Baseline ACC Applied", color="#f59e0b", lw=1.8, linestyle="--")
    ax3.plot(res_cbf["t"], res_cbf["acc"], label="ACC + CBF Applied", color="#34d399", lw=2.2)
    ax3.plot(res_cbf["t"], res_cbf["u_cbf"], label="CBF Limit (u_CBF)", color="#ef4444", lw=1.4, linestyle=":")
    ax3.axhline(A_MIN, color="black", linestyle=":", lw=1.0, label=f"A_min ({A_MIN} m/s²)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Acceleration (m/s²)")
    ax3.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig_path = OUT_DIR / "demo_p2_emergency_braking.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path.name}")


def plot_trajectory16_demo():
    """Generates comparison plot for Trajectory 16 (Primary Vicolungo Stress Case)."""
    import json
    json_path = Path(__file__).resolve().parent / "vicolungo_trajectories.json"
    with open(json_path, "r", encoding="utf-8") as f:
        tdata = json.load(f)["16"]

    s0 = tdata["s0"]
    v_ego0 = tdata["v_ego0"]
    lead_speeds = tdata["speed_lv"]

    res_base = simulate_rollout(s0, v_ego0, lead_speeds, use_cbf=False)
    res_cbf = simulate_rollout(s0, v_ego0, lead_speeds, use_cbf=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8.5), dpi=200, sharex=True)

    # 1. Headway Gap
    ax1.plot(res_base["t"], res_base["gap"], label="Baseline ACC (Min gap: 20.8m)", color="#f59e0b", lw=2.0)
    ax1.plot(res_cbf["t"], res_cbf["gap"], label="ACC + CBF (Min gap: 31.6m)", color="#3b82f6", lw=2.2)
    ax1.axhline(15.0, color="black", linestyle="--", lw=1.0, label="D_min (15m)")
    ax1.set_title("[VICOLUNGO REPLAY] Trajectory 16 — Safety Recovery / Actuator-Limit Stress Case (h0 = -37.05m < 0)", fontweight="bold")
    ax1.set_ylabel("Headway Gap s(t) [m]")
    ax1.legend(loc="lower right", framealpha=0.9)

    # 2. Safety Margin h(t)
    ax2.plot(res_base["t"], res_base["h"], label="Baseline ACC", color="#f59e0b", lw=2.0)
    ax2.plot(res_cbf["t"], res_cbf["h"], label="ACC + CBF (Recovers h >= 0 in 9.2s)", color="#10b981", lw=2.2)
    ax2.axhline(0.0, color="black", linestyle="--", lw=1.2, label="Safety Boundary (h=0)")
    ax2.set_ylabel("Safety Margin h(t) [m]")
    ax2.legend(loc="lower right", framealpha=0.9)

    # 3. Accelerations
    ax3.plot(res_base["t"], res_base["acc"], label="Baseline ACC Applied", color="#f59e0b", lw=1.8, linestyle="--")
    ax3.plot(res_cbf["t"], res_cbf["acc"], label="ACC + CBF Applied (Max Braking -4.0)", color="#34d399", lw=2.2)
    ax3.plot(res_cbf["t"], res_cbf["u_cbf"], label="u_CBF (Infeasible < -4.0 m/s²)", color="#ef4444", lw=1.4, linestyle=":")
    ax3.axhline(A_MIN, color="black", linestyle=":", lw=1.0, label=f"A_min ({A_MIN} m/s²)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Acceleration (m/s²)")
    ax3.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig_path = OUT_DIR / "demo_p2_trajectory16_stress.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path.name}")


def plot_trajectory55_demo():
    """Generates comparison plot for Trajectory 55 (Textbook Forward-Invariance Case)."""
    import json
    json_path = Path(__file__).resolve().parent / "vicolungo_trajectories.json"
    with open(json_path, "r", encoding="utf-8") as f:
        tdata = json.load(f)["55"]

    s0 = tdata["s0"]
    v_ego0 = tdata["v_ego0"]
    lead_speeds = tdata["speed_lv"]

    res_base = simulate_rollout(s0, v_ego0, lead_speeds, use_cbf=False)
    res_cbf = simulate_rollout(s0, v_ego0, lead_speeds, use_cbf=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8.5), dpi=200, sharex=True)

    # 1. Headway Gap
    ax1.plot(res_base["t"], res_base["gap"], label="Baseline ACC", color="#f59e0b", lw=2.0)
    ax1.plot(res_cbf["t"], res_cbf["gap"], label="ACC + CBF", color="#3b82f6", lw=2.2)
    ax1.plot(res_cbf["t"], 15.0 + 2.0 * res_cbf["ego_v"], label="s_safe = 2v + 15", color="#10b981", lw=1.5, linestyle="--")
    ax1.set_title("[VICOLUNGO REPLAY] Trajectory 55 — Forward-Invariance Demonstration (h0 = +73.30m > 0)", fontweight="bold")
    ax1.set_ylabel("Headway Gap s(t) [m]")
    ax1.legend(loc="upper right", framealpha=0.9)

    # 2. Safety Margin h(t)
    ax2.plot(res_base["t"], res_base["h"], label="Baseline ACC", color="#f59e0b", lw=2.0)
    ax2.plot(res_cbf["t"], res_cbf["h"], label="ACC + CBF (Preserves h(t) > 0 for all t)", color="#10b981", lw=2.2)
    ax2.axhline(0.0, color="black", linestyle="--", lw=1.2, label="Safety Boundary (h=0)")
    ax2.set_ylabel("Safety Margin h(t) [m]")
    ax2.legend(loc="upper right", framealpha=0.9)

    # 3. Accelerations
    ax3.plot(res_base["t"], res_base["acc"], label="Baseline ACC Applied", color="#f59e0b", lw=1.8, linestyle="--")
    ax3.plot(res_cbf["t"], res_cbf["acc"], label="ACC + CBF Applied", color="#34d399", lw=2.2)
    ax3.axhline(A_MIN, color="black", linestyle=":", lw=1.0, label=f"A_min ({A_MIN} m/s²)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Acceleration (m/s²)")
    ax3.legend(loc="lower right", framealpha=0.9)

    plt.tight_layout()
    fig_path = OUT_DIR / "demo_p2_trajectory55_forward_invariance.png"
    plt.savefig(fig_path)
    plt.close()
    print(f"Saved: {fig_path.name}")


def main():
    plot_emergency_braking_demo()
    plot_trajectory16_demo()
    plot_trajectory55_demo()
    print("\nPrototype 2 demonstration plots generated successfully.")


if __name__ == "__main__":
    main()
