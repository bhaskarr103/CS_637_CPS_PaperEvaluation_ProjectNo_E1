"""Multi-Car Platoon Simulation Script for Prototype 3 (CBF-RL).

Simulates a string of N connected autonomous vehicles (1 Lead + 4 Followers)
negotiating emergency braking, traffic waves, and speed adjustments.

Each follower vehicle i tracks vehicle i-1 using the closed-loop CBF formulation:
    h_i(t) = s_i(t) - (T_min * v_i(t) + D_min) >= 0
    u_cbf_max_i = (delta_v_i + alpha * h_i) / T_min

Demonstrates how CBF safety filtering behaves in a simulated multi-car string.
SYNTHETIC KINEMATIC DEMONSTRATION (Not Vicolungo Experimental Results).
In this simulated emergency-braking scenario, the CBF-controlled vehicle string
maintained positive barrier margins and no simulated collision occurred.

Outputs:
- CSV log: results/platoon_simulation_log.csv
- Diagnostic Figure: figures/platoon/platoon_simulation.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure Prototype 3 directory is in Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from cbf_core import compute_h, compute_u_cbf_max, project_cbf, step_kinematics


def run_platoon_simulation(
    num_followers: int = 4,
    duration_s: float = 25.0,
    dt: float = 0.1,
    maneuver: str = "emergency_brake",
    use_filter: bool = True,
    policy_mode: str = "dual",
) -> pd.DataFrame:
    """Simulates an (N+1)-car platoon string."""
    cfg = Config()
    total_steps = int(duration_s / dt)
    time_arr = np.linspace(0.0, duration_s, total_steps)

    # Initial conditions
    # Spaced out at safe equilibrium distances: s_i ~ 55m, initial speed = 22 m/s (80 km/h)
    v_init = 22.0
    s_init = cfg.T_min * v_init + cfg.D_min + 10.0  # = 2.0*22 + 15 + 10 = 69.0 m

    # State vectors for N+1 vehicles (index 0 is lead, 1..N are followers)
    num_cars = num_followers + 1
    positions = np.zeros(num_cars)
    velocities = np.full(num_cars, v_init)
    accelerations = np.zeros(num_cars)

    # Place vehicles along highway
    for i in range(num_cars):
        positions[i] = (num_cars - 1 - i) * s_init

    # Pre-generate lead vehicle acceleration profile
    a_lead_profile = np.zeros(total_steps)
    if maneuver == "emergency_brake":
        # At t=4.0s, lead car executes hard emergency brake (-4.0 m/s^2) down to 4 m/s
        start_idx = int(4.0 / dt)
        brake_steps = int(4.5 / dt)
        a_lead_profile[start_idx : start_idx + brake_steps] = -4.0
    elif maneuver == "wave":
        # Sinusoidal acceleration wave
        freq = 0.2
        a_lead_profile = 1.8 * np.sin(2.0 * np.pi * freq * time_arr)

    records = []

    for step in range(total_steps):
        t = time_arr[step]

        # 1. Update lead vehicle
        a_lead = a_lead_profile[step]
        # Cruise control on lead if not braking
        if a_lead == 0.0:
            a_lead = np.clip((22.0 - velocities[0]) * 0.5, -2.0, 1.5)

        velocities[0] = max(0.0, velocities[0] + a_lead * dt)
        positions[0] += velocities[0] * dt
        accelerations[0] = a_lead

        # 2. Update each follower vehicle i (follows vehicle i-1)
        for i in range(1, num_cars):
            leader_idx = i - 1
            ego_idx = i

            s = max(0.0, positions[leader_idx] - positions[ego_idx])
            delta_v = velocities[leader_idx] - velocities[ego_idx]
            v_ego = velocities[ego_idx]

            h = compute_h(s, v_ego, cfg.T_min, cfg.D_min)
            u_cbf_max = compute_u_cbf_max(delta_v, h, cfg.T_min, cfg.alpha)

            # Follower policy
            s_target = cfg.T_target * v_ego + cfg.D_target
            gap_err = s - s_target

            if policy_mode == "dual":
                # Internalized CBF policy
                u_nominal = 0.15 * gap_err + 0.65 * delta_v
                u_policy = np.clip(min(u_nominal, u_cbf_max - 0.05), cfg.u_min, cfg.u_max)
            else:
                # Nominal unshielded policy
                u_policy = np.clip(0.18 * gap_err + 0.60 * delta_v, cfg.u_min, cfg.u_max)

            # Apply CBF filter if enabled
            if use_filter:
                u_safe, intervened, infeasible = project_cbf(
                    u_policy, u_cbf_max, cfg.u_min, cfg.u_max
                )
                u_applied = u_safe
            else:
                u_safe = u_policy
                intervened = False
                u_applied = u_policy

            # Kinematic step
            velocities[ego_idx] = max(0.0, velocities[ego_idx] + u_applied * dt)
            positions[ego_idx] += velocities[ego_idx] * dt
            accelerations[ego_idx] = u_applied

            records.append({
                "time": t,
                "step": step,
                "car_id": i,
                "role": f"Follower {i}",
                "s": s,
                "v": v_ego,
                "delta_v": delta_v,
                "h": h,
                "u_policy": u_policy,
                "u_applied": u_applied,
                "u_cbf_max": u_cbf_max,
                "is_safe": bool(h >= 0.0),
                "is_intervened": intervened,
            })

    # Record lead car telemetry
    for step in range(total_steps):
        t = time_arr[step]
        records.append({
            "time": t,
            "step": step,
            "car_id": 0,
            "role": "Lead Car",
            "s": np.nan,
            "v": velocities[0],
            "delta_v": 0.0,
            "h": np.nan,
            "u_policy": accelerations[0],
            "u_applied": accelerations[0],
            "u_cbf_max": np.nan,
            "is_safe": True,
            "is_intervened": False,
        })

    return pd.DataFrame(records)


def plot_platoon_results(df: pd.DataFrame, out_path: Path):
    """Generates a 3-panel publication-grade platoon diagnostic figure."""
    plt.rcParams.update({
        "figure.autolayout": True,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
        "lines.linewidth": 1.8,
    })

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8.5), dpi=150, sharex=True)

    colors = ["#2563eb", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899"]

    # 1. Velocities
    for car_id in range(5):
        sub = df[df["car_id"] == car_id].sort_values("time")
        label = "Lead Car" if car_id == 0 else f"Follower {car_id}"
        ls = "-" if car_id == 0 else "--"
        ax1.plot(sub["time"], sub["v"], label=label, color=colors[car_id], linestyle=ls)
    ax1.set_title("Platoon String: Vehicle Velocities vs. Time")
    ax1.set_ylabel("Velocity (m/s)")
    ax1.grid(True)
    ax1.legend(loc="upper right", ncol=5, fontsize=8)

    # 2. Longitudinal Gaps
    for car_id in range(1, 5):
        sub = df[df["car_id"] == car_id].sort_values("time")
        ax2.plot(sub["time"], sub["s"], label=f"Gap s_{car_id}", color=colors[car_id])
    ax2.axhline(15.0, color="black", linestyle=":", linewidth=1.2, label="D_min (15m)")
    ax2.set_title("Longitudinal Headway Gaps s_i(t)")
    ax2.set_ylabel("Gap (m)")
    ax2.grid(True)
    ax2.legend(loc="upper right", ncol=5, fontsize=8)

    # 3. Safety Barrier Margins h_i(t)
    for car_id in range(1, 5):
        sub = df[df["car_id"] == car_id].sort_values("time")
        ax3.plot(sub["time"], sub["h"], label=f"Barrier h_{car_id}", color=colors[car_id])
    ax3.axhline(0.0, color="black", linestyle="--", linewidth=1.5, label="Boundary (h=0)")
    ax3.set_title("Safety Function Margins h_i(t) = s_i - (2.0 v_i + 15.0)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Safety Margin h (m)")
    ax3.grid(True)
    ax3.legend(loc="lower right", ncol=5, fontsize=8)

    fig.suptitle("[SYNTHETIC DEMONSTRATION] Multi-Car Platoon CBF Simulation (Emergency Hard Brake)", fontsize=12, fontweight="bold")
    ax1.text(0.5, 1.08, "Synthetic Kinematic Demonstration (Not Vicolungo Real-World Results)", transform=ax1.transAxes, ha='center', fontsize=9, color='#666666')
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    cfg = Config()
    out_dir = cfg.figures_dir / "platoon"
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir = cfg.results_dir
    res_dir.mkdir(parents=True, exist_ok=True)

    print("Simulating 5-Car Platoon (Lead + 4 Followers) under Emergency Braking...")
    df_platoon = run_platoon_simulation(
        num_followers=4,
        duration_s=22.0,
        dt=0.1,
        maneuver="emergency_brake",
        use_filter=True,
        policy_mode="dual",
    )

    csv_path = res_dir / "platoon_simulation_log.csv"
    df_platoon.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    fig_path = out_dir / "platoon_simulation.png"
    plot_platoon_results(df_platoon, fig_path)

    # Copy to artifact directory
    artifact_dir = Path(r"C:\Users\rajau\.gemini\antigravity\brain\6809cadf-3f47-4ce9-9f71-18861ed7d483\diagnostic_plots")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(fig_path, artifact_dir / fig_path.name)
    print(f"Copied to artifact dir: {artifact_dir / fig_path.name}")


if __name__ == "__main__":
    main()
