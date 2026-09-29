"""Generates high-resolution simulation telemetry for the interactive visualizer.

Evaluates trained policy checkpoints across 6 scenarios:
1. Normal Cruising
2. Moderate Lead Braking
3. Sudden Emergency Braking
4. Close Headway / Cut-In
5. Vicolungo Trajectory 3 (Initially Safe Real-Data Replay)
6. Vicolungo Trajectory 1 (Initially Penetrated Real-Data Replay)

Across 4 Ablations (Nominal, Filter Only, Reward Only, Dual CBF-RL)
and 2 Deployment Modes (Filter ON vs Filter OFF).

Uses exact kinematics and CBF equations from cbf_core.py and env.py.
Outputs: visualizer_data.json
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
import torch

# Ensure Prototype 3 is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from cbf_core import compute_h, compute_u_cbf_max, project_cbf, step_kinematics
from ppo import PPOAgent


def load_checkpoints(cfg: Config) -> Dict[str, PPOAgent]:
    """Loads Seed 42 checkpoints for all 4 ablations."""
    ckpt_dir = cfg.results_dir / "checkpoints"
    agents = {}
    for var in ["Nominal", "Filter_Only", "Reward_Only", "Dual_CBF_RL"]:
        ckpt_file = ckpt_dir / f"{var}_seed42.pt"
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_file}")
        agent = PPOAgent(obs_dim=4, act_dim=1, device="cpu")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        agent.network.load_state_dict(ckpt["model_state_dict"])
        agents[var] = agent
    return agents


def get_vicolungo_speeds(cfg: Config, traj_id: int) -> Tuple[float, float, np.ndarray]:
    """Extracts initial gap, follower speed, and lead speed sequence from Vicolungo.csv."""
    df = pd.read_csv(cfg.vicolungo_csv_path)
    tdf = df[df["Trajectory_ID"] == traj_id].sort_values("Time_Index").reset_index(drop=True)
    s_0 = float(tdf.loc[0, "Spatial_Gap"])
    v_ego_0 = float(tdf.loc[0, "Speed_FAV"])
    lead_speeds = tdf["Speed_LV"].to_numpy(dtype=float)
    return s_0, v_ego_0, lead_speeds


def run_scenario_rollout(
    agent: PPOAgent,
    cfg: Config,
    s_init: float,
    v_ego_init: float,
    v_lead_profile: np.ndarray,
    use_filter: bool,
) -> Dict[str, Any]:
    """Runs a single closed-loop rollout using exact Prototype 3 kinematics."""
    steps = len(v_lead_profile) - 1
    dt = cfg.dt

    s = s_init
    v_ego = v_ego_init
    v_lead = float(v_lead_profile[0])

    t_series = [0.0]
    s_series = [round(s, 2)]
    v_ego_series = [round(v_ego, 2)]
    v_lead_series = [round(v_lead, 2)]
    delta_v_series = [round(v_lead - v_ego, 2)]
    h_series = [round(compute_h(s, v_ego, cfg.T_min, cfg.D_min), 2)]
    s_safe_series = [round(cfg.T_min * v_ego + cfg.D_min, 2)]
    u_policy_series = [0.0]
    u_cbf_series = [0.0]
    u_safe_series = [0.0]
    u_applied_series = [0.0]
    intervened_series = [False]
    infeasible_series = [False]
    collision_series = [False]

    collisions = 0
    interventions = 0

    for k in range(steps):
        # 1. Observation
        delta_v = v_lead - v_ego
        h = compute_h(s, v_ego, cfg.T_min, cfg.D_min)

        norm_s = (s - cfg.D_min) / cfg.obs_s_scale
        norm_v = (v_ego - cfg.v_target) / cfg.obs_v_scale
        norm_dv = delta_v / cfg.obs_dv_scale
        norm_h = h / cfg.obs_h_scale
        obs = np.array([norm_s, norm_v, norm_dv, norm_h], dtype=np.float32)

        # 2. Query policy deterministically
        action, _, _ = agent.select_action(obs, deterministic=True)
        a_scalar = float(np.clip(np.squeeze(action), -1.0, 1.0))
        if a_scalar < 0.0:
            u_policy = a_scalar * abs(cfg.u_min)
        else:
            u_policy = a_scalar * cfg.u_max

        # 3. CBF bound & projection
        u_cbf_max = compute_u_cbf_max(delta_v, h, cfg.T_min, cfg.alpha)
        u_safe, is_intervened, is_infeasible = project_cbf(
            u_policy=u_policy,
            u_cbf_max=u_cbf_max,
            u_min=cfg.u_min,
            u_max=cfg.u_max,
        )

        # 4. Action execution
        if use_filter:
            u_applied = u_safe
        else:
            u_applied = max(cfg.u_min, min(cfg.u_max, u_policy))

        # 5. Physical transition
        v_lead_next = float(v_lead_profile[k + 1])
        s_next, v_ego_next, _ = step_kinematics(
            s=s,
            v_ego=v_ego,
            v_lead=v_lead,
            u_applied=u_applied,
            a_lead=0.0,
            dt=dt,
            v_lead_next=v_lead_next,
        )

        is_coll = bool(s_next <= 0.0)
        if is_coll:
            collisions += 1
        if is_intervened:
            interventions += 1

        # Update state
        s = s_next
        v_ego = v_ego_next
        v_lead = v_lead_next

        # Record telemetry
        t_now = round((k + 1) * dt, 2)
        h_now = round(compute_h(s, v_ego, cfg.T_min, cfg.D_min), 2)
        s_safe_now = round(cfg.T_min * v_ego + cfg.D_min, 2)

        t_series.append(t_now)
        s_series.append(round(s, 2))
        v_ego_series.append(round(v_ego, 2))
        v_lead_series.append(round(v_lead, 2))
        delta_v_series.append(round(v_lead - v_ego, 2))
        h_series.append(h_now)
        s_safe_series.append(s_safe_now)
        u_policy_series.append(round(u_policy, 2))
        u_cbf_series.append(round(min(10.0, max(-15.0, u_cbf_max)), 2))
        u_safe_series.append(round(u_safe, 2))
        u_applied_series.append(round(u_applied, 2))
        intervened_series.append(bool(is_intervened))
        infeasible_series.append(bool(is_infeasible))
        collision_series.append(is_coll)

    h_arr = np.array(h_series)
    gaps_arr = np.array(s_series)

    return {
        "telemetry": {
            "t": t_series,
            "s": s_series,
            "v_ego": v_ego_series,
            "v_lead": v_lead_series,
            "delta_v": delta_v_series,
            "h": h_series,
            "s_safe": s_safe_series,
            "u_policy": u_policy_series,
            "u_cbf": u_cbf_series,
            "u_safe": u_safe_series,
            "u_applied": u_applied_series,
            "is_intervened": intervened_series,
            "is_infeasible": infeasible_series,
            "is_collision": collision_series,
        },
        "summary": {
            "min_gap": float(np.min(gaps_arr)),
            "min_h": float(np.min(h_arr)),
            "pct_safe": float(np.mean(h_arr >= 0.0) * 100.0),
            "had_collision": bool(np.min(gaps_arr) <= 0.0),
            "intervention_pct": float((interventions / max(1, steps)) * 100.0),
        },
    }


def build_all_scenarios(cfg: Config) -> Dict[str, Any]:
    """Constructs the lead speed profiles for all 6 scenarios."""
    dt = cfg.dt
    scenarios = {}

    # Scenario 1: Normal Cruising (20s)
    t1 = np.arange(0, 20.1, dt)
    v1 = 22.0 + 0.8 * np.sin(0.4 * t1) + 0.4 * np.sin(0.9 * t1)
    scenarios["cruising"] = {
        "title": "1. Normal Highway Cruising",
        "description": "Lead vehicle cruises at ~80 km/h with subtle speed fluctuations. Follower maintains smooth headway equilibrium.",
        "category": "SYNTHETIC DEMONSTRATION",
        "s_init": 68.0,
        "v_ego_init": 22.0,
        "v_lead_profile": v1,
    }

    # Scenario 2: Moderate Lead Braking (20s)
    t2 = np.arange(0, 20.1, dt)
    v2 = np.full_like(t2, 24.0)
    for idx, t in enumerate(t2):
        if 3.0 <= t <= 7.0:
            v2[idx] = 24.0 - 2.0 * (t - 3.0)  # decelerate at -2.0 m/s² down to 16 m/s
        elif t > 7.0:
            v2[idx] = 16.0
    scenarios["lead_braking"] = {
        "title": "2. Moderate Lead Braking",
        "description": "Lead decelerates steadily from 86 km/h to 58 km/h at -2.0 m/s². Tests standard deceleration tracking.",
        "category": "SYNTHETIC DEMONSTRATION",
        "s_init": 72.0,
        "v_ego_init": 24.0,
        "v_lead_profile": v2,
    }

    # Scenario 3: Sudden Emergency Braking (20s)
    t3 = np.arange(0, 20.1, dt)
    v3 = np.full_like(t3, 25.0)
    for idx, t in enumerate(t3):
        if 3.0 <= t <= 7.0:
            v3[idx] = 25.0 - 3.8 * (t - 3.0)  # hard braking at -3.8 m/s² down to 9.8 m/s
        elif t > 7.0:
            v3[idx] = 9.8
    scenarios["emergency_braking"] = {
        "title": "3. Sudden Emergency Braking",
        "description": "Lead slams brakes at -3.8 m/s² (near physical limit). Severe collision risk if unshielded or uninternalized.",
        "category": "SYNTHETIC DEMONSTRATION",
        "s_init": 70.0,
        "v_ego_init": 25.0,
        "v_lead_profile": v3,
    }

    # Scenario 4: Close Initial Headway / Cut-In (20s)
    t4 = np.arange(0, 20.1, dt)
    v4 = np.full_like(t4, 22.0)
    scenarios["close_cutin"] = {
        "title": "4. Close Initial Cut-In (h0 < 0)",
        "description": "A vehicle cuts in at close range (gap = 26m at 80 km/h, h0 = -33m). Tests recovery from initial penetration.",
        "category": "SYNTHETIC DEMONSTRATION",
        "s_init": 26.0,
        "v_ego_init": 22.0,
        "v_lead_profile": v4,
    }

    # Scenario 5: Vicolungo Trajectory 3 (Real-Data Lead Replay, Initially Safe)
    s0_t3, v0_t3, lead_t3 = get_vicolungo_speeds(cfg, traj_id=3)
    scenarios["vicolungo_traj3"] = {
        "title": "5. Vicolungo Trajectory 3 (Initially Safe Replay)",
        "description": "Real recorded human lead speed from Italian A4 highway. Starts safe (h0 = +51.3m). Tests Forward Invariance preservation.",
        "category": "VICOLUNGO REAL-DATA REPLAY",
        "s_init": s0_t3,
        "v_ego_init": v0_t3,
        "v_lead_profile": lead_t3[:250],  # 25 seconds
    }

    # Scenario 6: Vicolungo Trajectory 1 (Real-Data Lead Replay, Initially Penetrated)
    s0_t1, v0_t1, lead_t1 = get_vicolungo_speeds(cfg, traj_id=1)
    scenarios["vicolungo_traj1"] = {
        "title": "6. Vicolungo Trajectory 1 (Initially Penetrated Replay)",
        "description": "Real recorded human lead speed from Italian A4 highway. Starts penetrated (h0 = -34.1m). Tests Safe Recovery under actuator saturation.",
        "category": "VICOLUNGO REAL-DATA REPLAY",
        "s_init": s0_t1,
        "v_ego_init": v0_t1,
        "v_lead_profile": lead_t1[:200],  # 20 seconds
    }

    return scenarios


def generate_all_telemetry():
    """Compiles and exports the complete visualizer database."""
    print("=" * 70)
    print("GENERATING VISUALIZER DATABASE (visualizer_data.json)")
    print("=" * 70)

    cfg = Config()
    agents = load_checkpoints(cfg)
    scenarios_meta = build_all_scenarios(cfg)

    database: Dict[str, Any] = {
        "constants": {
            "T_min": cfg.T_min,
            "D_min": cfg.D_min,
            "alpha": cfg.alpha,
            "u_min": cfg.u_min,
            "u_max": cfg.u_max,
            "dt": cfg.dt,
        },
        "scenarios": {},
    }

    for sc_key, sc_info in scenarios_meta.items():
        print(f"\nProcessing Scenario: {sc_info['title']}...")
        sc_data = {
            "title": sc_info["title"],
            "description": sc_info["description"],
            "category": sc_info["category"],
            "s_init": sc_info["s_init"],
            "v_ego_init": sc_info["v_ego_init"],
            "v_lead_init": float(sc_info["v_lead_profile"][0]),
            "runs": {},
        }

        for var_name, agent in agents.items():
            for use_filter in [True, False]:
                mode_str = "filter_on" if use_filter else "filter_off"
                run_key = f"{var_name}__{mode_str}"
                rollout = run_scenario_rollout(
                    agent=agent,
                    cfg=cfg,
                    s_init=sc_info["s_init"],
                    v_ego_init=sc_info["v_ego_init"],
                    v_lead_profile=sc_info["v_lead_profile"],
                    use_filter=use_filter,
                )
                sc_data["runs"][run_key] = rollout

        database["scenarios"][sc_key] = sc_data

    out_file = Path(__file__).parent / "visualizer_data.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(database, f, indent=2)

    print(f"\nSuccessfully saved visualizer database: {out_file.name} ({out_file.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    generate_all_telemetry()
