"""Central Configuration for Prototype 3 (CBF-RL Car-Following).

All physical constants, barrier parameters, reward weights, and simulation
settings are defined here in a single, transparent dataclass.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # -------------------------------------------------------------------------
    # System & Kinematics Constants
    # -------------------------------------------------------------------------
    dt: float = 0.1  # Integration time-step (s) - matches 10 Hz Vicolungo data
    u_min: float = -4.0  # Maximum emergency braking deceleration (m/s^2)
    u_max: float = 2.5  # Maximum throttle acceleration (m/s^2)
    jerk_max: float = 5.0  # Maximum comfortable jerk (m/s^3)
    v_min: float = 0.0  # Follower non-negative velocity bound (m/s)
    v_max: float = 35.0  # Maximum highway cruising velocity (m/s)

    # -------------------------------------------------------------------------
    # Control Barrier Function (CBF) Specification
    # Safety boundary: h(s, v_ego) = s - (T_min * v_ego + D_min) >= 0
    # -------------------------------------------------------------------------
    T_min: float = 2.0  # Minimum safe time headway (s)
    D_min: float = 15.0  # Minimum safe standstill margin (m)
    alpha: float = 0.1  # Linear class-K decay parameter (s^-1)

    # -------------------------------------------------------------------------
    # Nominal Car-Following Task Specification
    # Target equilibrium: s*(v_ego) = T_target * v_ego + D_target
    # -------------------------------------------------------------------------
    T_target: float = 2.5  # Nominal target time headway (s) (> T_min)
    D_target: float = 18.0  # Nominal target standstill distance (m) (> D_min)
    v_target: float = 20.0  # Desired cruising speed when open road (m/s)

    # -------------------------------------------------------------------------
    # Reward Design Weights
    # r = r_track + r_comf + r_prog + r_cbf
    # -------------------------------------------------------------------------
    w_gap: float = 0.6  # Weight for saturated headway error penalty
    w_vel: float = 0.4  # Weight for saturated relative velocity penalty
    w_ctrl: float = 0.1  # Weight for control effort penalty (u / u_max)^2
    w_jerk: float = 0.1  # Weight for jerk / comfort penalty
    r_prog_max: float = 0.3  # Maximum forward progress bonus per step
    r_crash: float = -50.0  # Terminal penalty for physical collision (s <= 0)

    # Bounding scales for saturated hyperbolic tangent penalties
    d_scale: float = 15.0  # Headway error normalization scale (m)
    v_scale: float = 10.0  # Velocity error normalization scale (m/s)

    # -------------------------------------------------------------------------
    # CBF Reward Penalty Weights (Yang et al. Equations 22-23)
    # r_cbf = w_cbf1 * min(psi, 0) + w_cbf2 * [exp(-||u_pol - u_safe||^2 / sigma^2) - 1]
    # -------------------------------------------------------------------------
    w_cbf1: float = 1.0  # Linear penalty on constraint violation magnitude
    w_cbf2: float = 2.0  # Saturated exponential action divergence penalty
    sigma_sq: float = 0.5  # Variance parameter in action correction penalty (m^2/s^4)

    # -------------------------------------------------------------------------
    # Observation Normalization Scales (Map state to ~ [-1, 1])
    # -------------------------------------------------------------------------
    obs_s_scale: float = 50.0  # (s - D_min) / obs_s_scale
    obs_v_scale: float = 30.0  # (v_ego - v_target) / obs_v_scale
    obs_dv_scale: float = 10.0  # delta_v / obs_dv_scale
    obs_h_scale: float = 30.0  # h / obs_h_scale

    # -------------------------------------------------------------------------
    # Simulation & Environment Limits
    # -------------------------------------------------------------------------
    max_episode_steps: int = 200  # 20.0 seconds at dt = 0.1s

    # -------------------------------------------------------------------------
    # Paths (Prioritize local Prototype 3 copy, fallback to Prototype 2)
    # -------------------------------------------------------------------------
    vicolungo_csv_path: Path = (
        Path(r"E:\CPS Project\Prototype 3\Vicolungo.csv")
        if Path(r"E:\CPS Project\Prototype 3\Vicolungo.csv").exists()
        else Path(r"E:\CPS Project\Prototype 2\Vicolungo.csv")
    )
    results_dir: Path = Path(r"E:\CPS Project\Prototype 3\results")
    figures_dir: Path = Path(r"E:\CPS Project\Prototype 3\figures")

