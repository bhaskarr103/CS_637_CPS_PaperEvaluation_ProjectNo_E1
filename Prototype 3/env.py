"""Longitudinal Car-Following Environment for CBF-RL.

Implements an interactive, closed-loop car-following environment compatible
with standard RL training and evaluation protocols.

Key Features:
- Two scenario modes:
    1. 'synthetic': Procedural generation of varied cruising, braking, and
       perturbation profiles for training.
    2. 'vicolungo': Real-world exogenous lead vehicle profiles from the
       OpenACC Vicolungo naturalistic dataset for zero-shot testing.
       The lead vehicle velocity v_lead[k] is taken directly from the recorded
       dataset Speed_LV[k] without numerical integration.
- Strictly isolated 4-way ablation regimes:
    * Nominal RL (use_cbf_filter=False, use_cbf_reward=False):
        u_applied = u_policy; nominal reward uses u_policy; no CBF reward.
    * Filter Only (use_cbf_filter=True, use_cbf_reward=False):
        u_applied = u_safe; nominal reward uses u_safe; no CBF reward.
    * Reward Only (use_cbf_filter=False, use_cbf_reward=True):
        u_applied = u_policy; nominal reward uses u_policy; CBF reward added.
    * Dual CBF-RL (use_cbf_filter=True, use_cbf_reward=True):
        u_applied = u_safe; nominal reward uses u_safe; CBF reward added.
- Genuinely closed-loop: The follower vehicle trajectory is 100% generated
  by the control action. Dataset follower recordings are discarded after t=0.
- Transparent logging: Returns rich telemetry in info dict (h, u_cbf_max,
  u_policy, u_safe, is_intervened, is_infeasible, is_violated, is_collision).
"""

from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd

from config import Config
from cbf_core import (
    compute_h,
    compute_u_cbf_max,
    project_cbf,
    step_kinematics,
    compute_cbf_reward,
    compute_nominal_reward,
)


class CarFollowingEnv:
    """Lightweight, standard-compliant car-following environment."""

    def __init__(
        self,
        config: Optional[Config] = None,
        mode: str = "synthetic",
        use_cbf_filter: bool = True,
        use_cbf_reward: bool = True,
    ):
        """Initializes the environment.

        Args:
            config: Configuration instance. If None, default Config() is used.
            mode: 'synthetic' for randomized training or 'vicolungo' for dataset replay.
            use_cbf_filter: If True, u_applied = u_safe and nominal reward uses u_safe.
                            If False, u_applied = u_policy and nominal reward uses u_policy.
            use_cbf_reward: If True, adds r_cbf to total reward.
        """
        self.cfg = config if config is not None else Config()
        self.mode = mode
        self.use_cbf_filter = use_cbf_filter
        self.use_cbf_reward = use_cbf_reward

        # State dimensions: [norm_s, norm_v_ego, norm_dv, norm_h]
        self.obs_dim = 4
        self.act_dim = 1

        # Internal state variables
        self.s: float = 0.0
        self.v_ego: float = 0.0
        self.v_lead: float = 0.0
        self.a_lead: float = 0.0
        self.u_prev: float = 0.0
        self.step_idx: int = 0

        # Dataset cache for Vicolungo mode
        self._vicolungo_df: Optional[pd.DataFrame] = None
        self._vicolungo_trajs: Dict[int, pd.DataFrame] = {}
        self.current_traj_id: Optional[int] = None
        self._lead_speeds: Optional[np.ndarray] = None
        self._lead_accs: Optional[np.ndarray] = None
        self._traj_len: int = 0

        if self.mode == "vicolungo":
            self._load_vicolungo()

        # Synthetic generator parameters
        self._synthetic_a_lead_profile: Optional[np.ndarray] = None

    def _load_vicolungo(self) -> None:
        """Loads and pre-processes the Vicolungo naturalistic dataset."""
        if not self.cfg.vicolungo_csv_path.exists():
            raise FileNotFoundError(
                f"Vicolungo dataset not found at: {self.cfg.vicolungo_csv_path}"
            )
        self._vicolungo_df = pd.read_csv(self.cfg.vicolungo_csv_path)

        # Filter out Trajectory 6 (initially overlapping / physically unrecoverable case)
        # to focus on the 60 causally valid trajectories
        valid_ids = sorted(
            [tid for tid in self._vicolungo_df["Trajectory_ID"].unique() if tid != 6]
        )
        for tid in valid_ids:
            traj_data = self._vicolungo_df[
                self._vicolungo_df["Trajectory_ID"] == tid
            ].sort_values("Time_Index").reset_index(drop=True)
            if len(traj_data) > 10:
                self._vicolungo_trajs[tid] = traj_data

    def get_valid_trajectory_ids(self) -> list:
        """Returns list of valid Vicolungo trajectory IDs."""
        return list(self._vicolungo_trajs.keys())

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment to an initial state.

        In 'vicolungo' mode:
            Exogenous lead vehicle speed sequence is loaded directly from dataset Speed_LV.
            Follower state is initialized once at t=0 (s_0 and v_ego_0).
            Recorded follower trajectory after t=0 is completely discarded.
        In 'synthetic' mode:
            Initial conditions and lead maneuvers are randomized procedurally.
        """
        if seed is not None:
            np.random.seed(seed)

        self.step_idx = 0
        self.u_prev = 0.0

        if self.mode == "vicolungo":
            if not self._vicolungo_trajs:
                self._load_vicolungo()

            traj_ids = self.get_valid_trajectory_ids()
            if options and "traj_id" in options:
                self.current_traj_id = int(options["traj_id"])
            else:
                self.current_traj_id = int(np.random.choice(traj_ids))

            traj_df = self._vicolungo_trajs[self.current_traj_id]
            self._lead_speeds = traj_df["Speed_LV"].to_numpy(dtype=float)
            self._lead_accs = traj_df["Acc_LV"].to_numpy(dtype=float)
            self._traj_len = len(traj_df)

            # Initialize follower state once at t=0
            # Note: Dataset follower trajectory for t > 0 is discarded!
            self.s = float(traj_df.loc[0, "Spatial_Gap"])
            self.v_ego = float(traj_df.loc[0, "Speed_FAV"])

            # Lead vehicle speed is taken EXACTLY from recorded trajectory
            self.v_lead = float(self._lead_speeds[0])
            self.a_lead = float(self._lead_accs[0])

        elif self.mode == "synthetic":
            # Procedural initialization:
            # Safe initial gap ensuring h0 >= 0
            self.v_ego = float(np.random.uniform(12.0, 26.0))
            self.v_lead = float(self.v_ego + np.random.uniform(-3.0, 3.0))

            # Initial gap placed safely within [s_safe_min + 5m, s_safe_min + 35m]
            s_safe_min = self.cfg.T_min * self.v_ego + self.cfg.D_min
            self.s = float(s_safe_min + np.random.uniform(5.0, 30.0))
            self.a_lead = 0.0

            # Pre-generate synthetic lead acceleration profile for episode
            self._traj_len = self.cfg.max_episode_steps
            self._synthetic_a_lead_profile = np.zeros(self._traj_len, dtype=float)

            # Insert diverse driving maneuvers
            maneuver_type = np.random.choice(["cruising", "braking", "wave", "cut_in"])
            if maneuver_type == "braking":
                # Lead executes moderate to hard brake at random time
                brake_start = np.random.randint(20, 60)
                brake_dur = np.random.randint(20, 50)
                brake_acc = np.random.uniform(-3.5, -1.0)
                self._synthetic_a_lead_profile[brake_start : brake_start + brake_dur] = brake_acc
            elif maneuver_type == "wave":
                # Sinusoidal stop-and-go disturbance
                freq = np.random.uniform(0.1, 0.4)
                amp = np.random.uniform(0.8, 2.0)
                t = np.arange(self._traj_len) * self.cfg.dt
                self._synthetic_a_lead_profile = amp * np.sin(2.0 * np.pi * freq * t)
            elif maneuver_type == "cruising":
                # Cruising with small smooth Ornstein-Uhlenbeck style noise
                noise = np.random.randn(self._traj_len) * 0.3
                self._synthetic_a_lead_profile = np.clip(noise, -1.0, 1.0)
            elif maneuver_type == "cut_in":
                # Close initial condition to test barrier boundary proximity
                self.s = float(s_safe_min + np.random.uniform(0.5, 4.0))

        obs = self._get_obs()
        h_0 = compute_h(self.s, self.v_ego, self.cfg.T_min, self.cfg.D_min)
        info = {
            "s": self.s,
            "v_ego": self.v_ego,
            "v_lead": self.v_lead,
            "delta_v": self.v_lead - self.v_ego,
            "h": h_0,
            "trajectory_id": self.current_traj_id,
        }
        return obs, info

    def _get_obs(self) -> np.ndarray:
        """Constructs the normalized observation vector o in ~ [-1, 1]."""
        delta_v = self.v_lead - self.v_ego
        h = compute_h(self.s, self.v_ego, self.cfg.T_min, self.cfg.D_min)

        norm_s = (self.s - self.cfg.D_min) / self.cfg.obs_s_scale
        norm_v = (self.v_ego - self.cfg.v_target) / self.cfg.obs_v_scale
        norm_dv = delta_v / self.cfg.obs_dv_scale
        norm_h = h / self.cfg.obs_h_scale

        return np.array([norm_s, norm_v, norm_dv, norm_h], dtype=np.float32)

    def _action_to_acc(self, action: Union[float, np.ndarray]) -> float:
        """Maps continuous policy action a in [-1, 1] to physical acceleration u.

        Piecewise mapping centered at a=0 -> u=0.0 m/s^2:
            a < 0  => u in [u_min, 0.0]  (braking)
            a >= 0 => u in [0.0, u_max]  (throttle)
        """
        a = float(np.clip(np.squeeze(action), -1.0, 1.0))
        if a < 0.0:
            u = a * abs(self.cfg.u_min)  # a=-1 => u=-4.0
        else:
            u = a * self.cfg.u_max  # a=+1 => u=+2.5
        return float(u)

    def step(
        self, action: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one discrete transition step.

        Args:
            action: Policy action in [-1, 1].

        Returns:
            obs: Normalized observation array.
            reward: Total scalar reward (nominal + cbf_reward if enabled).
            terminated: True if physical collision occurs (s <= 0).
            truncated: True if maximum episode duration is reached.
            info: Comprehensive telemetry dictionary.
        """
        # 1. Map policy action to physical candidate acceleration
        u_policy = self._action_to_acc(action)

        # 2. Compute safety margin and CBF acceleration bound
        delta_v = self.v_lead - self.v_ego
        h_curr = compute_h(self.s, self.v_ego, self.cfg.T_min, self.cfg.D_min)
        u_cbf_max = compute_u_cbf_max(
            delta_v, h_curr, self.cfg.T_min, self.cfg.alpha
        )

        # 3. Closed-form CBF projection
        u_safe, is_intervened, is_infeasible = project_cbf(
            u_policy=u_policy,
            u_cbf_max=u_cbf_max,
            u_min=self.cfg.u_min,
            u_max=self.cfg.u_max,
        )

        # 4. Determine executed action and action for nominal reward
        # Clean four-way ablation enforcement:
        if self.use_cbf_filter:
            u_applied = u_safe
            u_for_nominal_reward = u_safe
        else:
            u_applied = max(self.cfg.u_min, min(self.cfg.u_max, u_policy))
            u_for_nominal_reward = u_applied

        # 5. Determine next lead vehicle velocity and acceleration
        if self.mode == "vicolungo":
            # Exact exogenous lead velocity: v_lead[k+1] = Speed_LV[k+1]
            if self.step_idx + 1 < self._traj_len:
                v_lead_next = float(self._lead_speeds[self.step_idx + 1])
                a_lead = float(self._lead_accs[self.step_idx])
            else:
                v_lead_next = float(self._lead_speeds[-1])
                a_lead = 0.0
            a_lead_step = a_lead
        else:
            # Synthetic mode: lead acceleration from generated profile
            if (
                self._synthetic_a_lead_profile is not None
                and self.step_idx < len(self._synthetic_a_lead_profile)
            ):
                a_lead = float(self._synthetic_a_lead_profile[self.step_idx])
            else:
                a_lead = 0.0
            v_lead_next = None  # Will be integrated by step_kinematics
            a_lead_step = a_lead

        # 6. Step physical kinematics
        s_next, v_ego_next, v_lead_next_val = step_kinematics(
            s=self.s,
            v_ego=self.v_ego,
            v_lead=self.v_lead,
            u_applied=u_applied,
            a_lead=a_lead_step,
            dt=self.cfg.dt,
            v_lead_next=v_lead_next,
        )

        # 7. Compute nominal reward components using the applied action for this ablation
        r_nominal, r_components = compute_nominal_reward(
            s=self.s,
            v_ego=self.v_ego,
            v_lead=self.v_lead,
            u_action=u_for_nominal_reward,
            u_prev=self.u_prev,
            dt=self.cfg.dt,
            T_target=self.cfg.T_target,
            D_target=self.cfg.D_target,
            v_target=self.cfg.v_target,
            w_gap=self.cfg.w_gap,
            w_vel=self.cfg.w_vel,
            w_ctrl=self.cfg.w_ctrl,
            w_jerk=self.cfg.w_jerk,
            r_prog_max=self.cfg.r_prog_max,
            r_crash=self.cfg.r_crash,
            d_scale=self.cfg.d_scale,
            v_scale=self.cfg.v_scale,
            u_max=self.cfg.u_max,
            jerk_max=self.cfg.jerk_max,
            T_min=self.cfg.T_min,
            D_min=self.cfg.D_min,
        )

        # 8. Compute dual CBF reward penalty
        r_cbf, r_cbf1, r_cbf2 = compute_cbf_reward(
            u_policy=u_policy,
            u_cbf_max=u_cbf_max,
            u_safe=u_safe,
            w_cbf1=self.cfg.w_cbf1,
            w_cbf2=self.cfg.w_cbf2,
            sigma_sq=self.cfg.sigma_sq,
        )

        # Total step reward based on ablation configuration
        if self.use_cbf_reward:
            reward = float(r_nominal + r_cbf)
        else:
            reward = float(r_nominal)

        # 9. Update state
        self.s = s_next
        self.v_ego = v_ego_next
        self.v_lead = v_lead_next_val
        self.a_lead = a_lead_step
        self.u_prev = u_applied
        self.step_idx += 1

        # 10. Check termination conditions
        is_collision = bool(self.s <= 0.0)
        terminated = is_collision
        truncated = bool(self.step_idx >= self._traj_len or self.step_idx >= self.cfg.max_episode_steps)

        # 11. Construct telemetry dictionary
        h_next = compute_h(self.s, self.v_ego, self.cfg.T_min, self.cfg.D_min)
        info = {
            "step": self.step_idx,
            "s": self.s,
            "v_ego": self.v_ego,
            "v_lead": self.v_lead,
            "delta_v": self.v_lead - self.v_ego,
            "h": h_next,
            "u_policy": u_policy,
            "u_safe": u_safe,
            "u_applied": u_applied,
            "u_for_nominal_reward": u_for_nominal_reward,
            "u_cbf_max": u_cbf_max,
            "is_intervened": is_intervened,
            "is_infeasible": is_infeasible,
            "is_violated": bool(h_next < 0.0),
            "is_collision": is_collision,
            "r_nominal": r_nominal,
            "r_cbf": r_cbf,
            "r_cbf1": r_cbf1,
            "r_cbf2": r_cbf2,
            **r_components,
        }

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info
