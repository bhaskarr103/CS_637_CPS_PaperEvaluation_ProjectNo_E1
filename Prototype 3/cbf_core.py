"""Core Mathematical Functions for Control Barrier Functions (CBF) and Dynamics.

Pure mathematical implementations of:
- Safety function h(x)
- Discrete CBF acceleration upper bound u_cbf_max(x)
- Closed-form 1D projection with physical actuator bounds and infeasibility detection
- Forward Euler discrete vehicle kinematics (supporting exogenous recorded lead speed)
- Saturated nominal task reward and anti-stalling reward structure
- Dual-term CBF reward penalty (Yang et al. Equations 22-23)
"""

import math
from typing import Tuple, Dict, Any, Optional


def compute_h(s: float, v_ego: float, T_min: float = 2.0, D_min: float = 15.0) -> float:
    """Computes the continuous-time safety barrier function h.

    h(s, v_ego) = s - (T_min * v_ego + D_min)

    The safe set is C = { (s, v_ego) : h(s, v_ego) >= 0 }.
    """
    return s - (T_min * v_ego + D_min)


def compute_u_cbf_max(
    delta_v: float, h: float, T_min: float = 2.0, alpha: float = 0.1
) -> float:
    """Computes the maximum permissible follower acceleration satisfying the CBF condition.

    Continuous condition: dh/dt + alpha * h >= 0
    where dh/dt = delta_v - T_min * u, with delta_v = v_lead - v_ego.
    => delta_v - T_min * u + alpha * h >= 0
    => u <= (delta_v + alpha * h) / T_min

    Under forward Euler discretization, this yields the exact same discrete
    first-order bound for the linear headway barrier.
    """
    return (delta_v + alpha * h) / T_min


def project_cbf(
    u_policy: float,
    u_cbf_max: float,
    u_min: float = -4.0,
    u_max: float = 2.5,
) -> Tuple[float, bool, bool]:
    """Projects a candidate policy action onto the admissible safe control set.

    Solves the 1D QP:
        min_u  (1/2) * (u - u_policy)^2
        s.t.   u <= u_cbf_max
               u_min <= u <= u_max

    Returns:
        u_safe (float): Filtered action to apply to the vehicle actuator.
        is_intervened (bool): True if the filter altered u_policy.
        is_infeasible (bool): True if the barrier condition cannot be satisfied
                              even under maximum physical braking (u_cbf_max < u_min).
                              In this case, u_safe = u_min (maximum emergency brake).
    """
    # Check physical feasibility of the mathematical CBF constraint
    is_infeasible = u_cbf_max < u_min

    # If the policy action satisfies the CBF bound
    if u_policy <= u_cbf_max:
        # Respect physical actuator bounds
        u_safe = max(u_min, min(u_max, u_policy))
        is_intervened = abs(u_safe - u_policy) > 1e-6
    else:
        # Policy action violates CBF: project to u_cbf_max bounded by [u_min, u_max]
        u_safe = max(u_min, min(u_max, u_cbf_max))
        is_intervened = True

    return u_safe, is_intervened, is_infeasible


def step_kinematics(
    s: float,
    v_ego: float,
    v_lead: float,
    u_applied: float,
    a_lead: float = 0.0,
    dt: float = 0.1,
    v_lead_next: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Forward Euler integration of longitudinal two-car kinematics.

    Follower velocity clipping ensures physical non-negativity (cannot drive backward).
    Lead velocity clipping ensures non-negativity.
    Longitudinal gap s is lower-bounded at 0 (collision point).

    If v_lead_next is supplied directly (e.g. from an exogenous recorded trajectory
    like OpenACC Vicolungo), it is used as the next lead speed without integrating a_lead.

    Returns:
        (s_next, v_ego_next, v_lead_next)
    """
    # Update follower velocity from applied control action
    v_ego_next = max(0.0, v_ego + u_applied * dt)

    # Lead vehicle velocity: either exact exogenous next speed or integrated from a_lead
    if v_lead_next is not None:
        v_lead_next_val = max(0.0, float(v_lead_next))
    else:
        v_lead_next_val = max(0.0, v_lead + a_lead * dt)

    # Forward Euler gap update based on relative velocity during the step
    s_next = max(0.0, s + (v_lead - v_ego) * dt)

    return s_next, v_ego_next, v_lead_next_val


def compute_cbf_reward(
    u_policy: float,
    u_cbf_max: float,
    u_safe: float,
    w_cbf1: float = 1.0,
    w_cbf2: float = 2.0,
    sigma_sq: float = 0.5,
) -> Tuple[float, float, float]:
    """Computes the dual-term barrier reward penalty from Yang et al. (Equations 22-23).

    r_cbf = r_cbf1 + r_cbf2
    where:
      r_cbf1 = w_cbf1 * min(u_cbf_max - u_policy, 0.0)
          (Linear penalty on the extent of CBF constraint violation)
      r_cbf2 = w_cbf2 * [exp( - (u_policy - u_safe)^2 / sigma_sq ) - 1.0]
          (Saturated exponential penalty on action divergence, bounded in [-w_cbf2, 0])

    When u_policy satisfies CBF (u_policy <= u_cbf_max), both terms are 0.0.

    Returns:
        (r_cbf, r_cbf1, r_cbf2)
    """
    # Linear barrier breach term
    psi = u_cbf_max - u_policy
    r_cbf1 = w_cbf1 * min(0.0, psi)

    # Saturated exponential action divergence term
    diff_sq = (u_policy - u_safe) ** 2
    r_cbf2 = w_cbf2 * (math.exp(-diff_sq / sigma_sq) - 1.0)

    r_cbf = r_cbf1 + r_cbf2
    return r_cbf, r_cbf1, r_cbf2


def compute_nominal_reward(
    s: float,
    v_ego: float,
    v_lead: float,
    u_action: float,
    u_prev: float,
    dt: float = 0.1,
    T_target: float = 2.5,
    D_target: float = 18.0,
    v_target: float = 20.0,
    w_gap: float = 0.6,
    w_vel: float = 0.4,
    w_ctrl: float = 0.1,
    w_jerk: float = 0.1,
    r_prog_max: float = 0.3,
    r_crash: float = -50.0,
    d_scale: float = 15.0,
    v_scale: float = 10.0,
    u_max: float = 2.5,
    jerk_max: float = 5.0,
    T_min: float = 2.0,
    D_min: float = 15.0,
) -> Tuple[float, Dict[str, float]]:
    """Computes the bounded nominal task reward based on the action applied in this ablation.

    For Filter ON (Filter Only, Dual): u_action = u_safe
    For Filter OFF (Nominal, Reward Only): u_action = u_policy (bounded by actuator limits)

    Balances headway tracking, velocity alignment, passenger comfort, and
    forward progress while avoiding the zero-velocity stalling trap.

    Returns:
        r_nominal (float): Sum of nominal reward terms.
        components (dict): Breakdown of individual terms.
    """
    # Check terminal collision condition
    if s <= 0.0:
        components = {
            "r_gap": 0.0,
            "r_vel": 0.0,
            "r_ctrl": 0.0,
            "r_jerk": 0.0,
            "r_prog": 0.0,
            "r_crash": r_crash,
        }
        return r_crash, components

    # 1. Bounded headway tracking error: s*(v_ego) = T_target * v_ego + D_target
    s_target = T_target * v_ego + D_target
    gap_err = (s - s_target) / d_scale
    r_gap = -w_gap * math.tanh(gap_err * gap_err)

    # 2. Bounded velocity matching error
    vel_err = (v_ego - v_lead) / v_scale
    r_vel = -w_vel * math.tanh(vel_err * vel_err)

    # 3. Control effort penalty based on the applied action
    r_ctrl = -w_ctrl * (u_action / u_max) ** 2

    # 4. Comfort / jerk penalty based on the applied action change
    jerk = (u_action - u_prev) / dt
    r_jerk = -w_jerk * (jerk / jerk_max) ** 2

    # 5. Forward progress bonus (awarded only when in the safe set h >= 0)
    h_curr = compute_h(s, v_ego, T_min, D_min)
    if h_curr >= 0.0 and v_target > 0.0:
        r_prog = r_prog_max * min(1.0, v_ego / v_target)
    else:
        r_prog = 0.0

    r_nominal = r_gap + r_vel + r_ctrl + r_jerk + r_prog

    components = {
        "r_gap": r_gap,
        "r_vel": r_vel,
        "r_ctrl": r_ctrl,
        "r_jerk": r_jerk,
        "r_prog": r_prog,
        "r_crash": 0.0,
    }
    return r_nominal, components
