"""
cbf_controller.py - Modular Control Barrier Function (CBF) Safety Filter
Implements the exact CBF formulation from Gunter et al. (ICCPS 2025):
  h(x, e) = s - (T_MIN * v + D_MIN)
  u_CBF <= (K_CBF * h + delta_v) / T_MIN
  u_raw = min(u_nom, u_CBF)
  u_sat = clip(u_raw, A_MIN, A_MAX)
  da/dt <= JERK_MAX
"""

import numpy as np
from project_config import (
    DT,
    T_MIN,
    D_MIN,
    K_CBF,
    A_MAX,
    A_MIN,
    JERK_MAX,
)

def compute_h(gap: float, ego_speed: float, t_min: float = T_MIN, d_min: float = D_MIN) -> float:
    """Computes the paper's Control Barrier Function safety value h."""
    return gap - (t_min * ego_speed + d_min)

def compute_u_cbf(h: float, delta_v: float, k_cbf: float = K_CBF, t_min: float = T_MIN) -> float:
    """
    Computes the maximum safe acceleration command allowed by the CBF.
    delta_v = v_lead - v_ego
    """
    return (k_cbf * h + delta_v) / t_min

class CBFSafetyFilter:
    """
    Stateful CBF supervisor and actuator saturation filter.
    Maintains previous acceleration state for jerk limiting.
    """
    def __init__(
        self,
        dt: float = DT,
        t_min: float = T_MIN,
        d_min: float = D_MIN,
        k_cbf: float = K_CBF,
        a_min: float = A_MIN,
        a_max: float = A_MAX,
        jerk_max: float = JERK_MAX,
    ):
        self.dt = dt
        self.t_min = t_min
        self.d_min = d_min
        self.k_cbf = k_cbf
        self.a_min = a_min
        self.a_max = a_max
        self.jerk_max = jerk_max
        self.max_acc_change = jerk_max * dt
        self.prev_acc = 0.0

    def reset(self, initial_acc: float = 0.0):
        self.prev_acc = initial_acc

    def filter_control(
        self,
        u_nom: float,
        gap: float,
        ego_speed: float,
        lead_speed: float,
    ) -> dict:
        """
        Filters the nominal controller command through the CBF safety barrier
        and actuator physical limits.
        """
        # Relative speed: delta_v = v_lead - v_ego
        delta_v = lead_speed - ego_speed

        # 1. Compute barrier safety value
        h = compute_h(gap, ego_speed, self.t_min, self.d_min)

        # 2. Compute theoretical CBF acceleration constraint
        u_cbf = compute_u_cbf(h, delta_v, self.k_cbf, self.t_min)

        # 3. Detect actuator feasibility (demands more braking than physical maximum)
        infeasible = (u_cbf < self.a_min)

        # 4. CBF safety supervisor (min-filtering)
        u_raw = min(u_nom, u_cbf)

        # 5. Detect whether CBF intervened to restrict nominal command
        # (intervention occurs whenever CBF specifies a stricter upper bound)
        intervention = (u_cbf < u_nom)

        # 6. Physical actuator saturation [-4.0, +2.0] m/s^2
        u_sat = np.clip(u_raw, self.a_min, self.a_max)

        # 7. Jerk rate limiting (|da/dt| <= JERK_MAX)
        acc_change = np.clip(
            u_sat - self.prev_acc,
            -self.max_acc_change,
            self.max_acc_change
        )
        applied_acc = self.prev_acc + acc_change
        self.prev_acc = applied_acc

        return {
            "applied_acc": applied_acc,
            "u_nom": u_nom,
            "u_cbf": u_cbf,
            "u_raw": u_raw,
            "u_sat": u_sat,
            "h": h,
            "delta_v": delta_v,
            "intervention": intervention,
            "infeasible": infeasible,
        }

if __name__ == "__main__":
    print("Running CBF Controller Unit Tests...")
    cbf_filter = CBFSafetyFilter()

    # Test 1: Safe state, ego slower than lead -> u_cbf large positive, no intervention
    res1 = cbf_filter.filter_control(u_nom=1.0, gap=75.0, ego_speed=25.0, lead_speed=27.0)
    assert not res1["intervention"], "Test 1 Failed: CBF intervened when safe"
    assert res1["h"] > 0, "Test 1 Failed: h should be positive"
    print(f"Test 1 Passed: Safe state h={res1['h']:.1f} m, applied_acc={res1['applied_acc']:.2f} m/s^2")

    # Test 2: Unsafe state, close gap -> CBF intervenes
    cbf_filter.reset()
    res2 = cbf_filter.filter_control(u_nom=1.0, gap=30.0, ego_speed=25.0, lead_speed=25.0)
    assert res2["intervention"], "Test 2 Failed: CBF should intervene in close following"
    assert res2["h"] < 0, "Test 2 Failed: h should be negative"
    print(f"Test 2 Passed: Intervening state h={res2['h']:.1f} m, u_cbf={res2['u_cbf']:.2f} m/s^2, applied_acc={res2['applied_acc']:.2f} m/s^2")

    # Test 3: Infeasible CBF command (extreme closing at tiny gap)
    cbf_filter.reset()
    res3 = cbf_filter.filter_control(u_nom=0.0, gap=2.0, ego_speed=30.0, lead_speed=15.0)
    assert res3["infeasible"], "Test 3 Failed: Should flag infeasibility when u_cbf < -4 m/s^2"
    assert res3["u_cbf"] < -4.0, "Test 3 Failed: u_cbf should be < -4.0"
    print(f"Test 3 Passed: Infeasible state u_cbf={res3['u_cbf']:.2f} m/s^2 flagged correctly")

    print("All CBF unit tests passed successfully.")
