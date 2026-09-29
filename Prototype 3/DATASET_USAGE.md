# Dataset Usage & Data Flow Specification: Prototype 3

**Document Version:** 1.1  
**Context:** CBF-RL Autonomous Car-Following  
**Dataset Reference:** OpenACC Vicolungo Naturalistic Driving Dataset (`Prototype 3/Vicolungo.csv` / `Prototype 2/Vicolungo.csv`)  
**Campaign Provenance:** OpenACC field experiment conducted on 26–28 February 2019 on the A4 highway (Turin–Milan, near Vicolungo, Italy).

---

## 1. Dataset Source & Context
The dataset used in this project is the **OpenACC Vicolungo** naturalistic car-following dataset collected on 26–28 February 2019 on Italian highways. It contains 10 Hz time-series data from instrumented lead and follower vehicles driven by human drivers under real-world traffic conditions.

The dataset contains 61 recorded car-following trajectories. Following our earlier research audits, **Trajectory 6 is excluded** as an initially overlapping / physically unrecoverable case ($s_0 \approx 0.032\,\text{m}$ at highway closing speeds), leaving **60 causally valid trajectories**.


---

## 2. Variables Used vs. NOT Used

| Variable Name in CSV | Description in Dataset | Used in Prototype 3? | Exact Role in Pipeline |
| :--- | :--- | :---: | :--- |
| `Trajectory_ID` | Trajectory identification index (1 to 61) | **YES** | Scenario indexing and selection |
| `Time_Index` | Step timestamp (10 Hz, $\Delta t = 0.1\,\text{s}$) | **YES** | Temporal alignment |
| `Speed_LV` | Lead Vehicle Speed ($\text{m/s}$) | **YES** | **Exact exogenous lead speed:** $v_{\text{lead}}[k] = \text{Speed\_LV}[k]$ |
| `Acc_LV` | Lead Vehicle Acceleration ($\text{m/s}^2$) | **NO** | Discarded. Lead velocity is drawn directly from `Speed_LV` without integration |
| `Pos_LV` | Lead Vehicle Position ($\text{m}$) | **NO** | Discarded. Kinematics use relative gap integration |
| `Spatial_Gap` | Bumper-to-bumper distance ($\text{m}$) | **INITIAL ONLY** | Used **only at $t=0$** to set $s_0 = \text{Spatial\_Gap}[0]$. Discarded for $t > 0$ |
| `Speed_FAV` | Human Follower Speed ($\text{m/s}$) | **INITIAL ONLY** | Used **only at $t=0$** to set $v_{\text{ego}, 0} = \text{Speed\_FAV}[0]$. Discarded for $t > 0$ |
| `Acc_FAV` | Human Follower Acceleration ($\text{m/s}^2$) | **NO** | Discarded completely |
| `Pos_FAV` | Human Follower Position ($\text{m}$) | **NO** | Discarded completely |
| `Spatial_Headway` | Front-bumper to front-bumper headway | **NO** | Discarded. Physical gap `Spatial_Gap` is used |

---

## 3. Training Usage: Zero (Purely Synthetic)

> [!IMPORTANT]
> **Vicolungo.csv is NOT used during PPO training.**

- In `pilot_train.py`, the training function calls `CarFollowingEnv(mode="synthetic")`.
- The PPO policy is trained exclusively on a procedural scenario generator that synthesizes randomized initial headways ($s_0 \sim \mathcal{U}(20, 60)\,\text{m}$), speeds ($v_0 \sim \mathcal{U}(12, 26)\,\text{m/s}$), and diverse lead maneuvers (cruising noise, braking steps, sinusoidal traffic waves, and close cut-ins).
- **Scientific Rationale:** Training on synthetic scenarios ensures the policy learns generalizable car-following principles rather than memorizing specific road geometries or braking timestamps from historical recordings.

---

## 4. Evaluation Usage: Counterfactual Real-World Replay

- Vicolungo is used exclusively as a **real-world scenario/disturbance source for testing**.
- The evaluation tests whether a policy trained on synthetic primitives can successfully control a simulated vehicle when confronted with real human driving profiles recorded on highways.
- This is a **counterfactual closed-loop simulation**:
  1. The historical human lead driver's speed sequence is replayed exactly: $v_{\text{lead}}[k] = \text{Speed\_LV}[k]$.
  2. The historical human follower is **replaced** by our RL agent.
  3. The follower's acceleration, velocity, and longitudinal headway evolve dynamically in response to the RL agent's control actions.

---

## 5. Architectural Data Flow Diagram

```text
========================================================================================
PPO TRAINING PIPELINE (100% Synthetic Domain)
========================================================================================
Synthetic Procedural Generator
   │ (Randomized s0, v0, cruising / braking / wave maneuvers)
   ▼
CarFollowingEnv(mode="synthetic")
   │
   ├── Observation: o_k = [norm_s, norm_v, norm_dv, norm_h]
   ▼
PPO Policy Network (pi_theta)
   │
   ├── Action Proposal: a_k in [-1, 1]  --> u_policy in [-4.0, +2.5] m/s²
   ▼
Closed-Form CBF Filter (if enabled)
   │
   ├── u_safe = clip(min(u_policy, u_cbf_max), -4.0, 2.5)
   ▼
Environment Transition:
   v_ego[k+1] = max(0, v_ego[k] + u_applied * dt)
   s[k+1]     = max(0, s[k] + (v_lead[k] - v_ego[k]) * dt)
   │
   ▼
PPO Rollout Buffer Stores:
   [o_k, a_k (u_policy), log_pi(a_k), r_k, o_k+1]   <-- NEVER stores u_safe as action!


========================================================================================
VICOLUNGO EVALUATION PIPELINE (Counterfactual Closed-Loop Benchmark)
========================================================================================
Vicolungo.csv (Historical Italian Highway Dataset)
   │
   ├── At t=0 ONLY:
   │     s_0       = Spatial_Gap[0]
   │     v_ego,0   = Speed_FAV[0]
   │     v_lead,0  = Speed_LV[0]
   │
   ├── For all t > 0:
   │     Speed_FAV is DISCARDED (Historical follower is NOT replayed)
   │     Acc_LV    is DISCARDED (No numerical integration of lead acceleration)
   │     Speed_LV[k+1] is read directly as the EXOGENOUS next lead vehicle speed
   │
   ▼
CarFollowingEnv(mode="vicolungo", traj_id=T)
   │
   ├── Observation: o_k = [norm_s_sim, norm_v_sim, norm_dv_sim, norm_h_sim]
   ▼
Trained PPO Policy (deterministic evaluation: a_k = mu(o_k))
   │
   ├── u_policy = action_to_acc(a_k)
   ▼
CBF Supervisor:
   Mode A (Filter ON):  u_applied = u_safe
   Mode B (Filter OFF): u_applied = u_policy  (Key test of policy internalization!)
   ▼
Simulated Vehicle Kinematics:
   v_ego[k+1]  = max(0, v_ego[k] + u_applied * dt)
   s[k+1]      = max(0, s[k] + (Speed_LV[k] - v_ego[k]) * dt)
   v_lead[k+1] = Speed_LV[k+1]   (Exact recorded dataset value)
   │
   ▼
Logged Research Metrics:
   Headway s(t), Safety Margin h(t), Intervention Rate, Minimum Headway, Collisions
========================================================================================
```

---

## 6. Why This Constitutes Genuinely Closed-Loop Evaluation

- **Open-Loop Replay (What we DO NOT do):** Replaying the recorded follower velocity $\text{Speed\_FAV}[t]$ from the CSV and merely evaluating whether $h(t) \ge 0$ would be an open-loop replay.
- **Closed-Loop Follower Simulation (What we DO):**  
  At every step $k \ge 1$, the follower position $x_{\text{ego}}[k]$ and velocity $v_{\text{ego}}[k]$ depend strictly on the historical accumulation of actions $u_0, u_1, \dots, u_{k-1}$ chosen by the RL policy:
  $$v_{\text{ego}}[k] = v_{\text{ego}}[0] + \Delta t \sum_{j=0}^{k-1} u_{\text{applied}}[j]$$
  $$s[k] = s[0] + \Delta t \sum_{j=0}^{k-1} \left(\text{Speed\_LV}[j] - v_{\text{ego}}[j]\right)$$
  If the RL agent commands excessive acceleration, the simulated vehicle physically approaches the lead car and crashes ($s \le 0$). If it commands hard braking, the gap expands. The feedback loop is fully active.

---

## 7. Methodological Limitations & Assumptions

1. **Non-Reactive (Exogenous) Lead Vehicle:**  
   The lead vehicle follows its historically recorded speed profile $\text{Speed\_LV}[k]$ regardless of what the follower vehicle does. In real-world driving, if a follower tailgates aggressively, human lead drivers might adjust speed or change lanes. In this counterfactual replay, the lead driver does not react to the follower.
2. **Initial Condition Reality on Real Highways:**  
   In the raw Vicolungo dataset, real human drivers frequently follow at headways smaller than the conservative ICCPS 2025 research barrier ($T_{\min} = 2.0\,\text{s}, D_{\min} = 15.0\,\text{m}$). Consequently, **51 out of 60 trajectories begin with $h_0 < 0$ in the dataset itself**. The evaluation of those trajectories represents a **recovery challenge**, whereas only the 9 initially safe trajectories test **forward invariance**.
