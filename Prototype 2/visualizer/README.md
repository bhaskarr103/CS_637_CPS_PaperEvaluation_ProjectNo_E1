# Prototype 2: Autonomous CBF Car-Following Visualizer

**Context:** Research Evaluation of Gunter, Nice, Bunting, Sprinkle, Work (ICCPS 2025)  
*"Can Control Barrier Functions Keep Automated Vehicles Safe in Live Freeway Traffic?"*  
**Location:** [`E:\CPS Project\Prototype 2\visualizer\index.html`](file:///E:/CPS%20Project/Prototype%202/visualizer/index.html)  
**Launcher:** [`E:\CPS Project\Prototype 2\launch_visualizer.py`](file:///E:/CPS%20Project/Prototype%202/launch_visualizer.py)  

---

## 1. Simulation Architecture & Core Research Purpose

This visualizer provides a lightweight, presentation-quality 2.5D autonomous vehicle simulation demonstrating the core research contribution of **Prototype 2**:
```text
        LEAD VEHICLE (Exogenous Highway Disturbance)
             ↓
        AUTONOMOUS VEHICLE (Follower)
             ↓
        NOMINAL ACC CONTROLLER (Time-Gap Spacing Policy)
             ↓
        CONTROL BARRIER FUNCTION (CBF) SUPERVISORY FILTER
             ↓
        ACTUATOR & JERK CONSTRAINTS ([-4.0, +2.0] m/s², 1.5 m/s³)
             ↓
        VEHICLE KINEMATICS (Forward Euler @ 10 Hz)
```

The user can directly watch:
1. **What the nominal ACC wants to do** (accelerating when gap expands, braking when closing).
2. **What the CBF constraint calculates** ($u_{\text{CBF}} \le (\Delta v + k_{\text{cbf}} h)/T_{\min}$).
3. **When the CBF intervenes** ($u_{\text{CBF}} < u_{\text{nom}}$).
4. **The effect of actuator saturation and jerk limitation** ($A_{\min} = -4.0\,\text{m/s}^2, |da/dt| \le 1.5\,\text{m/s}^3$).
5. **A direct side-by-side comparison:** Baseline ACC vs. ACC + CBF operating behind the identical lead vehicle from identical starting states!

---

## 2. Reused Mathematical Equations & Parameters

The visualizer enforces the **exact numerical parameters and equations** from Prototype 2:

| Parameter / Concept | Exact Value in Prototype 2 | Source File Reference |
| :--- | :--- | :--- |
| **Time-step ($\Delta t$)** | $0.1\,\text{s}$ (10 Hz) | `project_config.py` (line 40) |
| **Desired Standstill Gap ($D_{\text{des}}$)** | $5.0\,\text{m}$ | `project_config.py` (line 45) |
| **Desired Time Headway ($T_{\text{des}}$)** | $1.5\,\text{s}$ | `project_config.py` (line 46) |
| **ACC Gap Gain ($K_{\text{gap}}$)** | $0.20\,\text{s}^{-2}$ | `project_config.py` (line 47) |
| **ACC Relative Speed Gain ($K_{\text{rel}}$)** | $0.80\,\text{s}^{-1}$ | `project_config.py` (line 48) |
| **CBF Minimum Safe Standstill ($D_{\min}$)** | $15.0\,\text{m}$ | `project_config.py` (line 56) |
| **CBF Minimum Safe Headway ($T_{\min}$)** | $2.0\,\text{s}$ | `project_config.py` (line 55) |
| **CBF Class-$\mathcal{K}$ Decay ($k_{\text{cbf}}$)** | $0.1\,\text{s}^{-1}$ | `project_config.py` (line 57) |
| **Max Acceleration ($A_{\max}$)** | $+2.0\,\text{m/s}^2$ | `project_config.py` (line 62) |
| **Max Braking Deceleration ($A_{\min}$)** | $-4.0\,\text{m/s}^2$ | `project_config.py` (line 63) |
| **Jerk Rate Limit ($\text{Jerk}_{\max}$)** | $1.5\,\text{m/s}^3$ | `project_config.py` (line 64) |

### Control Equations:
1. **Barrier Safety Function:**
   $$h(s, v_{\text{ego}}) = s - (T_{\min} \cdot v_{\text{ego}} + D_{\min})$$
2. **CBF Acceleration Upper Bound:**
   $$u_{\text{CBF}} \le \frac{\Delta v + k_{\text{cbf}} \cdot h}{T_{\min}}, \quad \Delta v = v_{\text{lead}} - v_{\text{ego}}$$
3. **Nominal ACC Spacing Law:**
   $$s_{\text{des}} = D_{\text{des}} + T_{\text{des}} \cdot v_{\text{ego}}$$
   $$u_{\text{nom}} = K_{\text{gap}}(s - s_{\text{des}}) + K_{\text{rel}} \Delta v$$
4. **Mediation & Saturation:**
   $$u_{\text{raw}} = \min(u_{\text{nom}},\, u_{\text{CBF}})$$
   $$u_{\text{sat}} = \text{clip}(u_{\text{raw}},\, A_{\min},\, A_{\max})$$
5. **Jerk Rate Limiting:**
   $$\Delta a = \text{clip}(u_{\text{sat}} - a_{\text{prev}},\, -\text{Jerk}_{\max}\Delta t,\, +\text{Jerk}_{\max}\Delta t)$$
   $$a_{\text{applied}} = a_{\text{prev}} + \Delta a$$

---

## 3. Automated Numerical Cross-Validation (Python vs JS)

To ensure that the browser JavaScript controller is an exact mathematical mirror of the Python research code, we implemented [`cross_validate_controllers.py`](file:///E:/CPS%20Project/Prototype%202/visualizer/cross_validate_controllers.py):
- Evaluates 50 diverse states covering safe following, near boundary, severe closing speeds, saturation limits, and jerk-limiting transients.
- Compares `applied_acc`, `u_nom`, `u_cbf`, `h`, `intervention`, and `infeasible` between Python (`cbf_controller.py`) and JavaScript (`app.js`).
- **Result:** **50/50 test cases passed with exact agreement (maximum absolute difference $< 10^{-6}$).**

---

## 4. Simulation Modes & Scenarios

### Mode 1: `[SYNTHETIC DEMONSTRATION]`
- **Scenario 1 (Normal Car Following):** Lead cruises at ~80 km/h with subtle wave noise. Follower tracks desired spacing smoothly.
- **Scenario 2 (Lead Vehicle Acceleration):** Lead speeds up from 72 to 101 km/h at $+1.3\,\text{m/s}^2$. Follower accelerates smoothly without violating bounds.
- **Scenario 3 (Moderate Lead Deceleration):** Lead brakes steadily from 90 to 61 km/h at $-1.8\,\text{m/s}^2$. Demonstrates standard deceleration tracking.
- **Scenario 4 (Sudden Emergency Hard Braking):** Lead slams brakes at $-3.6\,\text{m/s}^2$ down to 5.2 m/s. The CBF immediately restricts acceleration and avoids collision.
- **Scenario 5 (Closing-Speed Stress Test):** Ego at 108 km/h closing rapidly on Lead at 65 km/h with a 35m gap. Shows jerk-limited deceleration lag and safe recovery.

### Mode 2: `[VICOLUNGO — RECORDED LEAD REPLAY]`
- **Trajectory 16 — Safety Recovery / Actuator-Limit Stress Case ($h_0 = -37.05\,\text{m} < 0$):**  
  Real recorded data from Italian A4 highway (26–28 Feb 2019). Lead is much slower ($19.05\,\text{m/s}$) than ego ($27.57\,\text{m/s}$) with $33.09\,\text{m}$ initial headway. Because $h_0 < 0$, the ego starts *outside* the safe set $\mathcal{C}$. Rapid closing speed triggers severe CBF intervention and actuator saturation ($u_{\text{CBF}} < -4.0\,\text{m/s}^2$). **This tests emergency recovery and physical saturation handling, NOT forward invariance.**
- **Trajectory 55 — Forward-Invariance Demonstration ($h_0 = +73.30\,\text{m} > 0$):**  
  Starts safely inside the safe set at $s_0 = 119.56\,\text{m}, v_{\text{ego}, 0} = 15.63\,\text{m/s}$. **Demonstrates textbook continuous preservation of forward invariance ($h(t) \ge 0$ for all $t$)** under authentic human lead speed oscillations.
- **Trajectory 1 — Highway Headway Deficit Recovery ($h_0 = -34.10\,\text{m} < 0$):**  
  Highway tailgating at 116 km/h with 45.4m gap. Tests safe recovery without collision.
- **Trajectory 20 — Ultra-Tight Spacing with Lead Opening ($s_0 = 0.88\,\text{m}$):**  
  Initial gap is only $0.88\,\text{m}$, but lead vehicle accelerates away faster ($\Delta v > 0$). Tests opening response.

---

## 5. Flagship View: Parallel Counterfactual Comparison Mode

The visualizer **defaults to Dual Comparison mode** so that the research contribution of Prototype 2 is immediately obvious within 20 seconds:
- **Parallel Counterfactual Setup:** Two independent, non-interacting parallel simulations run simultaneously:
  - **Track 1 (Upper Track, Amber AV):** Baseline ACC (Nominal Spacing Law without barrier filter).
  - **Track 2 (Lower Track, Blue AV):** ACC + CBF (Safety Barrier Filter Supervised).
- Both tracks execute behind the **exact same lead vehicle disturbance profile** from the **exact same initial condition** ($s_0, v_{\text{ego}, 0}$).
- **Direct Contrast:**
  - In Scenario 4 (Emergency Hard Braking): Baseline ACC delays braking until spacing error accumulates, resulting in severe bumper penetration (collision $s \le 0\,\text{m}$). Meanwhile, ACC+CBF preemptively intervenes as soon as the barrier boundary is approached, clamping deceleration and preventing collision.
  - Telemetry charts superimpose both runs with a moving vertical cursor synchronized to the 2.5D animation.

---

## 6. Defensible Scientific Framing & Boundary Conditions

To maintain strict scientific integrity, the simulation avoids overstated claims such as "CBF guarantees the car will never crash." Instead:

> **Core Scientific Principle:**  
> The Control Barrier Function (CBF) imposes a mathematically defined forward-invariance constraint on the nominal controller, subject to model fidelity, discrete-time sampling ($\Delta t = 0.1\,\text{s}$), and physical actuator limits ($a \in [-4.0, +2.0]\,\text{m/s}^2$, $|\dot{a}| \le 1.5\,\text{m/s}^3$).
>
> 1. **Forward Invariance Guarantee:** If and only if the initial state is inside the safe set ($h(0) \ge 0$, e.g. Trajectory 55) and the required supervisory acceleration remains within the actuator set $\mathcal{U}$, Nagumo's theorem ensures that $h(t) \ge 0$ for all $t \ge 0$.
> 2. **Infeasibility & Actuator Saturation:** If the vehicle starts outside the safe set ($h(0) < 0$, e.g. Trajectory 16) or the lead vehicle decelerates harder than the ego vehicle can physically match ($u_{\text{CBF}} < A_{\min} = -4.0\,\text{m/s}^2$), the CBF constraint becomes infeasible and the supervisor commands maximum saturation braking. Recovery depends on initial energy, not forward invariance.

---

## 7. How to Launch

Run from PowerShell:
```powershell
& "E:\CPS Project\venv\Scripts\python.exe" "E:\CPS Project\Prototype 2\launch_visualizer.py"
```
Or open directly in any web browser (zero setup, zero CORS dependencies):
```text
file:///E:/CPS Project/Prototype 2/visualizer/index.html
```
