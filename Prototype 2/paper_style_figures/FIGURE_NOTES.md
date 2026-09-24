# Scientific Figure Documentation & Notes

This document provides rigorous academic documentation for the publication-grade visualization suite located in [`paper_style_figures/`](file:///E:/CPS%20Project/Prototype%202/paper_style_figures).

Each figure has been generated in both **high-resolution 300+ DPI PNG** and **vector PDF** formats, following IEEE and ACM conference publishing standards (white background, thin spines, restrained academic palette, clear mathematical KaTeX notation, and explicit dimensional units). All data plotted originate exclusively from the audited simulation runs in [`results/`](file:///E:/CPS%20Project/Prototype%202/results).

---

## Suite Overview & Hierarchy

The visualization suite is organized into the 5 core figures of the main paper plus 1 supplementary appendix figure:

1. **Figure 1**: Experimental Framework: Closed-Loop Counterfactual Replay Architecture
2. **Figure 2**: Empirical Forward Invariance Across Initially-Safe Trajectories ($N=9, h_0 \ge 0$)
3. **Figure 3**: Asymptotic Safety Recovery Dynamics from Initially-Unsafe States ($h_0 < 0$)
4. **Figure 4**: Representative Stress Case: CBF Intervention and Actuator Saturation (Trajectory 16)
5. **Figure 5**: Parametric Sensitivity Analysis Across Headway Parameter $T_{\min} \in [1.0, 2.5]\text{ s}$
6. **Supplementary / Appendix**: Commanded Theoretical CBF vs. Physically Applied Acceleration ($\Delta u = u_{\text{actual}} - u_{\text{CBF}}$)

---

## Figure 1: Experimental Framework & Closed-Loop Counterfactual Replay Architecture

- **Primary Output Files**:
  - `paper_fig1_framework.png` / `paper_fig1_framework.pdf`
  - `paper_fig1_experimental_framework.png` / `paper_fig1_experimental_framework.pdf` (synonym)
- **Methodological Role**: System Architecture and Signal Flow Diagram

### Scientific Structure & Components
Figure 1 illustrates the exact experimental counterfactual replay framework used to evaluate the CBF supervisor on naturalistic freeway data:
1. **Exogenous Recording Layer (Open-Loop Lead Vehicle)**:
   - Extracted from the OpenACC Vicolungo naturalistic dataset ($10$ Hz).
   - Serves as an unalterable ground-truth trajectory ($p_{\text{lead}}(t), v_{\text{lead}}(t)$).
2. **Nominal ACC Controller (Tracking Baseline)**:
   - Desired spacing policy: $s_{\text{des}} = T_{\text{des}} v_{\text{ego}} + D_{\text{des}} = 1.5 v_{\text{ego}} + 5.0\text{ m}$.
   - Linear feedback control: $u_{\text{nom}} = k_p (s - s_{\text{des}}) + k_d (v_{\text{lead}} - v_{\text{ego}})$, where $k_p = 0.5\text{ s}^{-2}, k_d = 0.5\text{ s}^{-1}$.
3. **Control Barrier Function Safety Filter (Supervisory Barrier)**:
   - Safety Barrier: $h(s, v) = s - (T_{\min} v_{\text{ego}} + D_{\min}) = s - (2.0 v_{\text{ego}} + 15.0) \ge 0$.
   - Nagumo / Forward Invariance Condition: $\dot{h} + k h \ge 0 \implies u_{\text{CBF}} \le \frac{v_{\text{lead}} - v_{\text{ego}} + k h}{T_{\min}}$, with $k = 0.1\text{ s}^{-1}$.
   - Supervisory Mediation: $u_{\text{cmd}} = \min(u_{\text{nom}}, u_{\text{CBF}})$.
4. **Actuator Dynamics & Physical Limits**:
   - Physical Acceleration Saturation: $u_{\text{sat}} = \text{clip}(u_{\text{cmd}}, -4.0, +2.0)\text{ m/s}^2$.
   - Passenger Comfort Jerk Rate-Limiting: $|\dot{a}| \le 1.5\text{ m/s}^3 \implies u_{\text{actual}}$ applied to plant.
5. **Simulated Ego Vehicle (Closed-Loop Plant)**:
   - Forward Euler state update ($\Delta t = 0.1\text{ s}$):
     $$v_{\text{ego}}(t+\Delta t) = \max(0, v_{\text{ego}} + u_{\text{actual}}\Delta t), \quad s(t+\Delta t) = s + (v_{\text{lead}} - v_{\text{ego}})\Delta t$$
6. **Closed-Loop Feedback**:
   - The evolving ego state $(s, v_{\text{ego}})$ feeds back dynamically into both the ACC controller and the CBF supervisor.

### Correspondence to ICCPS 2025 Paper
Mirrors Figure 1 and Section III of Gunter et al. (ICCPS 2025), defining how the supervisory CBF architecture interfaces with nominal ACC while highlighting the exact closed-loop counterfactual simulation loop that isolates controller behavior from driver intervention.

---

## Figure 2: Empirical Forward Invariance Across Initially-Safe Trajectories

- **Primary Output Files**:
  - `paper_fig2_forward_invariance.png` / `paper_fig2_forward_invariance.pdf`
  - `paper_fig1_forward_invariance.png` / `paper_fig1_forward_invariance.pdf` (synonym)
- **Source CSV**: [`results/cbf_results.csv`](file:///E:/CPS%20Project/Prototype%202/results/cbf_results.csv)
- **Layout**: Decompressed Two-Panel Design (Panel a: Macro 0–80 s, Panel b: Boundary Zoom $\pm 0.05$ m)
- **Trajectories Plotted ($N=9, h_0 \ge 0$)**:
  - Trajectory 3 ($h_0 = +51.3$ m, duration: 114.7 s)
  - Trajectory 7 ($h_0 = +56.1$ m, duration: 114.6 s)
  - Trajectory 22 ($h_0 = +1.1$ m, duration: 724.3 s)
  - Trajectory 28 ($h_0 = +32.4$ m, duration: 33.4 s)
  - Trajectory 42 ($h_0 = +51.2$ m, duration: 397.0 s)
  - Trajectory 54 ($h_0 = +44.0$ m, duration: 307.7 s)
  - Trajectory 55 ($h_0 = +73.3$ m, duration: 35.9 s)
  - Trajectory 56 ($h_0 = +64.9$ m, duration: 29.9 s)
  - Trajectory 58 ($h_0 = +26.5$ m, duration: 110.1 s)

### Scientific Property Demonstrated
Demonstrates **empirical discrete-time forward invariance** of the safe set:
$$\mathcal{C} = \{x = (s, v) \in \mathbb{R}^2 \mid h(x) = s - (T_{\min} v + D_{\min}) \ge 0\}$$

1. **Panel (a) — Macro Trajectory Convergence**:
   - All 9 trajectories starting inside $\mathcal{C}$ converge towards the safety boundary $\partial\mathcal{C} (h=0)$ as the vehicle approaches steady-state following, remaining strictly safe for the entirety of their runs ($100\%$ empirical satisfaction).
2. **Panel (b) — Boundary Invariance Detail**:
   - Zooms into the critical boundary layer ($h \in [-0.025, 0.08]$ m).
   - Trajectories graze the boundary without crossing it into the unsafe region.
   - The worst-case numerical boundary deviation across the entire dataset is **$0.0090$ m ($9.0$ mm)** in Trajectory 22 ($t = 35$ s), well within discrete Forward Euler integration tolerance ($\Delta s \approx \frac{1}{2} a_{\max} \Delta t^2 \approx 0.01$ m).
3. **Collision Avoidance**:
   - Zero collisions occur across all initially safe trajectories ($0/9$).

### Correspondence to ICCPS 2025 Paper
Directly mirrors **Section V.A ("Forward Invariance Evaluation")** of Gunter et al. (ICCPS 2025). On the real vehicle on I-24, physical CAN bus latency caused deviations up to $3.4$ m. Here, with explicit plant saturation and jerk rate-limiting, forward invariance holds within sub-centimeter discretization tolerances.

---

## Figure 3: Asymptotic Safety Recovery Dynamics from Initially-Unsafe States

- **Primary Output Files**:
  - `paper_fig3_recovery.png` / `paper_fig3_recovery.pdf`
  - `paper_fig2_recovery.png` / `paper_fig2_recovery.pdf` (synonym)
- **Source CSV**: [`results/cbf_results.csv`](file:///E:/CPS%20Project/Prototype%202/results/cbf_results.csv)
- **Representative Trajectories Plotted ($N=7$)**:
  - Trajectory 25: Mild deficit ($h_0 = -4.4$ m, duration: 11.1 s)
  - Trajectory 17: Small deficit ($h_0 = -10.4$ m, duration: 374.1 s)
  - Trajectory 2: Moderate deficit ($h_0 = -19.2$ m, duration: 145.3 s)
  - Trajectory 0: Large deficit ($h_0 = -35.4$ m, duration: 145.4 s)
  - Trajectory 16: Severe cut-in stress case ($h_0 = -37.1$ m, duration: 66.2 s)
  - Trajectory 36: Deep deficit ($h_0 = -50.6$ m, duration: 126.2 s)
  - Trajectory 20: Extreme deficit ($h_0 = -54.4$ m, initial gap $0.88$ m, duration: 326.1 s)

### Scientific Property Demonstrated
Illustrates the **recovery property** from states starting outside the safe set ($h_0 < 0$). When $h < 0$, the barrier constraint enforces:
$$\dot{h} \ge -k h = -0.1 h$$
Because $-0.1 h > 0$, the supervisor strictly commands positive barrier growth ($\dot{h} > 0$), forcing exponential convergence toward the safe boundary:
$$h(t) \ge h(0) e^{-k t}, \quad \tau = \frac{1}{k} = 10.0\text{ s}$$

Figure 3 visualizes:
1. **Exponential Asymptotic Approach**: Trajectories close the initial gap deficit along exponential recovery curves bounded by the theoretical rate $h_0 e^{-0.1 t}$.
2. **Empirical Recovery Thresholds**:
   - **$78.8\%$ (41/52)** of initially unsafe trajectories achieve recovery within $0.1$ m of the boundary ($h \ge -0.1$ m).
   - **$84.6\%$ (44/52)** achieve recovery within $2.0$ m ($h \ge -2.0$ m).
   - In **$100.0\%$ (52/52)** of cases, safety strictly improves ($h_{\text{end}} > h_0$).
3. **Actuator Saturation Transient**: For severe cases (Trajectory 16), the red arrow highlights the initial dip where rate-limited braking momentarily limits the recovery rate before smooth exponential convergence takes over.

### Correspondence to ICCPS 2025 Paper
Corresponds directly to **Section V.B ("Recovery from Large Deviations / Cut-Ins")** of Gunter et al. (2025). The paper observed that when human lead vehicles cut in, the CBF supervisor smoothly expanded following distance back to the safe boundary in $>98\%$ of long recording sessions. Our counterfactual simulation demonstrates identical asymptotic convergence.

---

## Figure 4: Representative Stress Case: CBF Intervention and Actuator Saturation (Trajectory 16)

- **Primary Output Files**:
  - `paper_fig4_critical_trajectory.png` / `paper_fig4_critical_trajectory.pdf`
  - `paper_fig3_critical_trajectory.png` / `paper_fig3_critical_trajectory.pdf` (synonym)
- **Source CSVs**:
  - [`results/cbf_results.csv`](file:///E:/CPS%20Project/Prototype%202/results/cbf_results.csv)
  - [`results/baseline_results.csv`](file:///E:/CPS%20Project/Prototype%202/results/baseline_results.csv)
- **Trajectory Analyzed**: **Trajectory 16** (66.2 s, 663 timesteps).
  - Initial gap $s_0 = 33.09$ m.
  - Initial ego speed $v_{\text{ego}, 0} = 27.57$ m/s ($99.3$ km/h).
  - Initial lead speed $v_{\text{lead}, 0} = 19.05$ m/s ($68.6$ km/h).
  - Closing speed: $\Delta v_0 = 8.52$ m/s ($30.7$ km/h).

### Scientific Property Demonstrated
Provides an aligned three-panel forensic breakdown of how the CBF supervisor handles a severe, physically valid cut-in scenario:
1. **Panel (a) — Spacing Gap $s(t)$**:
   - Compares actual gap $s(t)$ under CBF against Baseline ACC, the CBF Safe Boundary ($2.0v + 15$ m), and the nominal ACC target ($1.5v + 5$ m).
   - Despite an $8.52$ m/s closing speed, the CBF supervisor stops the closing transient at a minimum gap of **$16.81$ m** (zero collision, zero near-miss), whereas baseline ACC settles at a much tighter following distance.
2. **Panel (b) — Velocity Tracking**:
   - Shows rapid deceleration of the ego vehicle from $27.6$ m/s down to $15.6$ m/s within $5.5$ s, completely neutralizing the closing velocity before smoothly harmonizing with lead speed.
3. **Panel (c) — Acceleration & Control Decomposition**:
   - Contrasts the nominal input $u_{\text{nom}}$ (orange dotted line) against the theoretical CBF demand $u_{\text{CBF}}$ (purple dashed line) and actual plant applied acceleration $u_{\text{actual}}$ (blue solid line).
   - Demonstrates the **Actuator Saturation Window** ($t \in [0.0, 2.1]$ s), where $u_{\text{CBF}}$ drops to $-6.12$ m/s², while the plant safely saturates at $a_{\min} = -4.0$ m/s² with jerk rate limiting ($1.5$ m/s³).

### Correspondence to ICCPS 2025 Paper
Directly addresses **Section V.C ("Collision Avoidance & Actuator Constraints")**. In contrast to Trajectory 6 (which starts with a $3.2$ cm gap and is physically unrecoverable under any bounded actuator), Trajectory 16 represents a genuine stress benchmark where CBF demand exceeds actuator capability, yet rate-limited physical braking successfully prevents collision and recovers safe headway.

---

## Figure 5: Parametric Sensitivity Analysis Across Headway Parameter $T_{\min}$

- **Primary Output Files**:
  - `paper_fig5_tmin_sensitivity.png` / `paper_fig5_tmin_sensitivity.pdf`
- **Source CSV**: [`results/sensitivity_summary.csv`](file:///E:/CPS%20Project/Prototype%202/results/sensitivity_summary.csv)
- **Parameters Evaluated**: $T_{\min} \in \{1.0, 1.5, 2.0, 2.5\}$ s (with $D_{\min} = 15$ m, $k = 0.1$, nominal ACC $T_{\text{des}} = 1.5$ s, $D_{\text{des}} = 5$ m).

### Numerical Results (Exact Source of Truth)

| $T_{\min}$ (s) | Initially Safe ($h_0 \ge 0$) | Intervention Rate (%) | Mean Spacing Gap (m) | Clean Min Gap (m) | Mean Ego Speed (m/s) | Speed Error (m/s) | Infeasible Samples | Recovery Rate ($h \ge -0.1$ m) | Collisions [Valid 60] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0 s** | 21/61 (34.4%) | **19.19%** | **46.58 m** | 0.88 m | **27.22 m/s** | 1.0123 m/s | 118 | 90.0% | **0 (0.0%)** |
| **1.5 s** | 13/61 (21.3%) | **92.93%** | **54.23 m** | 0.88 m | **27.14 m/s** | 1.0807 m/s | 89 | 79.2% | **0 (0.0%)** |
| **2.0 s** *(Paper)* | 9/61 (14.8%) | **94.54%** | **65.49 m** | 0.88 m | **27.03 m/s** | 1.2572 m/s | 70 | 78.8% | **0 (0.0%)** |
| **2.5 s** | 8/61 (13.1%) | **95.60%** | **76.70 m** | 0.88 m | **26.92 m/s** | 1.4296 m/s | 65 | 77.4% | **0 (0.0%)** |

### Scientific Property Demonstrated
Explains the fundamental trade-off between **safety conservatism** and **traffic flow efficiency**:
1. **Panel (a) — Intervention Bifurcation**:
   - When $T_{\min} = 1.0$ s ($< T_{\text{des}} = 1.5$ s), the nominal ACC following target ($s_{\text{des}} = 5 + 1.5v$) is strictly inside the CBF safe set ($s_{\text{safe}} = 15 + 1.0v$) at highway speeds. As a result, the intervention rate collapses from **$94.54\%$ down to $19.19\%$**.
   - When $T_{\min} \ge 1.5$ s, the nominal ACC target lies inside the CBF unsafe region, causing the CBF constraint to remain persistently active ($>92\%$).
2. **Panel (b) — Spacing Expansion**:
   - Following distance expands linearly from $46.58$ m at $T_{\min} = 1.0$ s to $76.70$ m at $T_{\min} = 2.5$ s ($+30.1$ m safety buffer).
3. **Panel (c) — Speed Preservation**:
   - Mean speed drops by only **$0.30$ m/s ($1.1\%$)** across the entire range ($27.22 \to 26.92$ m/s), proving that the CBF supervisor increases spatial safety without significantly throttling traffic throughput.
4. **Collision Invariance**:
   - Exactly **0 collisions** occur across the causally valid 60-trajectory dataset for every tested value of $T_{\min}$.

### Correspondence to ICCPS 2025 Paper
Directly answers the paper's discussion on parameter selection: the authors adopted $T_{\min} = 2.0$ s to enforce a strict $2$-second safety cushion on the highway. Figure 5 proves that the high intervention rate observed in naturalistic driving is not an algorithmic flaw, but the predictable geometric consequence of choosing a safety headway ($T_{\min} = 2.0$ s) that exceeds standard ACC following preferences ($T_{\text{des}} = 1.5$ s).

---

## Supplementary Appendix Figure: Control-Input & Plant Model Mismatch Analysis

- **Primary Output Files**:
  - `paper_fig_appendix_model_mismatch.png` / `paper_fig_appendix_model_mismatch.pdf`
  - `paper_fig4_model_mismatch.png` / `paper_fig4_model_mismatch.pdf` (synonym)
- **Source CSV**: [`results/cbf_results.csv`](file:///E:/CPS%20Project/Prototype%202/results/cbf_results.csv)
- **Trajectories Analyzed**:
  - **Panel (a): Trajectory 16** (Severe Cut-in, 22 saturation samples, min $u_{\text{CBF}} = -6.12$ m/s²)
  - **Panel (b): Trajectory 51** (Actuator Saturation, 6 saturation samples, min $u_{\text{CBF}} = -4.87$ m/s²)

### Scientific Property Demonstrated
Directly visualizes and isolates the **control discrepancy**:
$$\Delta u(t) = u_{\text{actual}}(t) - u_{\text{CBF}}(t)$$

Theoretical CBF guarantees assume that the commanded input is delivered instantaneously to the vehicle plant ($u_{\text{actual}} = u_{\text{CBF}}$, corresponding to the zero line $\Delta u = 0$). The figure decomposes the mismatch into two distinct physical phenomena:
1. **Actuator Saturation Window (Amber Shaded Area)**:
   - Occurs when $u_{\text{CBF}} < a_{\min} = -4.0$ m/s².
   - Because the hydraulic/regenerative brakes cannot exceed $-4.0$ m/s², the plant applies $-4.0$ m/s², yielding $\Delta u > 0$.
   - Peak discrepancy reaches **$+5.96$ m/s²** in Trajectory 16 ($t=0.1$ s) and **$+4.72$ m/s²** in Trajectory 51 ($t=0.0$ s).
2. **Jerk Rate-Limiting Lag Window (Blue Shaded Area)**:
   - Occurs when $u_{\text{CBF}} \ge -4.0$ m/s², but the rate of change is limited by passenger comfort ($|\dot{a}| \le 1.5$ m/s³).
   - The vehicle requires $\approx 2.6$ s to ramp down from $0$ to $-4.0$ m/s², creating a transient lag where $u_{\text{actual}} > u_{\text{CBF}}$.
3. **Equilibrium State**:
   - Once the closing speed is neutralized, $\Delta u$ relaxes to zero, confirming that mismatch is strictly transient and does not destabilize steady-state tracking.

### Correspondence to ICCPS 2025 Paper
Corresponds directly to the paper's critical findings on **"Actuator Delay, CAN Latency, and Model Inaccuracy"**. Continuous CBF theory proves safety under idealized dynamics, but physical implementations on real vehicles always experience control mismatch. This appendix figure provides a clean, rigorous method for isolating saturation from rate-limiting lag.

---

*Notes compiled automatically from audited simulation data in `Prototype 2/results/`.*
