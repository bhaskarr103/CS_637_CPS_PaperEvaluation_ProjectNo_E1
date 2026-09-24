# Final Research Report: Rigorous Evaluation of Control Barrier Functions (CBF) on Naturalistic Freeway Traffic (Vicolungo Dataset)

**Authors / Implementation**: Pair Programming Team (CPS Project)  
**Reference Paper**: *"Can Control Barrier Functions Keep Automated Vehicles Safe in Live Freeway Traffic?"*  
**Authors**: George Gunter, Matthew Nice, Matt Bunting, Jonathan Sprinkle, and Daniel B. Work (ICCPS 2025 - Best Paper Award)  
**Implementation Directory**: `Prototype 2/`  
**Audit Status**: Methodologically Audited & Verified  

---

## 1. Executive Summary

This research report presents a scientifically audited evaluation of the Control Barrier Function (CBF) safety supervisor operating on the **OpenACC Vicolungo naturalistic freeway dataset** (Ultra-AV repository, 158,018 samples, 61 trajectories at 10 Hz).

Following the tripartite evaluation framework of the original ICCPS 2025 paper, the assessment explicitly separates **Forward Invariance**, **Recovery**, and **Collision Avoidance**:
1. **Forward Invariance ($h_0 \ge 0$)**: Evaluated on initially safe trajectories ($N=9$). The CBF achieves **100% empirical forward-invariance satisfaction** in discrete-time simulation, with a maximum boundary deviation of **0.0090 m** (9.0 mm discrete 10 Hz sampling tolerance) and zero collisions.
2. **Recovery from Unsafe States ($h_0 < 0$)**: Evaluated on initially unsafe trajectories ($N=52$). The CBF recovers **78.8% of trajectories within 0.1 m** of the safe boundary ($h \ge -0.1$ m) and **84.6% within 2.0 m**. In **100.0% of cases**, safety strictly improved ($h_{\text{end}} > h_0$). Because linear barrier dynamics enforce $\dot{h} + 0.1 h \ge 0$, convergence is asymptotic ($h(t) \approx h_0 e^{-0.1 t}$, time constant $\tau = 10$ s); the remaining 8 trajectories were short recordings ($<12$ s) actively closing the safety deficit ($\dot{h} > 0$) when recording ended.
3. **Collision Avoidance (Two-Set Presentation)**:
   - **All 61 Trajectories**: 1 collision occurred, located exclusively in **Trajectory 6**.
   - **Causally Valid Evaluation Set ($N=60$)**: **0 collisions (0.0%)** and **0 near-misses ($s \le 0.5$ m)** under CBF supervision.

---

## 2. Quantitative Performance Comparison

| Metric | Baseline ACC | ACC + CBF Supervisor | Research Impact / Assessment |
| :--- | :---: | :---: | :--- |
| **Total Trajectories Evaluated** | 61 | 61 | Complete Vicolungo dataset (158,018 rows) |
| **Causally Valid Trajectories** | 60 | 60 | Excluding initial kinematic overlap (Traj 6) |
| **Collisions: Causally Valid Set ($N=60$)** | **0 (0.0%)** | **0 (0.0%)** | Zero collisions under both controllers |
| **Collisions: Complete Set ($N=61$)** | 1 (Traj 6) | 1 (Traj 6) | Collision in Trajectory 6 only |
| **Near-Misses ($0 < s \le 0.5$ m) [Valid Set]**| **0 (0.0%)** | **0 (0.0%)** | Trajectory 20 maintained safe $0.88$ m margin |
| **Empirical Forward Invariance ($h_0 \ge 0$)** | N/A | **100.0% (9/9)** | Maintained across all valid initial safe states |
| **Max Invariance Boundary Deviation** | N/A | **0.0090 m** | Sub-centimeter discrete sampling tolerance |
| **Recovery Rate to $h \ge -0.1$ m ($h_0 < 0$)** | N/A | **78.8% (41/52)** | Asymptotic exponential approach to boundary |
| **Recovery Rate to $h \ge -2.0$ m ($h_0 < 0$)** | N/A | **84.6% (44/52)** | Effective mitigation of close-following risk |
| **Safety Trajectory Improvement ($h_{\text{end}} > h_0$)**| N/A | **100.0% (52/52)**| Active closure of initial safety deficit |
| **Average Spacing (Trajectory-Averaged)** | 45.74 m | **65.49 m** | $+19.75$ m safer headway enforced by CBF |
| **Average Spacing (Sample-Averaged)** | 48.75 m | **72.24 m** | $+23.49$ m pooled headway expansion |
| **Average Ego Speed (Trajectory-Averaged)** | 27.25 m/s | **27.03 m/s** | Minimal speed loss ($-0.22$ m/s, $-0.8\%$) |
| **Speed Tracking Error (Trajectory-Averaged)** | 1.0531 m/s | **1.2572 m/s** | Slight increase due to safety-priority spacing |
| **Actuator Saturation Demands ($u_{\text{CBF}} < -4$)**| 0 | **70 samples** | Confined to Trajectories 6 (42), 16 (22), 51 (6) |
| **Actuator Tracking Discrepancy ($u_{\text{actual}} > u_{\text{CBF}}$)**| 0 | **525 samples (0.33%)**| 70 brake saturation + 455 jerk-limiting lag |
| **Peak Deceleration / Peak Jerk** | -4.0 m/s² / 1.5 m/s³ | -4.0 m/s² / 1.5 m/s³ | Physical actuator limits strictly enforced |

---

## 3. Methodological Audit & Detailed Findings

### A. The Root Cause of the 94.54% Intervention Rate
A central investigation was whether the $94.54\%$ intervention rate indicated an implementation defect. **Our audit confirms that this is not an error, but the direct consequence of two factors:**
1. **High Traffic Density in Vicolungo**: Real human drivers in Vicolungo operate at high speeds (mean $\approx 27.2$ m/s) with tight spacing (mean gap $= 37.25$ m). Under the paper's boundary:
   $$h(x, e) = s - (2.0 v + 15.0)$$
   At $27.2$ m/s, $h \ge 0$ demands a gap of at least $69.4$ m. Across the entire dataset, **$96.81\%$ of natural driving samples reside in the CBF unsafe set ($h < 0$)**.
2. **Incompatibility Between ACC Target and CBF Boundary**: The nominal ACC controller tracks $s_{\text{des}} = 5.0 + 1.5 v$. At $27$ m/s, ACC targets $45.5$ m, which is **$23.5$ m deep inside the CBF unsafe zone**. Whenever ACC approaches its equilibrium, the CBF supervisor observes $h = -23.5$ m and commands:
   $$u_{\text{CBF}} \le \frac{0.1(-23.5) + 0}{2.0} = -1.175\text{ m/s}^2$$
   Since $u_{\text{cmd}} = \min(u_{\text{nom}}, u_{\text{CBF}})$, the CBF intervenes continuously to override the nominal controller and expand the headway from $45.5$ m toward $69$ m.

#### Decomposed Intervention Metrics
| Intervention Metric | Trajectory-Averaged (Macro) | Sample-Averaged (Micro) | Physical Interpretation |
| :--- | :---: | :---: | :--- |
| **1. CBF Constraint Active ($u_{\text{CBF}} < u_{\text{nom}}$)** | **94.54%** | **99.28%** (156,881 samples) | CBF boundary imposes an active upper bound |
| **2. Saturated Override ($\text{clip}(u_{\text{CBF}}) < u_{\text{nom}}$)** | **93.74%** | **99.24%** (156,811 samples) | Excludes dual saturation at $-4.0$ m/s² |
| **3. Applied Acceleration Deviation ($|\Delta a| > 0.05$)** | **37.08%** | **31.23%** (49,349 samples) | Transient braking & expansion divergence only |
| **3b. Applied Acceleration Deviation ($|\Delta a| > 0.01$)** | **78.41%** | **77.14%** (121,894 samples) | Minor cruising acceleration adjustment |

---

### B. Forensic Analysis of Trajectory 6
Trajectory 6 was identified as the sole collision case in both Baseline ACC and ACC+CBF ($s_{\min} = -22.49$ m). A kinematic audit revealed:
- At $t=0$, initial gap $s_0 = 0.0323$ m ($3.2$ cm).
- Initial ego speed $v_{\text{ego}, 0} = 35.82$ m/s ($129$ km/h), while lead speed $v_{\text{lead}, 0} = 27.14$ m/s ($98$ km/h).
- Closing speed is $\Delta v = -8.68$ m/s.
- **Stopping Distance Calculation**: Halting before contact at $8.68$ m/s closing speed requires deceleration:
  $$a = \frac{\Delta v^2}{2 s_0} = \frac{75.367}{0.0645} = 1168.0\text{ m/s}^2 \approx 119.1\text{ g}$$
- Under physical braking limits ($-4.0$ m/s²) and jerk limiting ($1.5$ m/s³), maximum deceleration at step 1 ($0.1$ s) is only $-0.15$ m/s², resulting in an immediate negative gap of $-0.8359$ m at $t = 0.1$ s.
- **Academic Verdict**:
  > **Trajectory 6 is classified as an initially overlapping/physically unrecoverable case rather than a meaningful causal safety benchmark. Given the 3.2 cm initial gap and 8.68 m/s closing speed, collision cannot be prevented under the modeled actuator and jerk constraints. The available trajectory data alone do not establish the underlying cause of the anomalous initial condition.**

---

### C. Actuator Limits & Plant Tracking Discrepancy
The theoretical CBF condition $\dot{h} \ge -k h$ assumes the control input applied to the plant satisfies $u \le u_{\text{CBF}}$. In the physical simulation:
- In **525 samples (0.33% of timesteps)**, $u_{\text{actual}} > u_{\text{CBF}}$:
  - **70 samples** were caused by **actuator saturation** ($u_{\text{CBF}} < -4.0$ m/s²), confined to Trajectories 6 (42), 16 (22), and 51 (6).
  - **455 samples** were caused by **jerk rate-limiting lag** ($|\dot{a}| \le 1.5$ m/s³ prevented instantaneous steps down to $u_{\text{CBF}}$; mean lag discrepancy: $1.12$ m/s²).
- In Trajectories 16 and 51, despite temporary demand saturation, rate-limited braking safely guided the vehicles to minimum gaps of $16.81$ m and $23.39$ m with zero collisions.

---

### D. Paired Trajectory Classification (N=61)

| Category | Trajectory Count | Percentage | Defining Characteristics |
| :--- | :---: | :---: | :--- |
| **Safety Improvement (Headway Expanded)** | 30 | 49.2% | Minimum gap increased by $>0.5$ m without speed penalty |
| **No Meaningful Minimum Gap Change** | 25 | 41.0% | $|\Delta s_{\min}| \le 0.5$ m (initial gap bounded $s_{\min}$, but mean gap expanded $+15-25$ m) |
| **Safety Improvement with Speed Penalty** | 3 | 4.9% | $\Delta s_{\min} > 0.5$ m, but mean speed dropped $>0.5$ m/s (Trajs 8, 11, 53) |
| **Actuator Saturation with Safe Recovery** | 2 | 3.3% | Temporary $u_{\text{CBF}} < -4.0$ m/s²; safely averted collision (Trajs 16, 51) |
| **Initially Overlapping / Unrecoverable** | 1 | 1.6% | Physically inevitable collision at $t=0$ (Trajectory 6) |

---

## 4. Parametric Sensitivity Analysis ($T_{\min} \in [1.0, 2.5]$ s)

Holding all other parameters fixed ($D_{\min} = 15$ m, $k = 0.1$, ACC $T_{\text{des}} = 1.5$ s, $D_{\text{des}} = 5$ m, limits $[-4, 2]$ m/s², jerk $1.5$ m/s³), we evaluated the sensitivity across $T_{\min}$:

| $T_{\min}$ (s) | Initially Safe ($h_0 \ge 0$) | Intervention Rate (%) | Mean Gap (m) | Clean Min Gap (m) | Mean Speed (m/s) | Speed Error (m/s) | Infeasible Samples | Recovery Rate ($h \ge -0.1$m) | Collisions [Valid 60] |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1.0 s** | 21/61 (34.4%) | **19.19%** | 46.58 m | 0.88 m | 27.22 m/s | 1.0123 m/s | 118 | 90.0% | **0 (0.0%)** |
| **1.5 s** | 13/61 (21.3%) | **92.93%** | 54.23 m | 0.88 m | 27.14 m/s | 1.0807 m/s | 89 | 79.2% | **0 (0.0%)** |
| **2.0 s** *(Paper)* | 9/61 (14.8%) | **94.54%** | 65.49 m | 0.88 m | 27.03 m/s | 1.2572 m/s | 70 | 78.8% | **0 (0.0%)** |
| **2.5 s** | 8/61 (13.1%) | **95.60%** | 76.70 m | 0.88 m | 26.92 m/s | 1.4296 m/s | 65 | 77.4% | **0 (0.0%)** |

**Sensitivity Takeaways**:
- When $T_{\min} = 1.0$ s ($< T_{\text{des}} = 1.5$ s), the nominal ACC setpoint ($5 + 1.5v$) is strictly inside the CBF safe zone ($15 + 1.0v$). As a result, the intervention rate **plummets from $94.5\%$ down to $19.2\%$**.
- As soon as $T_{\min} \ge 1.5$ s, nominal ACC targets an equilibrium that CBF considers unsafe, driving intervention above $92\%$.
- Zero collisions occur in the causally valid set across all tested values of $T_{\min}$.

---

## 5. Comparison: Simulation vs. Original ICCPS 2025 Paper Experiment

| Dimension | Original Paper (Gunter et al., 2025) | Our Prototype 2 Pipeline |
| :--- | :--- | :--- |
| **Environment** | Live freeway driving on I-24 (Nashville, TN) | Replayed naturalistic freeway driving (Vicolungo, Italy) |
| **Duration / Scale** | 1.17 hours of driving data | 158,018 timesteps across 61 trajectories (~4.4 hours) |
| **Safety Function** | $h = s - (2.0 v + 15.0)$, $k = 0.1$ | **Identical** ($T_{\min}=2.0, D_{\min}=15.0, K_{\text{CBF}}=0.1$) |
| **Nominal Control** | CAN bus injection on stock AV / human driver | Realistic ACC ($T_{\text{des}}=1.5, D_{\text{des}}=5.0$) with jerk limiting |
| **Actuator Modeling** | Physical vehicle actuators with CAN lag | Explicit acceleration saturation $[-4, +2]$ m/s² and jerk ($1.5$ m/s³) |
| **Forward Invariance** | Not strictly satisfied in real car (max violation $3.4$ m) | **100% satisfied empirically** in simulation (max violation $0.009$ m) |
| **Recovery from Cut-ins** | Recovered in $>98\%$ of large deviations | **84.6% recovered** within $2.0$ m; asymptotic convergence verified |
| **Collisions** | 0 collisions observed | **0 collisions** across all 60 causally valid scenarios |

---

## 6. Artifacts & Generated Files

### CSV Results (`results/`)
- `trajectory_classification.csv`: Initial condition classification ($h_0$, safe vs unsafe, speeds).
- `paired_trajectory_classification.csv`: 5-category paired trajectory safety vs speed classification.
- `sensitivity_summary.csv`: Parametric sweep results across $T_{\min} \in [1.0, 2.5]$ s.
- `baseline_results.csv` & `baseline_summary.csv`: Baseline ACC simulation outputs.
- `cbf_results.csv` & `cbf_summary.csv`: ACC + CBF simulation outputs.
- `comparative_metrics.csv`: Combined head-to-head trajectory comparative metrics.
- `stress_case_analysis.csv`: Forensic analysis of Trajectories 6, 16, 20, 51, 55, 56.

### Publication Figures (`figures/`)
- `fig1_critical_trajectories.png`: Time-series plots for Trajectories 16 and 55.
- `fig2_forward_invariance.png`: Barrier evolution $h(t)$ for all 9 initially safe trajectories.
- `fig3_recovery_analysis.png`: Asymptotic recovery trajectories converging toward boundary.
- `fig4_trajectory6_forensics.png`: Forensic demonstration of kinematic impossibility in Trajectory 6.
- `fig5_metrics_distribution.png`: Comparative histograms of Minimum Gap, Mean Gap, and Braking.
- `fig6_safety_vs_nominal_tradeoff.png`: Scatter and boundary analysis explaining 94.54% intervention.
- `fig7_paired_trajectory_deltas.png`: Paired delta scatter of safety margin gain vs speed impact.
- `fig8_tmin_sensitivity.png`: Sensitivity curves of intervention rate and headway vs $T_{\min}$.

---
*Report automatically generated by `08_final_report.py`.*
