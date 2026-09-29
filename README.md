# CS 637 — CPS Research Project

## Control Barrier Functions for Safe Autonomous Car-Following

This repository contains the work carried out for the **CS 637 Cyber-Physical Systems (CPS)** research project.

The project investigates how **Control Barrier Functions (CBFs)** can be used to enforce safety in autonomous vehicle car-following, and how the same safety mechanism can be integrated with **Reinforcement Learning (RL)**.

The work progresses from a classical ACC + CBF controller to a CBF-aware reinforcement learning framework.

---

---

## Demonstration Video

This video demonstrates the CPS project work developed across the autonomous car-following prototypes, including the ACC + Control Barrier Function (CBF) controller, Vicolungo trajectory-based evaluation, and the CBF-RL simulation and visualization.

https://github.com/user-attachments/assets/569dad05-da21-440c-97c7-576d933bfe9a

---

## Project Overview

The project is organized into three main prototypes:

```text
Prototype 1
    ↓
Initial data and controller experiments

Prototype 2
    ↓
ACC + Control Barrier Function
    ↓
Real-world Vicolungo trajectory evaluation

Prototype 3
    ↓
CBF + Reinforcement Learning
    ↓
Nominal RL / Filter Only / Reward Only / Dual CBF-RL
    ↓
Vicolungo closed-loop evaluation
```

The overall research question is:

> Can Control Barrier Functions provide safety supervision for autonomous car-following, and can reinforcement learning learn to internalize this safety behavior?

---

# Repository Structure

```text
CS_637_CPS_PaperEvaluation_Project_No_E1/
│
├── Notes/
├── Papers/
├── Prototype 1/
│
├── Prototype 2/
│   ├── Vicolungo.csv
│   ├── project_config.py
│   ├── cbf_controller.py
│   ├── 02_baseline_acc.py
│   ├── 04_cbf_simulation.py
│   ├── 05_cbf_evaluation.py
│   ├── 06_stress_cases.py
│   ├── 07_paper_figures.py
│   ├── 08_final_report.py
│   ├── 09_sensitivity_analysis.py
│   ├── results/
│   ├── figures/
│   ├── paper_style_figures/
│   └── visualizer/
│
├── Prototype 3/
│   ├── cbf_core.py
│   ├── config.py
│   ├── env.py
│   ├── ppo.py
│   ├── pilot_train.py
│   ├── train_phase3.py
│   ├── test_phase1_math.py
│   ├── results/
│   └── visualizer.html
│
├── .gitignore
└── README.md
```

---

# Prototype 1

Prototype 1 contains the initial investigation and experimental groundwork for the project.

It includes:

- Dataset inspection
- Initial car-following analysis
- Initial controller experiments
- Vicolungo trajectory investigation
- Safety metrics
- Supporting research analysis

---

# Prototype 2 — ACC + CBF

Prototype 2 implements a classical autonomous car-following controller with a **Control Barrier Function safety layer**.

## Controller Architecture

```text
              Lead Vehicle
                   │
                   │ v_lead
                   ▼
          Relative-State Calculation
                   │
          ┌────────┴────────┐
          │                 │
          ▼                 ▼
     Nominal ACC           CBF
          │                 │
       u_nom           u_CBF_max
          │                 │
          └────────┬────────┘
                   ▼
             Safety Filter
                   │
             u = min(...)
                   │
                   ▼
          Actuator Constraints
                   │
                   ▼
          Follower Vehicle
                   │
                   └────── feedback
```

## Barrier Function

The safety function is:

\[
h = s - (T_{\min}v_{\text{ego}} + D_{\min})
\]

where:

- \(s\) = bumper-to-bumper gap
- \(v_{\text{ego}}\) = follower velocity
- \(T_{\min}\) = minimum time-gap parameter
- \(D_{\min}\) = minimum distance parameter

The safe set is:

\[
h \geq 0
\]

## CBF Constraint

The implemented CBF constraint is:

\[
u_{\text{CBF}}^{\max}
=
\frac{\Delta v + kh}{T_{\min}}
\]

where:

\[
\Delta v = v_{\text{lead}} - v_{\text{ego}}
\]

The final command is obtained by mediating the nominal ACC command with the CBF constraint.

---

# Prototype 2 Parameters

| Parameter | Value |
|---|---:|
| \(dt\) | 0.1 s |
| \(T_{\min}\) | 2.0 s |
| \(D_{\min}\) | 15.0 m |
| \(k\) | 0.1 |
| \(T_{\text{des}}\) | 1.5 s |
| \(D_{\text{des}}\) | 5.0 m |
| Maximum acceleration | +2.0 m/s² |
| Maximum braking | -4.0 m/s² |
| Maximum jerk | 1.5 m/s³ |

---

# Vicolungo Dataset

The project uses the **Vicolungo car-following data from the OpenACC dataset**.

The dataset is primarily used as a realistic trajectory source for evaluating the controller.

The evaluation follows a counterfactual closed-loop setup:

```text
Recorded Vicolungo Lead Vehicle
              │
              ▼
        Lead trajectory
              │
              ▼
      Our autonomous follower
              │
       ACC / ACC + CBF
              │
              ▼
        Simulated follower
```

The historical follower trajectory is not simply replayed by our controller.

Instead, the recorded lead trajectory is treated as an external disturbance while the follower is controlled by our own controller.

---

# Prototype 2 Evaluation

Prototype 2 evaluates:

- Forward invariance
- Safety recovery
- Minimum gap
- Collision avoidance
- CBF intervention
- Actuator saturation
- Jerk limitation
- Sensitivity to \(T_{\min}\)
- Baseline ACC vs ACC + CBF

The project also includes stress-case analysis and paper-style figures.

---

# Prototype 2 Interactive Visualizer

A lightweight autonomous-driving style visualizer is included in:

```text
Prototype 2/visualizer/
```

It provides:

- Highway environment
- Lead vehicle
- Autonomous follower
- CBF safety envelope
- Live gap
- Relative velocity
- Barrier value \(h\)
- Nominal ACC command
- CBF command
- Applied acceleration
- CBF intervention state
- Synchronized plots

It supports both synthetic scenarios and Vicolungo lead-trajectory replay.

A direct launcher is also provided:

```text
Prototype 2/launch_visualizer.py
```

---

# Prototype 3 — CBF-RL

Prototype 3 extends the classical CBF controller into a reinforcement learning framework.

The RL framework investigates four configurations:

## 1. Nominal RL

```text
PPO
 ↓
Vehicle
```

No CBF filter and no CBF-specific reward.

## 2. Filter Only

```text
PPO
 ↓
CBF Safety Filter
 ↓
Vehicle
```

The CBF provides runtime safety filtering.

## 3. Reward Only

```text
PPO
 ↓
CBF-based Reward
 ↓
Vehicle
```

The policy receives CBF-related safety feedback during training but no runtime safety filter.

## 4. Dual CBF-RL

```text
                 ┌── CBF Reward ──┐
                 │                │
PPO ─────────────┴── CBF Filter ─┴── Vehicle
```

The policy receives CBF-based training feedback while also being protected by the runtime CBF filter.

---

# PPO State and Action

The environment uses a normalized four-dimensional observation:

\[
o =
[s,\;v_{\text{ego}},\;\Delta v,\;h]
\]

The PPO policy outputs a continuous action:

\[
a\in[-1,1]
\]

which is mapped to the physical acceleration range:

\[
u\in[-4,\;2.5]\;m/s^2
\]

---

# CBF-RL Training

Prototype 3 uses a lightweight PPO actor-critic architecture.

The main components include:

- Actor network
- Critic network
- Generalized Advantage Estimation (GAE)
- PPO clipping
- Entropy regularization
- Gradient clipping
- CBF safety filtering
- CBF-based reward shaping
- Episode telemetry

The rollout buffer preserves the **original policy action and its log probability**, even when the CBF modifies the action before it reaches the vehicle dynamics.

---

# Prototype 3 Evaluation

The RL policies are evaluated in two environments.

## Synthetic Environment

Used for PPO policy learning and controlled experimentation.

Synthetic scenarios include:

- cruising
- braking
- stop-and-go behavior
- closing-speed situations
- disturbance scenarios

## Vicolungo Evaluation

The trained policy is evaluated against recorded Vicolungo lead trajectories.

```text
Vicolungo Recorded Lead
          │
          ▼
      Lead Vehicle
          │
          ▼
      PPO Policy
          │
       CBF Filter
          │
          ▼
   Simulated Follower
```

The historical follower trajectory is not replayed after initialization.

This provides a counterfactual closed-loop evaluation of the learned policy.

---

# Prototype 3 Visualizer

Prototype 3 also contains an interactive car-following visualization.

It demonstrates:

- PPO policy behavior
- CBF safety filtering
- Nominal RL
- Filter Only
- Reward Only
- Dual CBF-RL
- Gap
- Velocity
- Barrier function
- Policy acceleration
- Safe acceleration
- CBF interventions

The visualizer is intended primarily as a research demonstration and explanatory tool.

---

# Simulation vs Research Results

The repository distinguishes between three types of outputs:

```text
[TRAINING — SYNTHETIC]
        │
        └── PPO learning experiments


[VICOLUNGO — REAL-DATA LEAD REPLAY]
        │
        └── Counterfactual closed-loop evaluation


[SYNTHETIC DEMONSTRATION]
        │
        └── Interactive visualization / platoon demonstrations
```

Synthetic visual demonstrations are not treated as empirical Vicolungo results.

---

# CPS Perspective

The project is structured as a cyber-physical control loop:

```text
        Physical Environment
                │
                ▼
          Vehicle State
                │
                ▼
       State / Sensor Model
                │
                ▼
        Cyber Controller
        ┌───────┴────────┐
        │                │
      ACC             CBF / RL
        │                │
        └───────┬────────┘
                ▼
          Control Command
                │
                ▼
        Actuator Constraints
                │
                ▼
          Vehicle Dynamics
                │
                └──── feedback
```

---

# Key Safety Concepts

## Forward Invariance

For an initially safe state:

\[
h(0)\geq0
\]

the objective is to maintain the system inside the safe set:

\[
h(t)\geq0
\]

subject to the assumptions and limitations of the implemented model.

## Recovery

For trajectories that begin with:

\[
h(0)<0
\]

the experiment studies whether the controller can recover the safety margin while avoiding physical collision.

## Actuator Limitations

The mathematical CBF command may require acceleration beyond the physical actuator limit.

Therefore the project explicitly considers:

- acceleration saturation
- braking limits
- jerk limits
- discrete-time implementation

---

# Research Outputs

The repository contains:

- Source code
- Dataset files
- Processed results
- Evaluation CSVs
- Research figures
- Diagnostic plots
- Interactive visualizers
- PPO checkpoints
- Controller validation scripts
- Methodology documentation

---

# Videos

A `videos/` directory will contain demonstration recordings.

Planned demonstrations include:

```text
videos/
├── prototype2_acc_vs_cbf.mp4
├── prototype2_vicolungo_replay.mp4
├── prototype2_emergency_braking.mp4
├── prototype3_rl_demo.mp4
└── prototype3_cbf_rl_comparison.mp4
```

These videos demonstrate the controllers and interactive simulations visually.

---

# Tools and Technologies

Main tools and libraries include:

- Python
- NumPy
- Pandas
- Matplotlib
- PyTorch
- Gymnasium-style environments
- PPO
- Control Barrier Functions
- Simulink / MATLAB (CPS implementation and modelling)

---

# Methodological Notes

The simulations in this repository are research models rather than full vehicle simulators.

The current longitudinal vehicle model abstracts:

- lateral dynamics
- steering
- detailed tire dynamics
- full vehicle powertrain
- complex sensor physics

The Vicolungo evaluation is counterfactual: the recorded lead trajectory is treated as an external trajectory and does not react to the simulated follower.

Therefore, results should be interpreted within the assumptions of the implemented model and experimental protocol.

---

# Project Progression

```text
                 CBF THEORY
                     │
                     ▼
          Prototype 1
       Initial Investigation
                     │
                     ▼
          Prototype 2
         ACC + CBF Safety
                     │
                     ▼
       Vicolungo Evaluation
                     │
                     ▼
          Prototype 3
             CBF + RL
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Filter      Reward      Dual
        Only        Only      CBF-RL
          │          │          │
          └──────────┼──────────┘
                     ▼
           Vicolungo Evaluation
                     │
                     ▼
           Interactive Simulation
                     │
                     ▼
             Simulink CPS Model
```

---

# Goal

The final objective is to understand the relationship between:

\[
\boxed{\text{Safety Constraints}}
\]

\[
\boxed{\text{Classical Control}}
\]

\[
\boxed{\text{Reinforcement Learning}}
\]

and

\[
\boxed{\text{Cyber-Physical Vehicle Systems}}
\]

with Control Barrier Functions providing the central safety mechanism.

---

## Status

- **Prototype 1:** Completed
- **Prototype 2 — ACC + CBF:** Completed
- **Prototype 2 — Vicolungo evaluation:** Completed
- **Prototype 2 — Interactive visualizer:** Completed
- **Prototype 3 — CBF-RL:** Implemented
- **Prototype 3 — PPO experiments:** Ongoing / under evaluation
- **Simulink CPS implementation:** In development
- **Demonstration videos:** To be added
