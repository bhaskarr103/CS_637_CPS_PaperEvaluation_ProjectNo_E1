import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAMETERS
# ============================================================

FILE = r"E:\CPS Project\Prototype 2\Vicolungo.csv"

# CBF parameters from the paper
T_MIN = 2.0
D_MIN = 15.0
K_CBF = 0.1

# Simple ACC parameters
K_SPEED = 0.5
K_GAP = 0.15

# Vehicle acceleration limits
MAX_ACCEL = 2.0
MAX_BRAKE = -8.0

# Collision threshold
COLLISION_GAP = 0.5


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FILE)

print("Dataset loaded")
print("Rows:", len(df))
print("Trajectories:", df["Trajectory_ID"].nunique())


# ============================================================
# ACC CONTROLLER
# ============================================================

def nominal_acc(speed, lead_speed, gap):
    """
    Simple nominal ACC controller.

    Desired gap:
        s_desired = T_MIN * speed + D_MIN

    If gap is larger than desired:
        accelerate

    If gap is smaller than desired:
        brake
    """

    desired_gap = T_MIN * speed + D_MIN

    gap_error = gap - desired_gap

    speed_error = lead_speed - speed

    u = (
        K_GAP * gap_error
        + K_SPEED * speed_error
    )

    return np.clip(u, MAX_BRAKE, MAX_ACCEL)


# ============================================================
# CBF CONTROLLER
# ============================================================

def cbf_acc(speed, lead_speed, gap):
    """
    CBF acceleration limit.

    h = gap - (T_MIN * speed + D_MIN)

    delta_v = lead_speed - speed

    u_CBF = (delta_v + K_CBF*h) / T_MIN
    """

    h = gap - (T_MIN * speed + D_MIN)

    delta_v = lead_speed - speed

    u_cbf = (
        delta_v + K_CBF * h
    ) / T_MIN

    return np.clip(
        u_cbf,
        MAX_BRAKE,
        MAX_ACCEL
    )


# ============================================================
# SIMULATION
# ============================================================

def simulate_trajectory(traj, use_cbf=False):

    # Sort by time
    traj = traj.sort_values("Time_Index")

    # --------------------------------------------------------
    # Initial conditions
    # --------------------------------------------------------

    speed = traj.iloc[0]["Speed_FAV"]
    gap = traj.iloc[0]["Spatial_Gap"]

    dt = 0.1

    results = []

    # --------------------------------------------------------
    # Simulation loop
    # --------------------------------------------------------

    for _, row in traj.iterrows():

        lead_speed = row["Speed_LV"]

        # --------------------------------------------
        # Nominal ACC
        # --------------------------------------------

        u_nom = nominal_acc(
            speed,
            lead_speed,
            gap
        )

        # --------------------------------------------
        # CBF
        # --------------------------------------------

        if use_cbf:

            u_cbf = cbf_acc(
                speed,
                lead_speed,
                gap
            )

            # CBF safety filter
            u_cmd = min(
                u_nom,
                u_cbf
            )

        else:

            u_cbf = np.nan
            u_cmd = u_nom

        # --------------------------------------------
        # Save current state
        # --------------------------------------------

        h = (
            gap
            - (T_MIN * speed + D_MIN)
        )

        delta_v = lead_speed - speed

        ttc = np.inf

        if delta_v < 0:
            ttc = gap / (-delta_v)

        results.append({
            "time": row["Time_Index"],
            "lead_speed": lead_speed,
            "speed": speed,
            "gap": gap,
            "h": h,
            "u_nom": u_nom,
            "u_cbf": u_cbf,
            "u_cmd": u_cmd,
            "ttc": ttc
        })

        # --------------------------------------------
        # Vehicle dynamics
        # --------------------------------------------

        speed = speed + u_cmd * dt

        speed = max(speed, 0.0)

        # Relative motion
        gap = gap + (lead_speed - speed) * dt

    return pd.DataFrame(results)


# ============================================================
# RUN SIMULATION
# ============================================================

all_acc = []
all_cbf = []

print("\nRunning simulations...")

for traj_id, traj in df.groupby("Trajectory_ID"):

    print("Trajectory:", traj_id)

    result_acc = simulate_trajectory(
        traj,
        use_cbf=False
    )

    result_cbf = simulate_trajectory(
        traj,
        use_cbf=True
    )

    result_acc["Trajectory_ID"] = traj_id
    result_cbf["Trajectory_ID"] = traj_id

    all_acc.append(result_acc)
    all_cbf.append(result_cbf)


acc_results = pd.concat(
    all_acc,
    ignore_index=True
)

cbf_results = pd.concat(
    all_cbf,
    ignore_index=True
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(results):

    min_gap = results["gap"].min()

    min_ttc = results["ttc"].replace(
        np.inf,
        np.nan
    ).min()

    collisions = (
        results["gap"] < COLLISION_GAP
    ).sum()

    unsafe = (
        results["h"] < 0
    ).sum()

    max_braking = results["u_cmd"].min()

    return {
        "Minimum gap (m)": min_gap,
        "Minimum TTC (s)": min_ttc,
        "CBF unsafe samples": unsafe,
        "Collision samples": collisions,
        "Maximum braking (m/s²)": max_braking
    }


# ============================================================
# RESULTS
# ============================================================

acc_metrics = evaluate(acc_results)
cbf_metrics = evaluate(cbf_results)


print("\n======================================")
print("FINAL COMPARISON")
print("======================================")

print("\nACC ONLY")

for key, value in acc_metrics.items():
    print(f"{key}: {value}")


print("\nACC + CBF")

for key, value in cbf_metrics.items():
    print(f"{key}: {value}")


# ============================================================
# CBF INTERVENTIONS
# ============================================================

interventions = (
    cbf_results["u_cmd"]
    < cbf_results["u_nom"] - 1e-6
)

print("\n======================================")
print("CBF INTERVENTIONS")
print("======================================")

print(
    "Intervention samples:",
    interventions.sum()
)

print(
    "Intervention percentage:",
    interventions.mean() * 100
)


# ============================================================
# SAVE RESULTS
# ============================================================

acc_results.to_csv(
    r"E:\CPS Project\Prototype 2\results_acc_only.csv",
    index=False
)

cbf_results.to_csv(
    r"E:\CPS Project\Prototype 2\results_acc_cbf.csv",
    index=False
)

print("\nResults saved.")


# ============================================================
# PLOT ONE CRITICAL TRAJECTORY
# ============================================================

# Find trajectory with smallest CBF gap
critical_id = (
    cbf_results
    .groupby("Trajectory_ID")["gap"]
    .min()
    .idxmin()
)

acc_plot = acc_results[
    acc_results["Trajectory_ID"] == critical_id
]

cbf_plot = cbf_results[
    cbf_results["Trajectory_ID"] == critical_id
]


# ------------------------------------------------------------
# Gap plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    acc_plot["time"],
    acc_plot["gap"],
    label="ACC only"
)

plt.plot(
    cbf_plot["time"],
    cbf_plot["gap"],
    label="ACC + CBF"
)

plt.axhline(
    COLLISION_GAP,
    linestyle="--",
    label="Collision threshold"
)

plt.xlabel("Time (s)")
plt.ylabel("Gap (m)")
plt.title(
    f"Gap Comparison - Trajectory {critical_id}"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# Speed plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    acc_plot["time"],
    acc_plot["speed"],
    label="ACC only"
)

plt.plot(
    cbf_plot["time"],
    cbf_plot["speed"],
    label="ACC + CBF"
)

plt.plot(
    cbf_plot["time"],
    cbf_plot["lead_speed"],
    label="Lead vehicle"
)

plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.title(
    f"Speed Comparison - Trajectory {critical_id}"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()