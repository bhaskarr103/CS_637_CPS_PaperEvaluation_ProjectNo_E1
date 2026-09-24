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

# Nominal ACC
K_SPEED = 0.5
K_GAP = 0.15

# Vehicle limits
MAX_ACCEL = 2.0
MAX_BRAKE = -8.0

# Collision definition
COLLISION_GAP = 0.5


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FILE)

print("Dataset loaded")
print("Rows:", len(df))
print("Trajectories:", df["Trajectory_ID"].nunique())


# ============================================================
# NOMINAL ACC
# ============================================================

def nominal_acc(speed, lead_speed, gap):

    desired_gap = T_MIN * speed + D_MIN

    gap_error = gap - desired_gap

    speed_error = lead_speed - speed

    u = (
        K_GAP * gap_error
        + K_SPEED * speed_error
    )

    return np.clip(
        u,
        MAX_BRAKE,
        MAX_ACCEL
    )


# ============================================================
# CBF
# ============================================================

def calculate_cbf(speed, lead_speed, gap):

    h = gap - (
        T_MIN * speed + D_MIN
    )

    delta_v = lead_speed - speed

    # Exact CBF acceleration limit
    u_cbf_raw = (
        delta_v + K_CBF * h
    ) / T_MIN

    return h, delta_v, u_cbf_raw


# ============================================================
# SIMULATE ONE TRAJECTORY
# ============================================================

def simulate_trajectory(traj, use_cbf=False):

    traj = traj.sort_values("Time_Index")

    dt = 0.1

    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    speed = float(traj.iloc[0]["Speed_FAV"])
    gap = float(traj.iloc[0]["Spatial_Gap"])

    initial_gap = gap

    results = []

    collision = False
    intervention_count = 0
    infeasible_count = 0

    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    for _, row in traj.iterrows():

        lead_speed = float(row["Speed_LV"])

        # --------------------------------------------
        # CBF quantities
        # --------------------------------------------

        h, delta_v, u_cbf_raw = calculate_cbf(
            speed,
            lead_speed,
            gap
        )

        # --------------------------------------------
        # Nominal ACC
        # --------------------------------------------

        u_nom = nominal_acc(
            speed,
            lead_speed,
            gap
        )

        # --------------------------------------------
        # Controller
        # --------------------------------------------

        if use_cbf:

            # Check whether CBF asks for more braking
            # than the vehicle can physically provide.

            if u_cbf_raw < MAX_BRAKE:
                infeasible_count += 1

            # Apply actuator saturation
            u_cbf = np.clip(
                u_cbf_raw,
                MAX_BRAKE,
                MAX_ACCEL
            )

            # CBF safety filter
            u_cmd = min(
                u_nom,
                u_cbf
            )

            if u_cmd < u_nom - 1e-6:
                intervention_count += 1

        else:

            u_cbf = np.nan
            u_cmd = u_nom

        # --------------------------------------------
        # TTC
        # --------------------------------------------

        if gap > 0 and delta_v < 0:

            ttc = gap / (-delta_v)

        else:

            ttc = np.inf

        # --------------------------------------------
        # Save state
        # --------------------------------------------

        results.append({
            "time": row["Time_Index"],
            "lead_speed": lead_speed,
            "speed": speed,
            "gap": gap,
            "h": h,
            "delta_v": delta_v,
            "u_nom": u_nom,
            "u_cbf": u_cbf,
            "u_cbf_raw": u_cbf_raw,
            "u_cmd": u_cmd,
            "ttc": ttc
        })

        # --------------------------------------------
        # Collision check
        # --------------------------------------------

        if gap <= COLLISION_GAP:

            collision = True
            break

        # --------------------------------------------
        # Vehicle dynamics
        # --------------------------------------------

        new_speed = speed + u_cmd * dt

        new_speed = max(
            new_speed,
            0.0
        )

        # Relative longitudinal motion

        new_gap = (
            gap
            + (lead_speed - speed) * dt
            - 0.5 * u_cmd * dt * dt
        )

        speed = new_speed
        gap = new_gap

    result_df = pd.DataFrame(results)

    return (
        result_df,
        collision,
        initial_gap,
        intervention_count,
        infeasible_count
    )


# ============================================================
# RUN ALL TRAJECTORIES
# ============================================================

acc_results = []
cbf_results = []

summary = []

print("\nRunning simulations...\n")

for traj_id, traj in df.groupby("Trajectory_ID"):

    print("Trajectory:", traj_id)

    # ACC only
    acc, acc_collision, initial_gap, _, _ = (
        simulate_trajectory(
            traj,
            use_cbf=False
        )
    )

    # ACC + CBF
    cbf, cbf_collision, _, interventions, infeasible = (
        simulate_trajectory(
            traj,
            use_cbf=True
        )
    )

    acc["Trajectory_ID"] = traj_id
    cbf["Trajectory_ID"] = traj_id

    acc_results.append(acc)
    cbf_results.append(cbf)

    summary.append({
        "Trajectory_ID": traj_id,
        "Initial_gap": initial_gap,
        "Initially_unsafe": initial_gap < COLLISION_GAP,
        "ACC_collision": acc_collision,
        "CBF_collision": cbf_collision,
        "CBF_interventions": interventions,
        "CBF_infeasible_samples": infeasible
    })


acc_results = pd.concat(
    acc_results,
    ignore_index=True
)

cbf_results = pd.concat(
    cbf_results,
    ignore_index=True
)

summary = pd.DataFrame(summary)


# ============================================================
# SUMMARY
# ============================================================

print("\n======================================")
print("TRAJECTORY SUMMARY")
print("======================================")

print(
    "Total trajectories:",
    len(summary)
)

print(
    "Initially unsafe:",
    summary["Initially_unsafe"].sum()
)

print(
    "Initially reasonable:",
    (~summary["Initially_unsafe"]).sum()
)


# ============================================================
# COLLISION RESULTS
# ============================================================

print("\n======================================")
print("COLLISION RESULTS")
print("======================================")

print(
    "ACC collisions:",
    summary["ACC_collision"].sum()
)

print(
    "CBF collisions:",
    summary["CBF_collision"].sum()
)


# Only consider trajectories that did NOT already
# start inside the collision threshold.

valid = summary[
    ~summary["Initially_unsafe"]
]

print("\nAfter removing initially-collided trajectories:")

print(
    "ACC collisions:",
    valid["ACC_collision"].sum(),
    "/",
    len(valid)
)

print(
    "CBF collisions:",
    valid["CBF_collision"].sum(),
    "/",
    len(valid)
)


# ============================================================
# MINIMUM GAP
# ============================================================

print("\n======================================")
print("MINIMUM GAP")
print("======================================")

print(
    "ACC minimum gap:",
    acc_results["gap"].min()
)

print(
    "CBF minimum gap:",
    cbf_results["gap"].min()
)


# ============================================================
# TTC
# ============================================================

print("\n======================================")
print("TTC")
print("======================================")

acc_ttc = acc_results["ttc"].replace(
    np.inf,
    np.nan
)

cbf_ttc = cbf_results["ttc"].replace(
    np.inf,
    np.nan
)

print(
    "ACC minimum TTC:",
    acc_ttc.min()
)

print(
    "CBF minimum TTC:",
    cbf_ttc.min()
)


# ============================================================
# CBF INTERVENTIONS
# ============================================================

total_interventions = (
    summary["CBF_interventions"].sum()
)

print("\n======================================")
print("CBF INTERVENTIONS")
print("======================================")

print(
    "Intervention samples:",
    total_interventions
)


print(
    "Total CBF samples:",
    len(cbf_results)
)

print(
    "Intervention percentage:",
    total_interventions
    / len(cbf_results)
    * 100
)


# ============================================================
# CBF INFEASIBILITY
# ============================================================

infeasible_total = (
    summary["CBF_infeasible_samples"].sum()
)

print("\n======================================")
print("CBF ACTUATOR FEASIBILITY")
print("======================================")

print(
    "Samples where raw CBF requires",
    f"more than {abs(MAX_BRAKE)} m/s² braking:",
    infeasible_total
)


# ============================================================
# SAVE RESULTS
# ============================================================

acc_results.to_csv(
    r"E:\CPS Project\Prototype 2\results_acc_only_fixed.csv",
    index=False
)

cbf_results.to_csv(
    r"E:\CPS Project\Prototype 2\results_acc_cbf_fixed.csv",
    index=False
)

summary.to_csv(
    r"E:\CPS Project\Prototype 2\trajectory_summary.csv",
    index=False
)

print("\nResults saved.")


# ============================================================
# SELECT A GOOD TRAJECTORY FOR VISUALIZATION
# ============================================================

# We want a trajectory that:
# 1. does not start already collided
# 2. has a meaningful close-following event

candidate_ids = valid[
    valid["ACC_collision"]
]["Trajectory_ID"].tolist()


if len(candidate_ids) == 0:

    # If no collision trajectory exists,
    # select one with the smallest minimum gap.

    candidate_ids = (
        cbf_results[
            cbf_results["Trajectory_ID"].isin(
                valid["Trajectory_ID"]
            )
        ]
        .groupby("Trajectory_ID")["gap"]
        .min()
        .sort_values()
        .index
        .tolist()
    )


if len(candidate_ids) > 0:

    critical_id = candidate_ids[0]

    print(
        "\nSelected trajectory for plot:",
        critical_id
    )

    acc_plot = acc_results[
        acc_results["Trajectory_ID"] == critical_id
    ]

    cbf_plot = cbf_results[
        cbf_results["Trajectory_ID"] == critical_id
    ]


    # ========================================================
    # GAP PLOT
    # ========================================================

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


    # ========================================================
    # SPEED PLOT
    # ========================================================

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

else:

    print(
        "\nNo suitable trajectory found for plotting."
    )