import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FILE
# ============================================================

FILE = r"E:\CPS Project\Prototype 2\Vicolungo.csv"


# ============================================================
# CBF PARAMETERS FROM THE PAPER
# ============================================================

T_MIN = 2.0
D_MIN = 15.0
K_CBF = 0.1


# ============================================================
# NOMINAL CONTROLLER
# ============================================================

# The paper defines:
#
#     u_nom = K_NOM * (v - v_des)
#
# The exact K_NOM and v_des values are not specified
# in the CBF paper section, so these are prototype assumptions.

K_NOM = 0.5
V_DES = 30.0


# ============================================================
# VEHICLE LIMITS
# ============================================================

MAX_ACCEL = 2.0
MAX_BRAKE = -8.0

COLLISION_GAP = 0.5

DT = 0.1


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FILE)

print("Dataset loaded")
print("Rows:", len(df))
print("Trajectories:", df["Trajectory_ID"].nunique())


# ============================================================
# NOMINAL CONTROLLER
# ============================================================

def nominal_controller(speed):

    u_nom = K_NOM * (
        speed - V_DES
    )

    return np.clip(
        u_nom,
        MAX_BRAKE,
        MAX_ACCEL
    )


# ============================================================
# CBF
# ============================================================

def cbf_controller(speed, lead_speed, gap):

    # Safety function
    h = (
        gap
        - (T_MIN * speed + D_MIN)
    )

    # Relative velocity
    #
    # Positive  -> lead vehicle faster
    # Negative  -> ego closing on lead
    delta_v = lead_speed - speed

    # CBF acceleration limit
    u_cbf = (
        delta_v
        + K_CBF * h
    ) / T_MIN

    return h, delta_v, u_cbf


# ============================================================
# SIMULATE ONE TRAJECTORY
# ============================================================

def simulate(traj, use_cbf):

    traj = traj.sort_values(
        "Time_Index"
    )

    speed = float(
        traj.iloc[0]["Speed_FAV"]
    )

    gap = float(
        traj.iloc[0]["Spatial_Gap"]
    )

    results = []

    collision = False
    interventions = 0
    infeasible = 0

    for _, row in traj.iterrows():

        lead_speed = float(
            row["Speed_LV"]
        )

        # ----------------------------------------------------
        # Nominal controller
        # ----------------------------------------------------

        u_nom = nominal_controller(
            speed
        )

        # ----------------------------------------------------
        # CBF
        # ----------------------------------------------------

        h, delta_v, u_cbf_raw = (
            cbf_controller(
                speed,
                lead_speed,
                gap
            )
        )

        # ----------------------------------------------------
        # Apply CBF
        # ----------------------------------------------------

        if use_cbf:

            # Check physical feasibility
            if u_cbf_raw < MAX_BRAKE:
                infeasible += 1

            # Physical actuator saturation
            u_cbf = np.clip(
                u_cbf_raw,
                MAX_BRAKE,
                MAX_ACCEL
            )

            # Safety filter
            u_cmd = min(
                u_nom,
                u_cbf
            )

            if u_cmd < u_nom - 1e-6:
                interventions += 1

        else:

            u_cbf = np.nan
            u_cmd = u_nom

        # ----------------------------------------------------
        # TTC
        # ----------------------------------------------------

        if (
            gap > 0
            and delta_v < 0
        ):

            ttc = (
                gap / (-delta_v)
            )

        else:

            ttc = np.inf

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        results.append({

            "time":
                row["Time_Index"],

            "lead_speed":
                lead_speed,

            "speed":
                speed,

            "gap":
                gap,

            "h":
                h,

            "delta_v":
                delta_v,

            "u_nom":
                u_nom,

            "u_cbf":
                u_cbf,

            "u_cbf_raw":
                u_cbf_raw,

            "u_cmd":
                u_cmd,

            "ttc":
                ttc
        })

        # ----------------------------------------------------
        # Collision
        # ----------------------------------------------------

        if gap <= COLLISION_GAP:

            collision = True
            break

        # ----------------------------------------------------
        # Vehicle dynamics
        # ----------------------------------------------------

        new_speed = (
            speed
            + u_cmd * DT
        )

        new_speed = max(
            new_speed,
            0.0
        )

        # ----------------------------------------------------
        # Gap dynamics
        # ----------------------------------------------------

        new_gap = (
            gap
            + delta_v * DT
            - 0.5 * u_cmd * DT * DT
        )

        speed = new_speed
        gap = new_gap

    return (
        pd.DataFrame(results),
        collision,
        interventions,
        infeasible
    )


# ============================================================
# RUN ALL TRAJECTORIES
# ============================================================

acc_results = []
cbf_results = []
summary = []

print("\nRunning simulations...\n")

for traj_id, traj in df.groupby(
    "Trajectory_ID"
):

    print(
        "Trajectory:",
        traj_id
    )

    # ACC / nominal controller
    acc, acc_collision, _, _ = simulate(
        traj,
        use_cbf=False
    )

    # ACC + CBF
    cbf, cbf_collision, interventions, infeasible = simulate(
        traj,
        use_cbf=True
    )

    acc["Trajectory_ID"] = traj_id
    cbf["Trajectory_ID"] = traj_id

    acc_results.append(acc)
    cbf_results.append(cbf)

    summary.append({

        "Trajectory_ID":
            traj_id,

        "Initial_gap":
            traj.iloc[0]["Spatial_Gap"],

        "ACC_collision":
            acc_collision,

        "CBF_collision":
            cbf_collision,

        "CBF_interventions":
            interventions,

        "CBF_infeasible":
            infeasible
    })


acc_results = pd.concat(
    acc_results,
    ignore_index=True
)

cbf_results = pd.concat(
    cbf_results,
    ignore_index=True
)

summary = pd.DataFrame(
    summary
)


# ============================================================
# RESULTS
# ============================================================

print("\n======================================")
print("RESULTS")
print("======================================")

print(
    "ACC collisions:",
    summary["ACC_collision"].sum()
)

print(
    "CBF collisions:",
    summary["CBF_collision"].sum()
)

print(
    "Minimum ACC gap:",
    acc_results["gap"].min()
)

print(
    "Minimum CBF gap:",
    cbf_results["gap"].min()
)


# ============================================================
# TTC
# ============================================================

acc_ttc = (
    acc_results["ttc"]
    .replace(np.inf, np.nan)
)

cbf_ttc = (
    cbf_results["ttc"]
    .replace(np.inf, np.nan)
)

print(
    "Minimum ACC TTC:",
    acc_ttc.min()
)

print(
    "Minimum CBF TTC:",
    cbf_ttc.min()
)


# ============================================================
# CBF INTERVENTIONS
# ============================================================

total_interventions = (
    summary["CBF_interventions"].sum()
)

total_samples = len(
    cbf_results
)

print("\n======================================")
print("CBF INTERVENTIONS")
print("======================================")

print(
    "Intervention samples:",
    total_interventions
)

print(
    "Intervention percentage:",
    total_interventions
    / total_samples
    * 100
)


# ============================================================
# CBF INFEASIBILITY
# ============================================================

print("\n======================================")
print("CBF FEASIBILITY")
print("======================================")

print(
    "Raw CBF requests below",
    MAX_BRAKE,
    "m/s²:",
    summary["CBF_infeasible"].sum()
)


# ============================================================
# FIND THE MOST INTERESTING TRAJECTORY
# ============================================================

# Calculate how much the CBF changes the speed trajectory.

difference = []

for traj_id in cbf_results[
    "Trajectory_ID"
].unique():

    a = acc_results[
        acc_results["Trajectory_ID"]
        == traj_id
    ]

    c = cbf_results[
        cbf_results["Trajectory_ID"]
        == traj_id
    ]

    n = min(
        len(a),
        len(c)
    )

    if n == 0:
        continue

    speed_difference = np.mean(
        np.abs(
            a["speed"].iloc[:n].values
            -
            c["speed"].iloc[:n].values
        )
    )

    gap_difference = np.mean(
        np.abs(
            a["gap"].iloc[:n].values
            -
            c["gap"].iloc[:n].values
        )
    )

    difference.append({

        "Trajectory_ID":
            traj_id,

        "speed_difference":
            speed_difference,

        "gap_difference":
            gap_difference
    })


difference = pd.DataFrame(
    difference
)

difference = difference.sort_values(
    "gap_difference",
    ascending=False
)


# ============================================================
# SELECT TRAJECTORY
# ============================================================

critical_id = int(
    difference.iloc[0]["Trajectory_ID"]
)

print("\n======================================")
print("SELECTED TRAJECTORY")
print("======================================")

print(
    "Trajectory:",
    critical_id
)


acc_plot = acc_results[
    acc_results["Trajectory_ID"]
    == critical_id
]

cbf_plot = cbf_results[
    cbf_results["Trajectory_ID"]
    == critical_id
]


# ============================================================
# GAP
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    acc_plot["time"],
    acc_plot["gap"],
    label="Nominal controller"
)

plt.plot(
    cbf_plot["time"],
    cbf_plot["gap"],
    label="Nominal + CBF"
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


# ============================================================
# SPEED
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    acc_plot["time"],
    acc_plot["speed"],
    label="Nominal controller"
)

plt.plot(
    cbf_plot["time"],
    cbf_plot["speed"],
    label="Nominal + CBF"
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


# ============================================================
# ACCELERATION
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    cbf_plot["time"],
    cbf_plot["u_nom"],
    label="Nominal acceleration"
)

plt.plot(
    cbf_plot["time"],
    cbf_plot["u_cmd"],
    label="CBF-filtered acceleration"
)

plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")

plt.title(
    f"Control Comparison - Trajectory {critical_id}"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# SAVE
# ============================================================

acc_results.to_csv(
    r"E:\CPS Project\Prototype 2\paper_style_nominal.csv",
    index=False
)

cbf_results.to_csv(
    r"E:\CPS Project\Prototype 2\paper_style_cbf.csv",
    index=False
)

summary.to_csv(
    r"E:\CPS Project\Prototype 2\paper_style_summary.csv",
    index=False
)

print("\nResults saved.")