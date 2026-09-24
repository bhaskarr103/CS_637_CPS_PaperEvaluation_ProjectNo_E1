import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROTOTYPE 2 - ACC + CBF
# ============================================================
#
# Same realistic ACC baseline as Prototype 2 Step 1.
#
# Added:
#   Control Barrier Function safety filter
#
# CBF from the paper:
#
#   h = s - (T_MIN * v + D_MIN)
#
#   u_CBF = (delta_v + K_CBF*h) / T_MIN
#
#   u_cmd = min(u_ACC, u_CBF)
#
# Vehicle constraints:
#   A_MIN <= acceleration <= A_MAX
#   |jerk| <= JERK_MAX
#
# ============================================================


# ============================================================
# FILE
# ============================================================

DATA_FILE = r"E:\CPS Project\Prototype 2\Vicolungo.csv"


# ============================================================
# ACC PARAMETERS
# ============================================================

D_DES = 5.0
T_DES = 1.5

K_GAP = 0.20
K_REL = 0.80


# ============================================================
# VEHICLE LIMITS
# ============================================================

A_MAX = 2.0
A_MIN = -4.0

JERK_MAX = 1.5


# ============================================================
# CBF PARAMETERS
# ============================================================
#
# Exact parameters used in the paper.
# ============================================================

T_MIN = 2.0
D_MIN = 15.0
K_CBF = 0.1


# ============================================================
# SIMULATION
# ============================================================

DT = 0.1


# ============================================================
# COLLISION
# ============================================================

COLLISION_GAP = 0.5


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Vicolungo dataset...")

df = pd.read_csv(DATA_FILE)

print("Rows:", len(df))
print("Trajectories:", df["Trajectory_ID"].nunique())


# ============================================================
# STORAGE
# ============================================================

all_results = []
trajectory_summary = []


# ============================================================
# PROCESS TRAJECTORIES
# ============================================================

trajectory_ids = sorted(
    df["Trajectory_ID"].unique()
)


print("\n======================================")
print("RUNNING PROTOTYPE 2 - ACC + CBF")
print("======================================")


for traj_id in trajectory_ids:

    data = df[
        df["Trajectory_ID"] == traj_id
    ].copy()

    data = data.sort_values(
        "Time_Index"
    ).reset_index(drop=True)


    if len(data) < 2:
        continue


    # --------------------------------------------------------
    # Initial state
    # --------------------------------------------------------

    ego_speed = float(
        data.loc[0, "Speed_FAV"]
    )

    gap = float(
        data.loc[0, "Spatial_Gap"]
    )

    previous_acc = 0.0


    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    times = []
    lead_speeds = []
    ego_speeds = []

    gaps = []
    desired_gaps = []

    relative_speeds = []

    h_values = []

    u_nom_values = []
    u_cbf_values = []
    u_raw_values = []
    u_cmd_values = []

    accelerations = []

    ttcs = []

    interventions = []


    # ========================================================
    # SIMULATION
    # ========================================================

    for i in range(len(data)):

        row = data.iloc[i]


        # ----------------------------------------------------
        # Lead speed
        # ----------------------------------------------------

        lead_speed = float(
            row["Speed_LV"]
        )


        # ----------------------------------------------------
        # Relative velocity
        #
        # Positive:
        # lead faster than ego
        #
        # Negative:
        # ego faster than lead
        # ----------------------------------------------------

        delta_v = (
            lead_speed
            -
            ego_speed
        )


        # ----------------------------------------------------
        # ACC desired gap
        # ----------------------------------------------------

        desired_gap = (
            D_DES
            +
            T_DES * ego_speed
        )


        # ----------------------------------------------------
        # ACC nominal controller
        # ----------------------------------------------------

        gap_error = (
            gap
            -
            desired_gap
        )


        u_nom = (
            K_GAP * gap_error
            +
            K_REL * delta_v
        )


        # ----------------------------------------------------
        # ACC acceleration limit
        # ----------------------------------------------------

        u_nom = np.clip(
            u_nom,
            A_MIN,
            A_MAX
        )


        # ====================================================
        # CBF
        # ====================================================

        # Safety function:
        #
        # h >= 0  -> inside safety set
        # h <  0  -> outside safety set
        #

        h = (
            gap
            -
            (
                T_MIN * ego_speed
                +
                D_MIN
            )
        )


        # ----------------------------------------------------
        # CBF acceleration limit
        #
        # u <= (delta_v + k*h) / T
        # ----------------------------------------------------

        u_cbf = (
            delta_v
            +
            K_CBF * h
        ) / T_MIN


        # ----------------------------------------------------
        # CBF supervisor
        # ----------------------------------------------------

        u_raw = min(
            u_nom,
            u_cbf
        )


        # ----------------------------------------------------
        # Acceleration saturation
        # ----------------------------------------------------

        u_raw_limited = np.clip(
            u_raw,
            A_MIN,
            A_MAX
        )


        # ----------------------------------------------------
        # Detect CBF intervention
        # ----------------------------------------------------

        intervention = (
            u_cbf
            <
            u_nom
        )


        # ====================================================
        # JERK LIMIT
        # ====================================================

        max_acc_change = (
            JERK_MAX * DT
        )


        acc_change = (
            u_raw_limited
            -
            previous_acc
        )


        acc_change = np.clip(
            acc_change,
            -max_acc_change,
            max_acc_change
        )


        acceleration = (
            previous_acc
            +
            acc_change
        )


        previous_acc = acceleration


        # ----------------------------------------------------
        # TTC
        # ----------------------------------------------------

        closing_speed = (
            ego_speed
            -
            lead_speed
        )


        if (
            closing_speed > 0
            and gap > 0
        ):

            ttc = (
                gap
                /
                closing_speed
            )

        else:

            ttc = np.inf


        # ====================================================
        # STORE
        # ====================================================

        times.append(
            float(row["Time_Index"])
        )

        lead_speeds.append(
            lead_speed
        )

        ego_speeds.append(
            ego_speed
        )

        gaps.append(
            gap
        )

        desired_gaps.append(
            desired_gap
        )

        relative_speeds.append(
            delta_v
        )

        h_values.append(
            h
        )

        u_nom_values.append(
            u_nom
        )

        u_cbf_values.append(
            u_cbf
        )

        u_raw_values.append(
            u_raw
        )

        u_cmd_values.append(
            acceleration
        )

        accelerations.append(
            acceleration
        )

        ttcs.append(
            ttc
        )

        interventions.append(
            intervention
        )


        # ====================================================
        # VEHICLE DYNAMICS
        # ====================================================

        ego_speed_next = (
            ego_speed
            +
            acceleration * DT
        )

        ego_speed_next = max(
            0.0,
            ego_speed_next
        )


        # ----------------------------------------------------
        # Gap dynamics
        # ----------------------------------------------------

        gap_next = (
            gap
            +
            (
                lead_speed
                -
                ego_speed
            )
            * DT
        )


        # ----------------------------------------------------
        # Update
        # ----------------------------------------------------

        ego_speed = ego_speed_next
        gap = gap_next


    # ========================================================
    # RESULT DATAFRAME
    # ========================================================

    result = pd.DataFrame({

        "Trajectory_ID":
            traj_id,

        "Time":
            times,

        "Lead_speed":
            lead_speeds,

        "Ego_speed":
            ego_speeds,

        "Gap":
            gaps,

        "Desired_gap":
            desired_gaps,

        "Relative_speed":
            relative_speeds,

        "h":
            h_values,

        "u_nom":
            u_nom_values,

        "u_cbf":
            u_cbf_values,

        "u_raw":
            u_raw_values,

        "u_cmd":
            u_cmd_values,

        "Acceleration":
            accelerations,

        "TTC":
            ttcs,

        "CBF_intervention":
            interventions
    })


    all_results.append(
        result
    )


    # ========================================================
    # TRAJECTORY METRICS
    # ========================================================

    min_gap = (
        result["Gap"].min()
    )


    finite_ttc = result[
        np.isfinite(
            result["TTC"]
        )
    ]["TTC"]


    if len(finite_ttc) > 0:

        min_ttc = (
            finite_ttc.min()
        )

    else:

        min_ttc = np.inf


    mean_gap = (
        result["Gap"].mean()
    )


    mean_speed = (
        result["Ego_speed"].mean()
    )


    mean_speed_error = np.mean(
        np.abs(
            result["Ego_speed"]
            -
            result["Lead_speed"]
        )
    )


    max_braking = (
        result["Acceleration"].min()
    )


    # --------------------------------------------------------
    # Jerk
    # --------------------------------------------------------

    jerk = (
        np.diff(
            result["Acceleration"]
        )
        /
        DT
    )


    if len(jerk) > 0:

        max_abs_jerk = np.max(
            np.abs(jerk)
        )

    else:

        max_abs_jerk = 0.0


    # --------------------------------------------------------
    # CBF intervention percentage
    # --------------------------------------------------------

    intervention_percent = (
        result["CBF_intervention"].mean()
        *
        100
    )


    # --------------------------------------------------------
    # Unsafe CBF samples
    # --------------------------------------------------------

    unsafe_samples = (
        result["h"] < 0
    ).sum()


    # --------------------------------------------------------
    # Raw CBF infeasible
    # --------------------------------------------------------

    infeasible_samples = (
        result["u_cbf"] < A_MIN
    ).sum()


    # --------------------------------------------------------
    # Collision
    # --------------------------------------------------------

    collision = (
        min_gap <= COLLISION_GAP
    )


    trajectory_summary.append({

        "Trajectory_ID":
            traj_id,

        "Initial_gap":
            gap if False else
            float(data.loc[0, "Spatial_Gap"]),

        "Minimum_gap":
            min_gap,

        "Minimum_TTC":
            min_ttc,

        "Mean_gap":
            mean_gap,

        "Mean_speed":
            mean_speed,

        "Mean_speed_error":
            mean_speed_error,

        "Maximum_braking":
            max_braking,

        "Maximum_abs_jerk":
            max_abs_jerk,

        "CBF_intervention_percent":
            intervention_percent,

        "CBF_unsafe_samples":
            unsafe_samples,

        "CBF_infeasible_samples":
            infeasible_samples,

        "Collision":
            collision
    })


    print(
        f"Trajectory {traj_id}: "
        f"min gap = {min_gap:.2f} m, "
        f"min TTC = {min_ttc:.2f} s, "
        f"CBF intervention = "
        f"{intervention_percent:.1f}%"
    )


# ============================================================
# COMBINE
# ============================================================

results = pd.concat(
    all_results,
    ignore_index=True
)

summary = pd.DataFrame(
    trajectory_summary
)


# ============================================================
# SAVE
# ============================================================

RESULT_FILE = (
    r"E:\CPS Project\Prototype 2"
    r"\prototype2_acc_cbf.csv"
)

SUMMARY_FILE = (
    r"E:\CPS Project\Prototype 2"
    r"\prototype2_acc_cbf_summary.csv"
)


results.to_csv(
    RESULT_FILE,
    index=False
)

summary.to_csv(
    SUMMARY_FILE,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n======================================")
print("PROTOTYPE 2 ACC + CBF RESULTS")
print("======================================")


print(
    "Trajectories:",
    len(summary)
)


print(
    "Collision trajectories:",
    summary["Collision"].sum()
)


print(
    "Minimum gap:",
    summary["Minimum_gap"].min(),
    "m"
)


finite_ttc = summary[
    np.isfinite(
        summary["Minimum_TTC"]
    )
]["Minimum_TTC"]


if len(finite_ttc) > 0:

    print(
        "Minimum TTC:",
        finite_ttc.min(),
        "s"
    )

else:

    print(
        "Minimum TTC: No closing events"
    )


print(
    "Average gap:",
    summary["Mean_gap"].mean(),
    "m"
)


print(
    "Average ego speed:",
    summary["Mean_speed"].mean(),
    "m/s"
)


print(
    "Average speed error:",
    summary["Mean_speed_error"].mean(),
    "m/s"
)


print(
    "Maximum braking:",
    summary["Maximum_braking"].min(),
    "m/s²"
)


print(
    "Maximum absolute jerk:",
    summary["Maximum_abs_jerk"].max(),
    "m/s³"
)


print(
    "Average CBF intervention:",
    summary[
        "CBF_intervention_percent"
    ].mean(),
    "%"
)


print(
    "Total CBF unsafe samples:",
    summary[
        "CBF_unsafe_samples"
    ].sum()
)


print(
    "Total infeasible CBF samples:",
    summary[
        "CBF_infeasible_samples"
    ].sum()
)


# ============================================================
# SELECT MOST CRITICAL TRAJECTORY
# ============================================================

selected_traj = summary.loc[
    summary["Minimum_gap"].idxmin(),
    "Trajectory_ID"
]


selected = results[
    results["Trajectory_ID"]
    ==
    selected_traj
]


print("\n======================================")
print("SELECTED STRESS TRAJECTORY")
print("======================================")

print(
    "Trajectory:",
    selected_traj
)


# ============================================================
# PLOT 1 - SPEED
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    selected["Time"],
    selected["Ego_speed"],
    label="Ego vehicle"
)

plt.plot(
    selected["Time"],
    selected["Lead_speed"],
    label="Lead vehicle"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Speed (m/s)"
)

plt.title(
    f"ACC + CBF Speed - "
    f"Trajectory {selected_traj}"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 2 - GAP
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    selected["Time"],
    selected["Gap"],
    label="Actual gap"
)

plt.plot(
    selected["Time"],
    selected["Desired_gap"],
    label="Desired ACC gap"
)

plt.axhline(
    COLLISION_GAP,
    linestyle="--",
    label="Collision threshold"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Gap (m)"
)

plt.title(
    f"ACC + CBF Gap - "
    f"Trajectory {selected_traj}"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 3 - CBF SAFETY FUNCTION
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    selected["Time"],
    selected["h"],
    label="CBF h"
)

plt.axhline(
    0,
    linestyle="--",
    label="Safety boundary h = 0"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "h (m)"
)

plt.title(
    f"CBF Safety Function - "
    f"Trajectory {selected_traj}"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 4 - CONTROL
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    selected["Time"],
    selected["u_nom"],
    label="ACC nominal"
)

plt.plot(
    selected["Time"],
    selected["u_cbf"],
    label="CBF limit"
)

plt.plot(
    selected["Time"],
    selected["u_cmd"],
    label="Applied acceleration"
)

plt.axhline(
    A_MIN,
    linestyle="--",
    label="Maximum braking"
)

plt.xlabel(
    "Time (s)"
)

plt.ylabel(
    "Acceleration (m/s²)"
)

plt.title(
    f"ACC + CBF Control - "
    f"Trajectory {selected_traj}"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 5 - INTERVENTIONS
# ============================================================

plt.figure(
    figsize=(10, 4)
)

intervention_times = selected[
    selected["CBF_intervention"]
]["Time"]

plt.scatter(
    intervention_times,
    np.ones(
        len(intervention_times)
    )
)

plt.xlabel(
    "Time (s)"
)

plt.yticks(
    [1],
    ["CBF active"]
)

plt.title(
    f"CBF Intervention Events - "
    f"Trajectory {selected_traj}"
)

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# COMPLETE
# ============================================================

print("\n======================================")
print("PROTOTYPE 2 STEP 2 COMPLETE")
print("======================================")

print("\nResults saved:")

print(
    RESULT_FILE
)

print(
    SUMMARY_FILE
)