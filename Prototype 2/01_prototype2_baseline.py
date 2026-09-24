import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PROTOTYPE 2
# IMPROVED ACC BASELINE
# ============================================================
#
# This version contains:
#
#   1. Time-gap based ACC
#   2. Relative-speed feedback
#   3. Acceleration saturation
#   4. Jerk limitation
#   5. Longitudinal vehicle dynamics
#
# IMPORTANT:
# CBF is NOT used here.
#
# This is our baseline for Prototype 2.
# ============================================================


# ============================================================
# FILE
# ============================================================

DATA_FILE = r"E:\CPS Project\Prototype 2\Vicolungo.csv"


# ============================================================
# ACC PARAMETERS
# ============================================================

# Desired standstill distance
D_DES = 5.0                 # m

# Desired time gap
T_DES = 1.5                 # s

# Position/gap control gain
K_GAP = 0.20

# Relative velocity gain
K_REL = 0.80


# ============================================================
# VEHICLE LIMITS
# ============================================================

# Maximum acceleration
A_MAX = 2.0                # m/s^2

# Maximum braking
A_MIN = -4.0               # m/s^2

# Maximum jerk
JERK_MAX = 1.5              # m/s^3


# ============================================================
# SIMULATION
# ============================================================

DT = 0.1                    # Vicolungo is 10 Hz


# ============================================================
# COLLISION THRESHOLD
# ============================================================

COLLISION_GAP = 0.5         # m


# ============================================================
# LOAD DATA
# ============================================================

print("Loading Vicolungo dataset...")

df = pd.read_csv(DATA_FILE)

print("Rows:", len(df))
print("Trajectories:", df["Trajectory_ID"].nunique())


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Trajectory_ID",
    "Time_Index",
    "Speed_LV",
    "Speed_FAV",
    "Spatial_Gap"
]

for col in required_columns:

    if col not in df.columns:

        raise ValueError(
            f"Missing required column: {col}"
        )


# ============================================================
# RESULT STORAGE
# ============================================================

all_results = []

trajectory_summary = []


# ============================================================
# PROCESS EACH TRAJECTORY
# ============================================================

trajectory_ids = sorted(
    df["Trajectory_ID"].unique()
)


print("\n======================================")
print("RUNNING PROTOTYPE 2 BASELINE")
print("======================================")


for traj_id in trajectory_ids:

    data = df[
        df["Trajectory_ID"] == traj_id
    ].copy()

    data = data.sort_values(
        "Time_Index"
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # Need at least two samples
    # --------------------------------------------------------

    if len(data) < 2:
        continue


    # --------------------------------------------------------
    # Initial conditions
    # --------------------------------------------------------

    ego_speed = float(
        data.loc[0, "Speed_FAV"]
    )

    initial_gap = float(
        data.loc[0, "Spatial_Gap"]
    )

    previous_acc = 0.0


    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    times = []

    ego_speeds = []

    lead_speeds = []

    gaps = []

    accelerations = []

    desired_gaps = []

    relative_speeds = []


    # ========================================================
    # SIMULATION LOOP
    # ========================================================

    for i in range(len(data)):

        row = data.iloc[i]


        # ----------------------------------------------------
        # Lead vehicle speed
        # ----------------------------------------------------

        lead_speed = float(
            row["Speed_LV"]
        )


        # ----------------------------------------------------
        # Current relative speed
        #
        # Positive:
        # ego faster than lead
        #
        # Negative:
        # ego slower than lead
        # ----------------------------------------------------

        relative_speed = (
            ego_speed - lead_speed
        )


        # ----------------------------------------------------
        # Desired gap
        #
        # s_des = d0 + T * v
        # ----------------------------------------------------

        desired_gap = (
            D_DES
            +
            T_DES * ego_speed
        )


        # ----------------------------------------------------
        # Current gap
        # ----------------------------------------------------

        if i == 0:

            gap = initial_gap

        else:

            gap = gaps[-1]


        # ----------------------------------------------------
        # ACC controller
        #
        # Positive gap error:
        # vehicle is too far away
        # -> accelerate
        #
        # Negative gap error:
        # vehicle is too close
        # -> brake
        # ----------------------------------------------------

        gap_error = (
            gap - desired_gap
        )


        acceleration_command = (
            K_GAP * gap_error
            -
            K_REL * relative_speed
        )


        # ----------------------------------------------------
        # Acceleration saturation
        # ----------------------------------------------------

        acceleration_command = np.clip(
            acceleration_command,
            A_MIN,
            A_MAX
        )


        # ----------------------------------------------------
        # JERK LIMIT
        #
        # Prevent instantaneous changes in acceleration.
        #
        # Δa <= JERK_MAX * DT
        # ----------------------------------------------------

        max_acc_change = (
            JERK_MAX * DT
        )


        acc_change = (
            acceleration_command
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


        # ----------------------------------------------------
        # Store acceleration
        # ----------------------------------------------------

        previous_acc = acceleration


        # ----------------------------------------------------
        # Vehicle dynamics
        #
        # v(k+1) = v(k) + a(k)*dt
        # ----------------------------------------------------

        if i < len(data) - 1:

            ego_speed_next = (
                ego_speed
                +
                acceleration * DT
            )

            # Vehicle cannot move backwards
            ego_speed_next = max(
                0.0,
                ego_speed_next
            )

        else:

            ego_speed_next = ego_speed


        # ----------------------------------------------------
        # Gap dynamics
        #
        # s_dot = v_lead - v_ego
        #
        # s(k+1) =
        # s(k) + (v_lead-v_ego)*dt
        # ----------------------------------------------------

        if i < len(data) - 1:

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

        else:

            gap_next = gap


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        times.append(
            float(row["Time_Index"])
        )

        ego_speeds.append(
            ego_speed
        )

        lead_speeds.append(
            lead_speed
        )

        gaps.append(
            gap
        )

        accelerations.append(
            acceleration
        )

        desired_gaps.append(
            desired_gap
        )

        relative_speeds.append(
            relative_speed
        )


        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        ego_speed = ego_speed_next

        if i < len(data) - 1:

            gaps[-1] = gap

            # We need the next gap available
            # in the following iteration.
            #
            # Store it temporarily by appending it
            # through a separate variable.
            pass


    # ========================================================
    # RECONSTRUCT GAP PROPERLY
    # ========================================================
    #
    # The first loop above calculates the state evolution.
    # Now create a clean simulation using the stored
    # acceleration history.
    #
    # This makes the resulting CSV easier to inspect.
    # ========================================================


    ego_speed = float(
        data.loc[0, "Speed_FAV"]
    )

    gap = initial_gap

    previous_acc = 0.0

    final_times = []
    final_ego_speed = []
    final_lead_speed = []
    final_gap = []
    final_acc = []
    final_desired_gap = []
    final_relative_speed = []
    final_ttc = []


    for i in range(len(data)):

        row = data.iloc[i]

        lead_speed = float(
            row["Speed_LV"]
        )


        # ----------------------------------------------------
        # Relative speed
        # ----------------------------------------------------

        relative_speed = (
            ego_speed - lead_speed
        )


        # ----------------------------------------------------
        # Desired gap
        # ----------------------------------------------------

        desired_gap = (
            D_DES
            +
            T_DES * ego_speed
        )


        # ----------------------------------------------------
        # ACC controller
        # ----------------------------------------------------

        gap_error = (
            gap - desired_gap
        )


        acceleration_command = (
            K_GAP * gap_error
            -
            K_REL * relative_speed
        )


        # ----------------------------------------------------
        # Acceleration limit
        # ----------------------------------------------------

        acceleration_command = np.clip(
            acceleration_command,
            A_MIN,
            A_MAX
        )


        # ----------------------------------------------------
        # Jerk limit
        # ----------------------------------------------------

        max_acc_change = (
            JERK_MAX * DT
        )

        acc_change = (
            acceleration_command
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
        #
        # Only meaningful while closing.
        # ----------------------------------------------------

        closing_speed = (
            ego_speed - lead_speed
        )

        if (
            closing_speed > 0
            and gap > 0
        ):

            ttc = (
                gap / closing_speed
            )

        else:

            ttc = np.inf


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        final_times.append(
            float(row["Time_Index"])
        )

        final_ego_speed.append(
            ego_speed
        )

        final_lead_speed.append(
            lead_speed
        )

        final_gap.append(
            gap
        )

        final_acc.append(
            acceleration
        )

        final_desired_gap.append(
            desired_gap
        )

        final_relative_speed.append(
            relative_speed
        )

        final_ttc.append(
            ttc
        )


        # ----------------------------------------------------
        # Integrate vehicle
        # ----------------------------------------------------

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
        # Integrate gap
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


        ego_speed = ego_speed_next
        gap = gap_next


    # ========================================================
    # CREATE TRAJECTORY RESULT
    # ========================================================

    result = pd.DataFrame({

        "Trajectory_ID":
            traj_id,

        "Time":
            final_times,

        "Lead_speed":
            final_lead_speed,

        "Ego_speed":
            final_ego_speed,

        "Gap":
            final_gap,

        "Desired_gap":
            final_desired_gap,

        "Relative_speed":
            final_relative_speed,

        "Acceleration":
            final_acc,

        "TTC":
            final_ttc
    })


    all_results.append(result)


    # ========================================================
    # TRAJECTORY METRICS
    # ========================================================

    min_gap = result["Gap"].min()

    min_ttc_values = result[
        np.isfinite(result["TTC"])
    ]["TTC"]

    if len(min_ttc_values) > 0:

        min_ttc = min_ttc_values.min()

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

    jerk = np.diff(
        result["Acceleration"]
    ) / DT


    if len(jerk) > 0:

        max_jerk = np.max(
            np.abs(jerk)
        )

    else:

        max_jerk = 0.0


    collision = (
        min_gap <= COLLISION_GAP
    )


    trajectory_summary.append({

        "Trajectory_ID":
            traj_id,

        "Initial_gap":
            initial_gap,

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
            max_jerk,

        "Collision":
            collision
    })


    print(
        f"Trajectory {traj_id}: "
        f"min gap = {min_gap:.2f} m, "
        f"min TTC = {min_ttc:.2f} s"
    )


# ============================================================
# COMBINE RESULTS
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
    r"\prototype2_acc_baseline.csv"
)

SUMMARY_FILE = (
    r"E:\CPS Project\Prototype 2"
    r"\prototype2_acc_summary.csv"
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
print("PROTOTYPE 2 BASELINE RESULTS")
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
    np.isfinite(summary["Minimum_TTC"])
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


# ============================================================
# PERCENTAGE OF COLLISIONS
# ============================================================

collision_rate = (
    summary["Collision"].mean()
    * 100
)


print(
    "Collision rate:",
    collision_rate,
    "%"
)


# ============================================================
# PLOT 1
# SELECTED TRAJECTORY
# ============================================================

# Choose trajectory automatically:
#
# Pick the trajectory with the smallest
# simulated minimum gap.

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
print("SELECTED TRAJECTORY")
print("======================================")

print(
    "Trajectory:",
    selected_traj
)


# ============================================================
# SPEED PLOT
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
    f"Prototype 2 ACC Speed - "
    f"Trajectory {selected_traj}"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# GAP PLOT
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
    f"Prototype 2 ACC Gap - "
    f"Trajectory {selected_traj}"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# ACCELERATION PLOT
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    selected["Time"],
    selected["Acceleration"],
    label="ACC acceleration"
)

plt.axhline(
    A_MAX,
    linestyle="--",
    label="Maximum acceleration"
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
    f"Prototype 2 ACC Control - "
    f"Trajectory {selected_traj}"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# COMPLETE
# ============================================================

print("\n======================================")
print("PROTOTYPE 2 STEP 1 COMPLETE")
print("======================================")

print(
    "\nResults saved:"
)

print(
    RESULT_FILE
)

print(
    SUMMARY_FILE
)