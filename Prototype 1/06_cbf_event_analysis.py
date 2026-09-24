import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FILES
# ============================================================

ACC_FILE = r"E:\CPS Project\Prototype 2\paper_style_nominal.csv"
CBF_FILE = r"E:\CPS Project\Prototype 2\paper_style_cbf.csv"

COLLISION_GAP = 0.5


# ============================================================
# LOAD RESULTS
# ============================================================

acc = pd.read_csv(ACC_FILE)
cbf = pd.read_csv(CBF_FILE)

print("Results loaded")


# ============================================================
# FIND TRAJECTORIES
# ============================================================

trajectory_ids = sorted(
    set(acc["Trajectory_ID"])
    &
    set(cbf["Trajectory_ID"])
)


saved_events = []


for traj_id in trajectory_ids:

    a = acc[
        acc["Trajectory_ID"] == traj_id
    ]

    c = cbf[
        cbf["Trajectory_ID"] == traj_id
    ]

    # ACC collision?
    acc_collision = (
        a["gap"] <= COLLISION_GAP
    ).any()

    # CBF collision?
    cbf_collision = (
        c["gap"] <= COLLISION_GAP
    ).any()

    # Initial gap
    initial_gap = a.iloc[0]["gap"]

    # We want:
    #
    # ACC crashes
    # CBF does NOT crash
    # trajectory didn't already start in collision

    if (
        acc_collision
        and not cbf_collision
        and initial_gap > COLLISION_GAP
    ):

        # Time of ACC collision
        collision_row = a[
            a["gap"] <= COLLISION_GAP
        ].iloc[0]

        collision_time = collision_row["time"]

        saved_events.append({

            "Trajectory_ID":
                traj_id,

            "Initial_gap":
                initial_gap,

            "Collision_time":
                collision_time
        })


# ============================================================
# PRINT SAVED EVENTS
# ============================================================

saved_events = pd.DataFrame(
    saved_events
)

print("\n======================================")
print("CBF-SAVED COLLISION EVENTS")
print("======================================")

if len(saved_events) == 0:

    print(
        "No clean ACC-collision / CBF-safe event found."
    )

    raise SystemExit


print(
    saved_events.to_string(index=False)
)


# ============================================================
# SELECT FIRST EVENT
# ============================================================

selected = saved_events.iloc[0]

traj_id = selected["Trajectory_ID"]
collision_time = selected["Collision_time"]

print("\nSelected trajectory:", traj_id)
print("ACC collision time:", collision_time)


# ============================================================
# EXTRACT TRAJECTORY
# ============================================================

a = acc[
    acc["Trajectory_ID"] == traj_id
].copy()

c = cbf[
    cbf["Trajectory_ID"] == traj_id
].copy()


# ============================================================
# ZOOM AROUND EVENT
# ============================================================

WINDOW_BEFORE = 8.0
WINDOW_AFTER = 4.0

start_time = (
    collision_time
    - WINDOW_BEFORE
)

end_time = (
    collision_time
    + WINDOW_AFTER
)


a = a[
    (a["time"] >= start_time)
    &
    (a["time"] <= end_time)
]

c = c[
    (c["time"] >= start_time)
    &
    (c["time"] <= end_time)
]


# ============================================================
# GAP COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    a["time"],
    a["gap"],
    label="Nominal controller"
)

plt.plot(
    c["time"],
    c["gap"],
    label="Nominal + CBF"
)

plt.axhline(
    COLLISION_GAP,
    linestyle="--",
    label="Collision threshold"
)

plt.axvline(
    collision_time,
    linestyle=":",
    label="Nominal collision"
)

plt.xlabel("Time (s)")
plt.ylabel("Gap (m)")

plt.title(
    f"CBF Collision Prevention - Trajectory {traj_id}"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# SPEED COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    a["time"],
    a["speed"],
    label="Nominal controller"
)

plt.plot(
    c["time"],
    c["speed"],
    label="Nominal + CBF"
)

plt.plot(
    c["time"],
    c["lead_speed"],
    label="Lead vehicle"
)

plt.axvline(
    collision_time,
    linestyle=":",
    label="Nominal collision"
)

plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")

plt.title(
    f"Speed During CBF Intervention - Trajectory {traj_id}"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# CONTROL COMPARISON
# ============================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    c["time"],
    c["u_nom"],
    label="Nominal acceleration"
)

plt.plot(
    c["time"],
    c["u_cmd"],
    label="CBF-filtered acceleration"
)

plt.axvline(
    collision_time,
    linestyle=":",
    label="Nominal collision"
)

plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s²)")

plt.title(
    f"CBF Safety Intervention - Trajectory {traj_id}"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# PRINT EVENT DETAILS
# ============================================================

print("\n======================================")
print("EVENT DETAILS")
print("======================================")

print(
    "Trajectory:",
    traj_id
)

print(
    "Initial gap:",
    selected["Initial_gap"],
    "m"
)

print(
    "Nominal collision time:",
    collision_time,
    "s"
)

print(
    "Minimum nominal gap:",
    a["gap"].min(),
    "m"
)

print(
    "Minimum CBF gap:",
    c["gap"].min(),
    "m"
)

print(
    "Minimum CBF TTC:",
    c["ttc"].replace(
        np.inf,
        np.nan
    ).min(),
    "s"
)

print(
    "Maximum nominal braking:",
    a["u_nom"].min(),
    "m/s²"
)

print(
    "Maximum CBF braking:",
    c["u_cmd"].min(),
    "m/s²"
)