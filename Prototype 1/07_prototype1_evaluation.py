import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FILES
# ============================================================

ACC_FILE = r"E:\CPS Project\Prototype 2\paper_style_nominal.csv"
CBF_FILE = r"E:\CPS Project\Prototype 2\paper_style_cbf.csv"
SUMMARY_FILE = r"E:\CPS Project\Prototype 2\paper_style_summary.csv"


# ============================================================
# PARAMETERS
# ============================================================

COLLISION_GAP = 0.5


# ============================================================
# LOAD
# ============================================================

acc = pd.read_csv(ACC_FILE)
cbf = pd.read_csv(CBF_FILE)
summary = pd.read_csv(SUMMARY_FILE)

print("======================================")
print("PROTOTYPE 1 EVALUATION")
print("======================================")

print("ACC samples:", len(acc))
print("CBF samples:", len(cbf))
print("Trajectories:", summary["Trajectory_ID"].nunique())


# ============================================================
# PER-TRAJECTORY METRICS
# ============================================================

metrics = []


for traj_id in sorted(
    summary["Trajectory_ID"].unique()
):

    a = acc[
        acc["Trajectory_ID"] == traj_id
    ].copy()

    c = cbf[
        cbf["Trajectory_ID"] == traj_id
    ].copy()

    if len(a) == 0 or len(c) == 0:
        continue

    # --------------------------------------------------------
    # Use common time interval
    #
    # This is important.
    #
    # If ACC crashes at 7.4 s, we don't compare its
    # 7.4-second trajectory against 60 seconds of CBF data.
    # --------------------------------------------------------

    end_time = min(
        a["time"].max(),
        c["time"].max()
    )

    a = a[
        a["time"] <= end_time
    ]

    c = c[
        c["time"] <= end_time
    ]

    # --------------------------------------------------------
    # Collision
    # --------------------------------------------------------

    acc_collision = (
        a["gap"] <= COLLISION_GAP
    ).any()

    cbf_collision = (
        c["gap"] <= COLLISION_GAP
    ).any()

    # --------------------------------------------------------
    # Minimum gap
    # --------------------------------------------------------

    acc_min_gap = a["gap"].min()
    cbf_min_gap = c["gap"].min()

    # --------------------------------------------------------
    # TTC
    # --------------------------------------------------------

    acc_ttc = (
        a["ttc"]
        .replace(np.inf, np.nan)
        .dropna()
    )

    cbf_ttc = (
        c["ttc"]
        .replace(np.inf, np.nan)
        .dropna()
    )

    if len(acc_ttc) > 0:
        acc_min_ttc = acc_ttc.min()
    else:
        acc_min_ttc = np.nan

    if len(cbf_ttc) > 0:
        cbf_min_ttc = cbf_ttc.min()
    else:
        cbf_min_ttc = np.nan

    # --------------------------------------------------------
    # Average gap
    # --------------------------------------------------------

    acc_mean_gap = a["gap"].mean()
    cbf_mean_gap = c["gap"].mean()

    # --------------------------------------------------------
    # Average speed
    # --------------------------------------------------------

    acc_mean_speed = a["speed"].mean()
    cbf_mean_speed = c["speed"].mean()

    # --------------------------------------------------------
    # Speed tracking error
    #
    # Difference between ego and lead vehicle.
    # --------------------------------------------------------

    acc_speed_error = np.mean(
        np.abs(
            a["speed"].values
            -
            a["lead_speed"].values
        )
    )

    cbf_speed_error = np.mean(
        np.abs(
            c["speed"].values
            -
            c["lead_speed"].values
        )
    )

    # --------------------------------------------------------
    # CBF interventions
    # --------------------------------------------------------

    intervention = (
        c["u_cmd"]
        <
        c["u_nom"] - 1e-6
    )

    intervention_percent = (
        intervention.mean() * 100
    )

    # --------------------------------------------------------
    # CBF safety-set violation
    # --------------------------------------------------------

    acc_unsafe_percent = (
        (a["h"] < 0).mean() * 100
    )

    cbf_unsafe_percent = (
        (c["h"] < 0).mean() * 100
    )

    # --------------------------------------------------------
    # Infeasible CBF
    # --------------------------------------------------------

    infeasible = (
        c["u_cbf_raw"] < -8.0
    ).sum()

    # --------------------------------------------------------
    # Was collision prevented?
    # --------------------------------------------------------

    collision_prevented = (
        acc_collision
        and not cbf_collision
    )

    metrics.append({

        "Trajectory_ID":
            traj_id,

        "ACC_collision":
            acc_collision,

        "CBF_collision":
            cbf_collision,

        "Collision_prevented":
            collision_prevented,

        "ACC_min_gap":
            acc_min_gap,

        "CBF_min_gap":
            cbf_min_gap,

        "ACC_min_TTC":
            acc_min_ttc,

        "CBF_min_TTC":
            cbf_min_ttc,

        "ACC_mean_gap":
            acc_mean_gap,

        "CBF_mean_gap":
            cbf_mean_gap,

        "ACC_mean_speed":
            acc_mean_speed,

        "CBF_mean_speed":
            cbf_mean_speed,

        "ACC_speed_error":
            acc_speed_error,

        "CBF_speed_error":
            cbf_speed_error,

        "CBF_intervention_percent":
            intervention_percent,

        "ACC_unsafe_percent":
            acc_unsafe_percent,

        "CBF_unsafe_percent":
            cbf_unsafe_percent,

        "CBF_infeasible_samples":
            infeasible
    })


metrics = pd.DataFrame(metrics)


# ============================================================
# OVERALL SAFETY RESULTS
# ============================================================

print("\n======================================")
print("SAFETY RESULTS")
print("======================================")

print(
    "ACC collisions:",
    metrics["ACC_collision"].sum()
)

print(
    "CBF collisions:",
    metrics["CBF_collision"].sum()
)

print(
    "Collisions prevented by CBF:",
    metrics["Collision_prevented"].sum()
)


# ============================================================
# MINIMUM VALUES
# ============================================================

print("\n======================================")
print("MINIMUM SAFETY VALUES")
print("======================================")

print(
    "ACC minimum gap:",
    metrics["ACC_min_gap"].min(),
    "m"
)

print(
    "CBF minimum gap:",
    metrics["CBF_min_gap"].min(),
    "m"
)

print(
    "ACC minimum TTC:",
    metrics["ACC_min_TTC"].min(),
    "s"
)

print(
    "CBF minimum TTC:",
    metrics["CBF_min_TTC"].min(),
    "s"
)


# ============================================================
# PERFORMANCE
# ============================================================

print("\n======================================")
print("DRIVING PERFORMANCE")
print("======================================")

print(
    "ACC average gap:",
    metrics["ACC_mean_gap"].mean(),
    "m"
)

print(
    "CBF average gap:",
    metrics["CBF_mean_gap"].mean(),
    "m"
)

print(
    "ACC average speed:",
    metrics["ACC_mean_speed"].mean(),
    "m/s"
)

print(
    "CBF average speed:",
    metrics["CBF_mean_speed"].mean(),
    "m/s"
)

print(
    "ACC mean speed error:",
    metrics["ACC_speed_error"].mean(),
    "m/s"
)

print(
    "CBF mean speed error:",
    metrics["CBF_speed_error"].mean(),
    "m/s"
)


# ============================================================
# CBF BEHAVIOR
# ============================================================

print("\n======================================")
print("CBF BEHAVIOR")
print("======================================")

print(
    "Average intervention rate:",
    metrics["CBF_intervention_percent"].mean(),
    "%"
)

print(
    "Total infeasible CBF samples:",
    metrics["CBF_infeasible_samples"].sum()
)

print(
    "Average ACC unsafe percentage:",
    metrics["ACC_unsafe_percent"].mean(),
    "%"
)

print(
    "Average CBF unsafe percentage:",
    metrics["CBF_unsafe_percent"].mean(),
    "%"
)


# ============================================================
# IMPROVEMENT
# ============================================================

acc_collisions = (
    metrics["ACC_collision"].sum()
)

cbf_collisions = (
    metrics["CBF_collision"].sum()
)

if acc_collisions > 0:

    collision_reduction = (
        (acc_collisions - cbf_collisions)
        / acc_collisions
        * 100
    )

else:

    collision_reduction = 0


print("\n======================================")
print("CBF SAFETY IMPROVEMENT")
print("======================================")

print(
    "Collision reduction:",
    collision_reduction,
    "%"
)


# ============================================================
# TRAJECTORIES WHERE CBF HELPED
# ============================================================

helped = metrics[
    metrics["Collision_prevented"]
]

print("\n======================================")
print("TRAJECTORIES SAVED BY CBF")
print("======================================")

if len(helped) > 0:

    print(
        helped[
            [
                "Trajectory_ID",
                "ACC_min_gap",
                "CBF_min_gap",
                "ACC_min_TTC",
                "CBF_min_TTC"
            ]
        ].to_string(index=False)
    )

else:

    print("No collision-prevention cases found.")


# ============================================================
# BEST CBF IMPROVEMENTS
# ============================================================

metrics["gap_improvement"] = (
    metrics["CBF_min_gap"]
    -
    metrics["ACC_min_gap"]
)

best = metrics.sort_values(
    "gap_improvement",
    ascending=False
).head(10)


print("\n======================================")
print("TOP 10 GAP IMPROVEMENTS")
print("======================================")

print(
    best[
        [
            "Trajectory_ID",
            "ACC_min_gap",
            "CBF_min_gap",
            "gap_improvement",
            "CBF_intervention_percent"
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE COMPLETE TABLE
# ============================================================

output_file = (
    r"E:\CPS Project\Prototype 2"
    r"\prototype1_metrics.csv"
)

metrics.to_csv(
    output_file,
    index=False
)

print("\nDetailed metrics saved to:")
print(output_file)


# ============================================================
# SUMMARY BAR CHART
# ============================================================

labels = [
    "ACC only",
    "Nominal + CBF"
]

collision_values = [
    metrics["ACC_collision"].sum(),
    metrics["CBF_collision"].sum()
]

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    labels,
    collision_values
)

plt.ylabel("Number of trajectories")
plt.title("Collision Comparison")

plt.grid(
    axis="y"
)

plt.tight_layout()
plt.show()


# ============================================================
# MINIMUM GAP DISTRIBUTION
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.hist(
    metrics["ACC_min_gap"],
    bins=20,
    alpha=0.6,
    label="ACC only"
)

plt.hist(
    metrics["CBF_min_gap"],
    bins=20,
    alpha=0.6,
    label="Nominal + CBF"
)

plt.xlabel("Minimum gap (m)")
plt.ylabel("Number of trajectories")

plt.title(
    "Minimum Gap Distribution"
)

plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# INTERVENTION VS GAP IMPROVEMENT
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.scatter(
    metrics["CBF_intervention_percent"],
    metrics["gap_improvement"]
)

plt.xlabel(
    "CBF intervention rate (%)"
)

plt.ylabel(
    "Minimum-gap improvement (m)"
)

plt.title(
    "CBF Intervention vs Safety Improvement"
)

plt.grid(True)

plt.tight_layout()
plt.show()


print("\n======================================")
print("PROTOTYPE 1 COMPLETE")
print("======================================")