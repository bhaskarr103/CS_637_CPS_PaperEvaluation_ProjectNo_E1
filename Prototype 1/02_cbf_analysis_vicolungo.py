import pandas as pd
import numpy as np

# ============================================================
# CBF PARAMETERS FROM THE PAPER
# ============================================================

T_MIN = 2.0       # minimum time gap [s]
D_MIN = 15.0      # minimum distance [m]
K_CBF = 0.1       # CBF gain


# ============================================================
# LOAD DATA
# ============================================================

FILE = r"E:\CPS Project\Prototype 2\Vicolungo.csv"

df = pd.read_csv(FILE)

print("Dataset loaded")
print("Rows:", len(df))
print("Trajectories:", df["Trajectory_ID"].nunique())


# ============================================================
# CBF SAFETY FUNCTION
# ============================================================

# Spatial_Gap = distance between LV and FAV
# Speed_FAV   = ego/following vehicle speed
#
# h >= 0  -> safe according to CBF condition
# h < 0   -> safety condition is violated

df["h"] = (
    df["Spatial_Gap"]
    - (T_MIN * df["Speed_FAV"] + D_MIN)
)


# ============================================================
# RELATIVE SPEED
# ============================================================

# Speed_Diff = Speed_LV - Speed_FAV
#
# Negative -> FAV is closing in on LV
# Positive -> FAV is moving slower relative to LV / gap opening

df["delta_v"] = df["Speed_Diff"]


# ============================================================
# CBF CONTROL LIMIT
# ============================================================

df["u_cbf"] = (
    df["delta_v"] + K_CBF * df["h"]
) / T_MIN


# ============================================================
# OBSERVED h DOT
# ============================================================

# h_dot = Delta_v - T_MIN * acceleration
#
# Here we use measured FAV acceleration.

df["h_dot"] = (
    df["delta_v"]
    - T_MIN * df["Acc_FAV"]
)


# ============================================================
# TIME GAP
# ============================================================

df["time_gap"] = (
    df["Spatial_Gap"] / df["Speed_FAV"]
)


# ============================================================
# BASIC RESULTS
# ============================================================

print("\n==============================")
print("CBF ANALYSIS")
print("==============================")

print("\nSafety function h:")
print("Minimum:", df["h"].min())
print("Maximum:", df["h"].max())
print("Mean   :", df["h"].mean())
print("Median :", df["h"].median())


# ============================================================
# SAFETY VIOLATIONS
# ============================================================

unsafe = df["h"] < 0

print("\nCBF safety violations:")
print("Unsafe samples:", unsafe.sum())
print("Unsafe %      :", unsafe.mean() * 100)


# ============================================================
# RECOVERY BEHAVIOR
# ============================================================

unsafe_df = df[unsafe]

recovering = unsafe_df["h_dot"] > 0
getting_worse = unsafe_df["h_dot"] < 0

print("\nWhen h < 0:")

print(
    "Recovering (h_dot > 0):",
    recovering.sum(),
    f"({recovering.mean() * 100:.2f}%)"
)

print(
    "Getting worse (h_dot < 0):",
    getting_worse.sum(),
    f"({getting_worse.mean() * 100:.2f}%)"
)


# ============================================================
# CLOSE GAP STATISTICS
# ============================================================

print("\n==============================")
print("SPATIAL GAP")
print("==============================")

for threshold in [1, 2, 5, 10, 15, 20, 30]:

    count = (df["Spatial_Gap"] < threshold).sum()

    print(
        f"Gap < {threshold:2} m : "
        f"{count:6} samples "
        f"({count / len(df) * 100:.2f}%)"
    )


# ============================================================
# TTC
# ============================================================

closing = df["delta_v"] < 0

df["TTC"] = np.inf

df.loc[closing, "TTC"] = (
    df.loc[closing, "Spatial_Gap"]
    / (-df.loc[closing, "delta_v"])
)

print("\n==============================")
print("TTC")
print("==============================")

print("Minimum TTC:", df["TTC"].min())

for threshold in [1, 2, 3, 5, 10]:

    count = (df["TTC"] < threshold).sum()

    print(
        f"TTC < {threshold:2} s : "
        f"{count:6} samples "
        f"({count / len(df) * 100:.2f}%)"
    )


# ============================================================
# CBF CONTROL LIMIT
# ============================================================

print("\n==============================")
print("CBF CONTROL LIMIT")
print("==============================")

print("Minimum u_CBF:", df["u_cbf"].min())
print("Maximum u_CBF:", df["u_cbf"].max())
print("Mean u_CBF   :", df["u_cbf"].mean())


# ============================================================
# MOST CRITICAL STATES
# ============================================================

print("\n==============================")
print("MOST CRITICAL STATES")
print("==============================")

critical = df.nsmallest(10, "h")

print(
    critical[
        [
            "Trajectory_ID",
            "Time_Index",
            "Spatial_Gap",
            "Speed_FAV",
            "Speed_LV",
            "delta_v",
            "h",
            "h_dot",
            "u_cbf",
            "TTC",
        ]
    ].to_string(index=False)
)