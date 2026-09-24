import pandas as pd

FILE = "Vicolungo.csv"

df = pd.read_csv(FILE)

print("\n===== BASIC INFORMATION =====")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== TRAJECTORIES =====")
print(df["Trajectory_ID"].nunique())
print(df["Trajectory_ID"].value_counts().sort_index())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== IMPORTANT VARIABLES =====")

for col in [
    "Speed_LV",
    "Speed_FAV",
    "Acc_LV",
    "Acc_FAV",
    "Spatial_Gap",
    "Spatial_Headway",
    "Speed_Diff"
]:
    if col in df.columns:
        print(f"\n{col}")
        print("  Min :", df[col].min())
        print("  Max :", df[col].max())
        print("  Mean:", df[col].mean())
        print("  Median:", df[col].median())

print("\n===== TIME INFORMATION =====")
print("Time min:", df["Time_Index"].min())
print("Time max:", df["Time_Index"].max())

dt = df.groupby("Trajectory_ID")["Time_Index"].diff().dropna()

print("Median sampling interval:", dt.median(), "s")
print("Approx sampling frequency:", 1 / dt.median(), "Hz")

print("\n===== CLOSING EVENTS =====")

# Speed_Diff = LV speed - FAV speed
# Negative => FAV is faster => closing the gap

closing = df[df["Speed_Diff"] < 0]

print("Closing samples:", len(closing))
print(
    "Maximum closing speed:",
    -closing["Speed_Diff"].min(),
    "m/s"
)

print("\n===== BASIC TTC =====")

closing = df[df["Speed_Diff"] < 0].copy()

# TTC = distance gap / closing speed
closing["TTC"] = (
    closing["Spatial_Gap"] /
    (-closing["Speed_Diff"])
)

print("Minimum TTC:", closing["TTC"].min(), "s")
print("Median TTC:", closing["TTC"].median(), "s")

print("\nTTC < 10 s:", (closing["TTC"] < 10).sum())
print("TTC < 5 s :", (closing["TTC"] < 5).sum())
print("TTC < 3 s :", (closing["TTC"] < 3).sum())


print("\n===== LOWEST TTC EVENTS =====")

lowest_ttc = closing.nsmallest(20, "TTC")

print(
    lowest_ttc[
        [
            "Trajectory_ID",
            "Time_Index",
            "ID_LV",
            "ID_FAV",
            "Speed_LV",
            "Speed_FAV",
            "Speed_Diff",
            "Spatial_Gap",
            "Spatial_Headway",
            "Acc_LV",
            "Acc_FAV",
            "TTC"
        ]
    ].to_string(index=False)
)


print("\n===== SMALLEST GAP EVENTS =====")

smallest_gap = df.nsmallest(20, "Spatial_Gap")

print(
    smallest_gap[
        [
            "Trajectory_ID",
            "Time_Index",
            "ID_LV",
            "ID_FAV",
            "Speed_LV",
            "Speed_FAV",
            "Speed_Diff",
            "Spatial_Gap",
            "Acc_LV",
            "Acc_FAV"
        ]
    ].to_string(index=False)
)


print("\n===== GAP DISTRIBUTION =====")

for threshold in [1, 2, 5, 10, 15, 20, 25, 30]:
    count = (df["Spatial_Gap"] < threshold).sum()
    percentage = count / len(df) * 100

    print(
        f"Gap < {threshold:2d} m : "
        f"{count:6d} samples "
        f"({percentage:.2f}%)"
    )


print("\n===== TTC DISTRIBUTION =====")

for threshold in [1, 2, 3, 5, 10, 15, 20]:
    count = (closing["TTC"] < threshold).sum()
    percentage = count / len(closing) * 100

    print(
        f"TTC < {threshold:2d} s : "
        f"{count:6d} samples "
        f"({percentage:.2f}% of closing samples)"
    )


print("\n===== CLOSING EVENTS WITH REASONABLE GAPS =====")

for threshold in [5, 10, 15, 20]:
    subset = closing[closing["Spatial_Gap"] < threshold]

    print(
        f"\nGap < {threshold} m:"
    )

    if len(subset) > 0:
        print("  Samples:", len(subset))
        print("  Minimum TTC:", subset["TTC"].min(), "s")
        print("  Maximum closing speed:",
              -subset["Speed_Diff"].min(), "m/s")

print("\n===== DONE =====")