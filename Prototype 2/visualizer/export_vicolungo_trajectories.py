"""Exports selected representative Vicolungo trajectories for the Prototype 2 visualizer.

Extracts:
- Trajectory 16: Primary research stress case (severe closing speed & actuator saturation)
- Trajectory 55: Textbook Forward Invariance case (starts safely at h0 = +73.3m)
- Trajectory 1: Highway headway deficit case (starts at h0 = -34.1m)
- Trajectory 20: Extreme initial proximity case (s0 = 0.88m)

Replays exact recorded Speed_LV[t] as exogenous lead vehicle speed.
Initial follower state (s0, v_ego0) is taken at t=0, and simulated closed-loop thereafter.
"""

import sys
import json
from pathlib import Path
import pandas as pd

# Paths
PROTO2_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROTO2_DIR / "Vicolungo.csv"
OUT_JSON = Path(__file__).resolve().parent / "vicolungo_trajectories.json"


def export_trajectories():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Vicolungo dataset not found at {CSV_PATH}")

    print(f"Loading {CSV_PATH.name}...")
    df = pd.read_csv(CSV_PATH)

    selected_tids = [16, 55, 1, 20]
    metadata = {
        16: {
            "title": "Vicolungo Trajectory 16 — Safety Recovery / Actuator-Limit Stress Case",
            "role": "Safety Recovery & Actuator Saturation Case",
            "desc": "Lead is much slower (19.05 m/s) than Ego (27.57 m/s) with a 33.09m gap. Starts outside the safe set (h0 = -37.05m < 0). Heavy closing speed (8.5 m/s) triggers CBF intervention and actuator saturation (-4.0 m/s²). Tests emergency recovery and saturation handling, NOT forward invariance.",
        },
        55: {
            "title": "Vicolungo Trajectory 55 — Forward-Invariance Demonstration",
            "role": "Forward Invariance Preservation",
            "desc": "Starts safely inside the safe set with large headway (s0 = 119.56m, h0 = +73.30m > 0). Demonstrates textbook continuous preservation of forward invariance (h(t) >= 0 for all t) under real highway lead speed fluctuations.",
        },
        1: {
            "title": "Vicolungo Trajectory 1 — Highway Headway Deficit Recovery",
            "role": "Safe Headway Recovery",
            "desc": "Real Italian highway vehicle tailgating at 116 km/h with 45.4m gap (h0 = -34.10m < 0). Tests graceful headway recovery back toward the safe set.",
        },
        20: {
            "title": "Vicolungo Trajectory 20 — Ultra-Tight Spacing with Lead Opening",
            "role": "Boundary Clamping & Opening Response",
            "desc": "Extremely close initial spacing (0.88m) but lead is pulling away faster. Tests controller response to positive delta-v.",
        },
    }


    exported = {}

    for tid in selected_tids:
        tdf = df[df["Trajectory_ID"] == tid].sort_values("Time_Index").reset_index(drop=True)
        s0 = float(tdf.loc[0, "Spatial_Gap"])
        v_ego0 = float(tdf.loc[0, "Speed_FAV"])
        v_lead0 = float(tdf.loc[0, "Speed_LV"])
        h0 = s0 - (2.0 * v_ego0 + 15.0)

        # Cap length at 400 steps (40.0s) for visualizer smoothness
        sub_len = min(400, len(tdf))
        lead_speeds = [round(float(v), 3) for v in tdf.loc[:sub_len - 1, "Speed_LV"]]

        exported[str(tid)] = {
            "traj_id": tid,
            "title": metadata[tid]["title"],
            "role": metadata[tid]["role"],
            "description": metadata[tid]["desc"],
            "s0": round(s0, 2),
            "v_ego0": round(v_ego0, 2),
            "v_lead0": round(v_lead0, 2),
            "h0": round(h0, 2),
            "dt": 0.1,
            "steps": len(lead_speeds),
            "duration_s": round(len(lead_speeds) * 0.1, 1),
            "speed_lv": lead_speeds,
        }
        print(f"  Exported Trajectory {tid:2d}: {len(lead_speeds)} steps ({len(lead_speeds)*0.1:.1f}s) | s0={s0:.2f}m, h0={h0:.2f}m")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(exported, f, indent=2)

    print(f"\nSuccessfully wrote {OUT_JSON.name} ({OUT_JSON.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    export_trajectories()
