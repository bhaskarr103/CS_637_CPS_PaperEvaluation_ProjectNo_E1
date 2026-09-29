"""One-Click Launcher for the Prototype 2 Autonomous CBF Car-Following Visualizer.

Opens index.html in the default system browser.
Zero external web server dependencies required.
"""

import webbrowser
from pathlib import Path

def main():
    viz_dir = Path(__file__).resolve().parent
    if (viz_dir / "visualizer" / "index.html").exists():
        html_file = viz_dir / "visualizer" / "index.html"
    elif (viz_dir / "index.html").exists():
        html_file = viz_dir / "index.html"
    else:
        print("Error: index.html not found.")
        return

    uri = html_file.resolve().as_uri()
    print("=" * 70)
    print("LAUNCHING PROTOTYPE 2 AUTONOMOUS CBF CAR-FOLLOWING SIMULATION")
    print(f"URI: {uri}")
    print("=" * 70)
    print("\nFeatures available:")
    print("  • Mode 1: Synthetic Demonstration (Cruising, Accel, Decel, Emergency Braking, Stress)")
    print("  • Mode 2: Vicolungo Replay (Trajectory 16 stress case, Trajectory 55 invariance case)")
    print("  • Dual Comparison Mode: Watch Baseline ACC vs ACC+CBF side-by-side on the road!")
    print("  • Interactive 'EMERGENCY BRAKE!' button")
    print("  • Exact Prototype 2 equations (T_min=2.0s, D_min=15m, A_min=-4.0m/s², Jerk=1.5m/s³)")
    print("  • Synchronized moving-cursor telemetry charts (Gap, h, Speeds, Accelerations)\n")

    webbrowser.open(uri)
    print("Visualizer opened in default web browser.")

if __name__ == "__main__":
    main()
