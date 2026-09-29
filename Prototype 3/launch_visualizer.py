"""One-Click Launcher for the CBF-RL Autonomous Car-Following Visualizer.

Opens visualizer.html in the system default web browser.
Zero external server dependencies required.
"""

import webbrowser
from pathlib import Path

def main():
    html_file = Path(__file__).parent / "visualizer.html"
    if not html_file.exists():
        print(f"Error: {html_file.name} not found. Please run build_visualizer_html.py first.")
        return

    file_uri = html_file.resolve().as_uri()
    print("=" * 70)
    print("LAUNCHING CBF-RL CAR-FOLLOWING INTERACTIVE VISUALIZER")
    print(f"URI: {file_uri}")
    print("=" * 70)
    print("\nFeatures available in browser:")
    print("  • 6 Scenarios: Cruising, Moderate Brake, Emergency Brake, Cut-In, Vicolungo Replays")
    print("  • 4 Ablations: Nominal RL, Filter Only, Reward Only, Dual CBF-RL")
    print("  • Deployment Toggle: Filter ON vs Filter OFF (Internalization Acid Test)")
    print("  • Synchronized moving-cursor telemetry charts (Gap, h, Velocities, Accelerations)")
    print("  • Keyboard shortcuts: Space (Play/Pause), Left/Right Arrows (Step)\n")

    webbrowser.open(file_uri)
    print("Visualizer opened in default web browser.")

if __name__ == "__main__":
    main()
