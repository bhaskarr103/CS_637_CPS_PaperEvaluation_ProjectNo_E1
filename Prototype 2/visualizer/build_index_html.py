"""build_index_html.py - Compiles index.html for Prototype 2 visualizer.

Embeds vicolungo_trajectories.json directly into index.html to guarantee
instant, offline execution via file:/// with zero CORS issues.
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
JSON_FILE = ROOT / "vicolungo_trajectories.json"
HTML_OUT = ROOT / "index.html"


def build():
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        traj_json = f.read()

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Autonomous CBF Car-Following Simulation — Prototype 2</title>
  <!-- Tailwind CSS via allowlisted gstatic CDN -->
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <link rel="stylesheet" href="styles.css">
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased p-3 md:p-6">

  <!-- Main Simulator Container -->
  <div class="max-w-7xl mx-auto space-y-4">

    <!-- Header & Metadata Banner -->
    <header class="bg-slate-900/90 backdrop-blur-md border border-slate-800 rounded-2xl p-4 md:p-5 shadow-2xl flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div>
        <div class="flex items-center gap-2 mb-1.5 flex-wrap">
          <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-blue-900/60 text-blue-300 border border-blue-700/50">
            PROTOTYPE 2 — ICCPS 2025 RESEARCH
          </span>
          <span id="badge-category" class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-purple-900/60 text-purple-300 border border-purple-700/50">
            [SYNTHETIC DEMONSTRATION]
          </span>
          <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-emerald-900/60 text-emerald-300 border border-emerald-700/50">
            100% EXACT PROTOTYPE 2 CONTROLLER MATH
          </span>
        </div>
        <h1 class="text-xl md:text-2xl font-bold text-white tracking-tight">
          Autonomous CBF Car-Following Simulator
        </h1>
        <p class="text-slate-400 text-xs md:text-sm mt-0.5">
          Lightweight 2.5D Autonomous Car Simulation • Gunter, Nice, Bunting, Sprinkle, Work (ICCPS 2025)
        </p>
      </div>

      <!-- Live Supervisory State Indicator -->
      <div id="status-pill" class="flex items-center gap-3 bg-slate-950/80 border border-slate-800 rounded-xl px-4 py-2.5 self-start md:self-auto shadow-inner">
        <div id="status-dot" class="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></div>
        <div>
          <div class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Supervisory State</div>
          <div id="status-text" class="text-sm font-semibold text-emerald-400">SAFE (h ≥ 5m)</div>
        </div>
      </div>
    </header>

    <!-- Interactive Simulation Controls & Toolbar -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3.5">
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3.5 items-end">
        
        <!-- Mode Selector: Synthetic vs Vicolungo -->
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1">Simulation Mode</label>
          <select id="select-sim-mode" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-blue-500 transition">
            <option value="synthetic" selected>1. Synthetic Demonstration</option>
            <option value="vicolungo">2. Vicolungo Real Lead Replay</option>
          </select>
        </div>

        <!-- Scenario Selector (Synthetic Mode) -->
        <div id="scenario-select-container">
          <label class="block text-xs font-semibold text-slate-300 mb-1">Synthetic Scenario</label>
          <select id="select-scenario" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-blue-500 transition">
            <option value="cruising">1. Normal Car Following</option>
            <option value="lead_accel">2. Lead Vehicle Acceleration</option>
            <option value="lead_decel">3. Moderate Lead Deceleration</option>
            <option value="emergency_braking" selected>4. Sudden Emergency Hard Braking</option>
            <option value="closing_speed_stress">5. Closing-Speed Stress Test</option>
          </select>
        </div>

        <!-- Trajectory Selector (Vicolungo Mode - Hidden initially) -->
        <div id="trajectory-select-container" class="hidden">
          <label class="block text-xs font-semibold text-slate-300 mb-1">Vicolungo Trajectory</label>
          <select id="select-trajectory" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-blue-500 transition">
            <option value="16" selected>Trajectory 16 — Safety Recovery / Actuator-Limit Stress Case (h0 = -37.05m &lt; 0)</option>
            <option value="55">Trajectory 55 — Forward-Invariance Demonstration (h0 = +73.30m &gt; 0)</option>
            <option value="1">Trajectory 1 — Highway Headway Deficit Recovery (h0 = -34.10m &lt; 0)</option>
            <option value="20">Trajectory 20 — Ultra-Tight Spacing with Lead Opening (s0 = 0.88m)</option>
          </select>
        </div>

        <!-- Controller Comparison Mode (Default: Side-by-Side Dual Comparison) -->
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1">Controller Setup</label>
          <select id="select-controller" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-100 focus:outline-none focus:border-blue-500 transition">
            <option value="comparison" selected>★ Dual Comparison (Side-by-Side: Baseline vs CBF)</option>
            <option value="cbf">ACC + CBF (Safety Supervised)</option>
            <option value="baseline">Baseline ACC (No Barrier Filter)</option>
          </select>
        </div>

        <!-- Interactive Emergency Brake Trigger Button -->
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1">Live Disturbance</label>
          <button id="btn-emergency-brake" class="w-full bg-red-700 hover:bg-red-600 text-white font-bold py-2 px-3 rounded-lg text-xs transition flex items-center justify-center gap-1.5 shadow-lg active:scale-95">
            <span>🚨</span> <span>EMERGENCY BRAKE!</span>
          </button>
        </div>

        <!-- Playback Controls -->
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1">Playback</label>
          <div class="flex items-center gap-1.5">
            <button id="btn-play" class="flex-1 bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2 px-2.5 rounded-lg text-xs transition flex items-center justify-center gap-1 shadow">
              <span id="play-icon">▶</span> <span id="play-label">Play</span>
            </button>
            <button id="btn-step" class="bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold py-2 px-2.5 rounded-lg text-xs transition">
              Step ❯
            </button>
            <button id="btn-reset" class="bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold py-2 px-2.5 rounded-lg text-xs transition">
              Reset
            </button>
            <select id="select-speed" class="bg-slate-950 border border-slate-700 rounded-lg px-1.5 py-2 text-xs text-slate-200">
              <option value="0.25">0.25x</option>
              <option value="0.5">0.5x</option>
              <option value="1.0" selected>1x</option>
              <option value="2.0">2x</option>
            </select>
          </div>
        </div>

      </div>

      <!-- Timeline Scrubber Bar -->
      <div class="pt-1 flex items-center gap-3">
        <span id="time-display" class="font-mono text-xs text-blue-400 font-semibold w-16">0.00s</span>
        <input id="timeline-slider" type="range" min="0" max="250" value="0" step="1" class="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer">
        <span id="duration-display" class="font-mono text-xs text-slate-500 w-16 text-right">25.0s</span>
      </div>
    </div>

    <!-- Active Scenario Description Banner -->
    <div class="bg-slate-900/60 border border-slate-800/80 rounded-xl px-4 py-2.5 text-xs text-slate-300 flex items-center justify-between flex-wrap gap-2">
      <div class="flex items-center gap-2">
        <span class="font-bold text-white uppercase tracking-wider" id="scenario-title">Scenario Title</span>
        <span class="text-slate-500">•</span>
        <span class="text-slate-400" id="scenario-desc">Scenario Description</span>
      </div>
      <div class="text-[11px] font-mono text-slate-400 flex items-center gap-3">
        <span>T_MIN = 2.0s</span>
        <span>D_MIN = 15m</span>
        <span>k_CBF = 0.1</span>
        <span>A_MIN = -4.0 m/s²</span>
        <span>JERK_MAX = 1.5 m/s³</span>
      </div>
    </div>

    <!-- 2.5D Autonomous Driving Highway Track Canvas -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-2xl space-y-2.5">
      <div class="flex items-center justify-between text-xs flex-wrap gap-2">
        <div class="font-semibold text-slate-300 flex items-center gap-2">
          <span>Autonomous Vehicle Test Track (Top-Down 2.5D View • Scale: 1m ≈ 4.8px)</span>
        </div>
        <div class="flex items-center gap-4 text-xs font-medium flex-wrap">
          <div class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-red-500 inline-block shadow-sm"></span> <span class="text-slate-400">Lead Vehicle</span></div>
          <div class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-blue-500 inline-block shadow-sm"></span> <span class="text-slate-400">AV (ACC + CBF)</span></div>
          <div class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-amber-500 inline-block shadow-sm"></span> <span class="text-slate-400">AV (Baseline ACC)</span></div>
          <div class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-emerald-500/40 border border-emerald-500 inline-block"></span> <span class="text-slate-400">Safe Envelope s_safe</span></div>
        </div>
      </div>

      <!-- Parallel Counterfactual Banner -->
      <div id="counterfactual-banner" class="bg-blue-950/40 border border-blue-800/50 rounded-xl px-3.5 py-2 flex items-center justify-between text-xs text-blue-200">
        <div class="flex items-center gap-2">
          <span class="font-bold text-blue-400 uppercase tracking-wide">Parallel Counterfactual Comparison:</span>
          <span class="text-slate-300">Track 1 (Baseline ACC) and Track 2 (ACC+CBF) run simultaneously under identical lead disturbance and initial state. Non-interacting parallel tracks for direct visual contrast.</span>
        </div>
        <span class="font-mono text-[10px] text-blue-400/80 uppercase font-semibold">Side-by-Side Verification</span>
      </div>

      <!-- Canvas Element -->
      <div class="relative w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950 canvas-container">
        <canvas id="canvas-road" width="1200" height="280" class="w-full block"></canvas>

        <!-- Collision Alert Overlay -->
        <div id="crash-overlay" class="absolute inset-0 bg-red-950/85 backdrop-blur-sm flex flex-col items-center justify-center gap-2 hidden pointer-events-none">
          <div class="text-3xl font-black text-red-500 tracking-wider animate-bounce">💥 CRITICAL COLLISION (s ≤ 0m)</div>
          <p id="crash-desc" class="text-sm text-red-200">The vehicle penetrated the bumper gap completely.</p>
        </div>
      </div>
    </div>

    <!-- Live Telemetry Dashboard (4 Metric Cards) -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-3.5">
      
      <!-- Card 1: Safety Margin h -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-3.5 shadow-lg">
        <div class="text-slate-400 text-xs font-semibold mb-1 flex items-center justify-between">
          <span>Barrier Margin h(t)</span>
          <span class="text-[10px] text-slate-500">h = s - (2v + 15)</span>
        </div>
        <div class="flex items-baseline gap-1">
          <span id="metric-h" class="metric-value text-2xl font-bold text-emerald-400">+15.20</span>
          <span class="text-xs text-slate-500">m</span>
        </div>
        <div class="w-full bg-slate-800 h-1.5 rounded-full mt-2.5 overflow-hidden">
          <div id="bar-h" class="bg-emerald-500 h-full transition-all duration-75" style="width: 70%;"></div>
        </div>
      </div>

      <!-- Card 2: Bumper Gap s -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-3.5 shadow-lg">
        <div class="text-slate-400 text-xs font-semibold mb-1 flex items-center justify-between">
          <span>Physical Headway s(t)</span>
          <span id="metric-req-gap" class="text-[10px] font-mono text-slate-400">Req: 59.0m</span>
        </div>
        <div class="flex items-baseline gap-1">
          <span id="metric-gap" class="metric-value text-2xl font-bold text-blue-400">74.2</span>
          <span class="text-xs text-slate-500">m</span>
        </div>
        <div class="w-full bg-slate-800 h-1.5 rounded-full mt-2.5 overflow-hidden">
          <div id="bar-gap" class="bg-blue-500 h-full transition-all duration-75" style="width: 65%;"></div>
        </div>
      </div>

      <!-- Card 3: Velocities -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-3.5 shadow-lg">
        <div class="text-slate-400 text-xs font-semibold mb-1 flex items-center justify-between">
          <span>Speeds (Lead / AV)</span>
          <span id="metric-delta-v" class="text-[10px] font-mono text-slate-400">Δv: 0.0 m/s</span>
        </div>
        <div class="flex items-baseline gap-2">
          <span id="metric-lead-speed" class="text-sm font-mono text-red-400 font-bold">22.0 m/s</span>
          <span class="text-xs text-slate-500">/</span>
          <span id="metric-ego-speed" class="text-sm font-mono text-white font-bold">22.0 m/s</span>
        </div>
        <div class="text-[10px] text-slate-500 mt-2 flex justify-between">
          <span>D_DES = 5.0m</span>
          <span>T_DES = 1.5s</span>
        </div>
      </div>

      <!-- Card 4: Accelerations -->
      <div class="bg-slate-900 border border-slate-800 rounded-xl p-3.5 shadow-lg">
        <div class="text-slate-400 text-xs font-semibold mb-1 flex items-center justify-between">
          <span>Control Accelerations</span>
          <div class="flex items-center gap-1">
            <span id="metric-intervention-badge" class="px-1.5 py-0.5 text-[9px] font-bold rounded bg-slate-800 text-slate-400">NO INTERV</span>
            <span id="metric-infeasible-badge" class="px-1.5 py-0.5 text-[9px] font-bold rounded bg-slate-800 text-slate-400">FEASIBLE</span>
          </div>
        </div>
        <div class="flex items-baseline gap-2">
          <span class="text-xs text-slate-400">Applied:</span>
          <span id="metric-u-applied" class="text-base font-bold font-mono text-emerald-400">+0.00 m/s²</span>
        </div>
        <div class="text-[10px] text-slate-500 mt-1 flex justify-between">
          <span id="metric-u-nom">u_nom: +0.00</span>
          <span id="metric-u-cbf">u_CBF: +1.20</span>
        </div>
      </div>

    </div>

    <!-- Synchronized Telemetry Charts (4 Subplots with Moving Cursor) -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3.5">
      <div class="flex items-center justify-between flex-wrap gap-2">
        <h2 class="text-xs font-bold text-white uppercase tracking-wider flex items-center gap-2">
          <span>Synchronized Telemetry Plots</span>
          <span class="text-xs font-normal text-slate-400 normal-case">• Vertical red line tracks current simulation frame</span>
        </h2>
        <div class="flex items-center gap-3 text-xs font-mono text-[11px]">
          <span class="text-blue-400">■ ACC+CBF Gap</span>
          <span class="text-amber-400">■ Baseline Gap</span>
          <span class="text-emerald-400">■ Safe Margin h</span>
          <span class="text-red-400">■ Lead Speed</span>
        </div>
      </div>

      <!-- 4 Plot Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-3.5">
        
        <!-- Plot 1: Headway Gap -->
        <div class="bg-slate-950 border border-slate-800/80 rounded-xl p-3 relative">
          <div class="text-xs font-semibold text-slate-300 mb-1 flex justify-between">
            <span>1. Headway Gap s(t) vs Safe Gap s_safe(t)</span>
            <span class="text-[10px] text-slate-500">D_min = 15m, D_des = 5m</span>
          </div>
          <canvas id="chart-gap" width="550" height="140" class="w-full block"></canvas>
        </div>

        <!-- Plot 2: Safety Margin h(t) -->
        <div class="bg-slate-950 border border-slate-800/80 rounded-xl p-3 relative">
          <div class="text-xs font-semibold text-slate-300 mb-1 flex justify-between">
            <span>2. Safety Barrier Function h(t) = s - (2v + 15)</span>
            <span class="text-[10px] text-slate-500">Safety Boundary h = 0</span>
          </div>
          <canvas id="chart-h" width="550" height="140" class="w-full block"></canvas>
        </div>

        <!-- Plot 3: Velocities -->
        <div class="bg-slate-950 border border-slate-800/80 rounded-xl p-3 relative">
          <div class="text-xs font-semibold text-slate-300 mb-1 flex justify-between">
            <span>3. Vehicle Velocities (Lead vs Follower AV)</span>
            <span class="text-[10px] text-slate-500">m/s</span>
          </div>
          <canvas id="chart-speed" width="550" height="140" class="w-full block"></canvas>
        </div>

        <!-- Plot 4: Accelerations -->
        <div class="bg-slate-950 border border-slate-800/80 rounded-xl p-3 relative">
          <div class="text-xs font-semibold text-slate-300 mb-1 flex justify-between">
            <span>4. Accelerations: u_nom vs u_CBF vs Applied</span>
            <span class="text-[10px] text-slate-500">Limits: [-4.0, +2.0] m/s²</span>
          </div>
          <canvas id="chart-accel" width="550" height="140" class="w-full block"></canvas>
        </div>

      </div>
    </div>

    <!-- Educational Deep-Dive Callouts -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
      <h3 class="text-xs font-bold text-white uppercase tracking-wider">Prototype 2 Architecture & Control Theory</h3>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-3.5 text-xs text-slate-300">
        
        <div class="bg-slate-950 border border-slate-800 rounded-xl p-3 space-y-1">
          <div class="font-bold text-amber-400">1. Nominal ACC Spacing Policy</div>
          <p class="text-slate-400 leading-relaxed text-[11px]">
            The unconstrained ACC controller attempts to track desired headway $s_\\text{des} = 5.0 + 1.5 v_\\text{ego}$:
            <br>
            $u_\\text{nom} = 0.20(s - s_\\text{des}) + 0.80(v_\\text{lead} - v_\\text{ego})$.
            <br>
            It does not guarantee non-negative headway during abrupt lead braking.
          </p>
        </div>

        <div class="bg-slate-950 border border-slate-800 rounded-xl p-3 space-y-1">
          <div class="font-bold text-blue-400">2. Control Barrier Function (CBF)</div>
          <p class="text-slate-400 leading-relaxed text-[11px]">
            Safety set $\\mathcal{C} = \\{x \\mid h(x) \\ge 0\\}$ with $h = s - (2.0 v_\\text{ego} + 15.0)$.
            <br>
            The barrier constraint enforces:
            <br>
            $u_\\text{CBF} \\le (0.1 h + \\Delta v) / 2.0$.
            <br>
            Supervisor enforces $u_\\text{cmd} = \\min(u_\\text{nom}, u_\\text{CBF})$.
          </p>
        </div>

        <div class="bg-slate-950 border border-slate-800 rounded-xl p-3 space-y-1">
          <div class="font-bold text-emerald-400">3. Physical Actuator & Jerk Saturation</div>
          <p class="text-slate-400 leading-relaxed text-[11px]">
            Acceleration is clipped to $[-4.0, +2.0]\\,\\text{m/s}^2$. Rate of acceleration change is bounded by $|da/dt| \\le 1.5\\,\\text{m/s}^3$ (jerk limitation), creating physical lag during emergency braking.
          </p>
        </div>

      </div>

      <!-- Defensible Scientific Framing Callout -->
      <div class="p-3.5 rounded-xl bg-slate-950/80 border border-slate-800 text-xs text-slate-300 flex items-start gap-3">
        <span class="text-amber-400 text-base">⚖️</span>
        <div class="space-y-1">
          <div class="font-bold text-white text-[12px]">Defensible Scientific Framing & Boundary Conditions:</div>
          <p class="text-slate-400 leading-relaxed text-[11px]">
            The Control Barrier Function (CBF) imposes a mathematically defined forward-invariance constraint on the nominal controller, subject to model fidelity, discrete-time sampling ($\Delta t = 0.1\,\text{s}$), and physical actuator limits ($a \in [-4.0, +2.0]\,\text{m/s}^2$, $|\dot{a}| \le 1.5\,\text{m/s}^3$).
            CBF theory guarantees forward invariance ($h(t) \ge 0$) <em>if and only if</em> the initial state is safe ($h(0) \ge 0$) and the required safety control input remains within the admissible actuator set $\mathcal{U}$. When $x_0 \notin \mathcal{C}$ (e.g., Trajectory 16) or required braking exceeds $-4.0\,\text{m/s}^2$, the constraint becomes infeasible ($u_\text{CBF} < a_\min$) and the supervisor triggers actuator saturation recovery.
          </p>
        </div>
      </div>
    </div>

  </div>

  <!-- EMBEDDED VICOLUNGO BENCHMARK DATASET -->
  <script>
    window.EMBEDDED_VICOLUNGO = """ + traj_json + """;
  </script>

  <!-- CORE SIMULATOR SCRIPT -->
  <script src="app.js"></script>

</body>
</html>
"""

    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Successfully compiled {HTML_OUT.name} ({HTML_OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build()
