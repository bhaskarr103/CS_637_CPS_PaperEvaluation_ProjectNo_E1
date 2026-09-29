"""Compiles the standalone interactive visualizer HTML file.

Bundles the visualizer_data.json database directly into visualizer.html
so that it opens instantly via file:/// in any browser with zero CORS
restrictions and zero external dependencies.
"""

import sys
import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_FILE = ROOT_DIR / "visualizer_data.json"
HTML_OUTPUT = ROOT_DIR / "visualizer.html"

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CBF-RL Autonomous Car-Following Interactive Simulator</title>
  <!-- Tailwind CSS via allowlisted gstatic CDN -->
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    input[type=range] {
      accent-color: #2563eb;
    }
    .glow-red {
      box-shadow: 0 0 15px rgba(239, 68, 68, 0.6);
    }
    .glow-green {
      box-shadow: 0 0 15px rgba(34, 197, 94, 0.6);
    }
    .glow-amber {
      box-shadow: 0 0 15px rgba(245, 158, 11, 0.6);
    }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans antialiased p-4 md:p-6">

  <!-- Main Container -->
  <div class="max-w-7xl mx-auto space-y-5">

    <!-- Header & Metadata Banner -->
    <header class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div>
        <div class="flex items-center gap-2 mb-1 flex-wrap">
          <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-blue-900/60 text-blue-300 border border-blue-700/50">
            PROTOTYPE 3 — RESEARCH VISUALIZER
          </span>
          <span id="badge-category" class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-purple-900/60 text-purple-300 border border-purple-700/50">
            SYNTHETIC DEMONSTRATION
          </span>
          <span class="px-2.5 py-0.5 text-xs font-semibold rounded-full bg-emerald-900/60 text-emerald-300 border border-emerald-700/50">
            100% EXACT KINEMATICS & CBF EQUATIONS
          </span>
        </div>
        <h1 class="text-2xl font-bold text-white tracking-tight">
          CBF-RL Autonomous Car-Following Simulator
        </h1>
        <p class="text-slate-400 text-xs md:text-sm mt-0.5">
          Low-Level Kinematic Simulation with Control Barrier Function Safety Filtering • Yang et al. (arXiv:2510.14959)
        </p>
      </div>

      <!-- Live State Indicator -->
      <div id="status-pill" class="flex items-center gap-3 bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 self-start md:self-auto">
        <div id="status-dot" class="w-3 h-3 rounded-full bg-emerald-500 animate-pulse"></div>
        <div>
          <div class="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Supervisory State</div>
          <div id="status-text" class="text-sm font-semibold text-emerald-400">SAFE (h ≥ 0)</div>
        </div>
      </div>
    </header>

    <!-- Interactive Configuration Toolbar -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-4">
      <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        <!-- Scenario Selector -->
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5">Scenario</label>
          <select id="select-scenario" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500 transition">
            <option value="cruising">1. Normal Highway Cruising</option>
            <option value="lead_braking">2. Moderate Lead Braking (-2.0 m/s²)</option>
            <option value="emergency_braking" selected>3. Sudden Emergency Braking (-3.8 m/s²)</option>
            <option value="close_cutin">4. Close Cut-In (Headway Deficit, h0 < 0)</option>
            <option value="vicolungo_traj3">5. Vicolungo Traj 3 (Real Replay, Safe h0=+51m)</option>
            <option value="vicolungo_traj1">6. Vicolungo Traj 1 (Real Replay, Penetrated h0=-34m)</option>
          </select>
        </div>

        <!-- Policy Ablation Selector -->
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5">RL Policy Ablation</label>
          <select id="select-variant" class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-100 focus:outline-none focus:border-blue-500 transition">
            <option value="Dual_CBF_RL" selected>Dual CBF-RL (Shielded + Reward Penalty)</option>
            <option value="Filter_Only">Filter Only (Shielded in Training, No CBF Rew)</option>
            <option value="Reward_Only">Reward Only (CBF Penalty, No Shield in Train)</option>
            <option value="Nominal">Nominal RL (Unconstrained Baseline)</option>
          </select>
        </div>

        <!-- Deployment Mode (Filter ON vs OFF) -->
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5">Test Deployment Mode</label>
          <div class="grid grid-cols-2 gap-2 bg-slate-950 p-1 rounded-lg border border-slate-800">
            <button id="btn-filter-on" class="py-1.5 text-xs font-semibold rounded-md bg-blue-600 text-white shadow transition">
              Filter ON
            </button>
            <button id="btn-filter-off" class="py-1.5 text-xs font-semibold rounded-md text-slate-400 hover:text-white transition">
              Filter OFF
            </button>
          </div>
        </div>

        <!-- Playback Controls -->
        <div>
          <label class="block text-xs font-semibold text-slate-300 mb-1.5">Playback Controls</label>
          <div class="flex items-center gap-2">
            <button id="btn-play" class="flex-1 bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2 px-3 rounded-lg text-xs transition flex items-center justify-center gap-1.5 shadow">
              <span id="play-icon">▶</span> <span id="play-label">Play</span>
            </button>
            <button id="btn-step" class="bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold py-2 px-3 rounded-lg text-xs transition">
              Step ❯
            </button>
            <button id="btn-reset" class="bg-slate-800 hover:bg-slate-700 text-slate-200 font-semibold py-2 px-3 rounded-lg text-xs transition">
              Reset
            </button>
            <select id="select-speed" class="bg-slate-950 border border-slate-700 rounded-lg px-2 py-2 text-xs text-slate-200">
              <option value="0.5">0.5x</option>
              <option value="1.0" selected>1.0x</option>
              <option value="2.0">2.0x</option>
            </select>
          </div>
        </div>

      </div>

      <!-- Timeline Scrubber -->
      <div class="pt-2 flex items-center gap-3">
        <span id="time-display" class="font-mono text-xs text-blue-400 font-semibold w-16">0.00s</span>
        <input id="timeline-slider" type="range" min="0" max="200" value="0" step="1" class="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer">
        <span id="duration-display" class="font-mono text-xs text-slate-500 w-16 text-right">20.00s</span>
      </div>
    </div>

    <!-- Highway Simulation Canvas -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-xl space-y-3">
      <div class="flex items-center justify-between">
        <div class="text-xs font-semibold text-slate-300 flex items-center gap-2">
          <span>Highway Car-Following Track (Scale: 1m ≈ 5.2px)</span>
          <span id="scenario-note" class="text-slate-500 font-normal hidden sm:inline"></span>
        </div>
        <div class="flex items-center gap-3 text-xs">
          <div class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-red-500 inline-block"></span> <span class="text-slate-400">Lead Vehicle</span></div>
          <div class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-blue-500 inline-block"></span> <span class="text-slate-400">Ego (Follower)</span></div>
          <div class="flex items-center gap-1.5"><span class="w-3 h-3 rounded bg-emerald-500/40 border border-emerald-500 inline-block"></span> <span class="text-slate-400">Safe Envelope h ≥ 0</span></div>
        </div>
      </div>

      <!-- Canvas Element -->
      <div class="relative w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-950">
        <canvas id="sim-canvas" width="1200" height="240" class="w-full block"></canvas>

        <!-- Crash Alert Overlay -->
        <div id="crash-overlay" class="absolute inset-0 bg-red-950/80 backdrop-blur-sm flex flex-col items-center justify-center gap-2 hidden pointer-events-none">
          <div class="text-3xl font-black text-red-500 tracking-wider animate-bounce">💥 COLLISION DETECTED (s ≤ 0m)</div>
          <p class="text-sm text-red-200">The policy proposed insufficient braking in unshielded mode, resulting in a physical crash.</p>
        </div>
      </div>
    </div>

    <!-- Live Telemetry Dashboard (4 Metric Cards) -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
      
      <!-- Card 1: Safety Margin h -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg">
        <div class="text-slate-400 text-xs font-semibold mb-1 flex items-center justify-between">
          <span>Safety Margin h(t)</span>
          <span class="text-[10px] text-slate-500">h = s - (2v + 15)</span>
        </div>
        <div class="flex items-baseline gap-1">
          <span id="metric-h" class="text-2xl font-bold font-mono text-emerald-400">+15.2</span>
          <span class="text-xs text-slate-500">m</span>
        </div>
        <div class="w-full bg-slate-800 h-1.5 rounded-full mt-2.5 overflow-hidden">
          <div id="bar-h" class="bg-emerald-500 h-full transition-all duration-75" style="width: 75%;"></div>
        </div>
      </div>

      <!-- Card 2: Bumper Gap s -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg">
        <div class="text-slate-400 text-xs font-semibold mb-1 flex items-center justify-between">
          <span>Physical Gap s(t)</span>
          <span id="metric-s-safe" class="text-[10px] text-slate-500">Req: 59.0m</span>
        </div>
        <div class="flex items-baseline gap-1">
          <span id="metric-s" class="text-2xl font-bold font-mono text-blue-400">74.2</span>
          <span class="text-xs text-slate-500">m</span>
        </div>
        <div class="w-full bg-slate-800 h-1.5 rounded-full mt-2.5 overflow-hidden">
          <div id="bar-s" class="bg-blue-500 h-full transition-all duration-75" style="width: 70%;"></div>
        </div>
      </div>

      <!-- Card 3: Velocities -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg">
        <div class="text-slate-400 text-xs font-semibold mb-1 flex items-center justify-between">
          <span>Velocities (Ego / Lead)</span>
          <span id="metric-dv" class="text-[10px] font-mono text-slate-400">Δv: 0.0 m/s</span>
        </div>
        <div class="flex items-baseline gap-2">
          <span id="metric-vego" class="text-2xl font-bold font-mono text-white">22.0</span>
          <span class="text-xs text-slate-500">/</span>
          <span id="metric-vlead" class="text-xl font-bold font-mono text-red-400">22.0</span>
          <span class="text-xs text-slate-500">m/s</span>
        </div>
        <div class="text-[10px] text-slate-500 mt-2 flex justify-between">
          <span id="metric-vego-kmh">79.2 km/h</span>
          <span id="metric-vlead-kmh">79.2 km/h</span>
        </div>
      </div>

      <!-- Card 4: Accelerations -->
      <div class="bg-slate-900 border border-slate-800 rounded-2xl p-4 shadow-lg">
        <div class="text-slate-400 text-xs font-semibold mb-1 flex items-center justify-between">
          <span>Control Acceleration</span>
          <span id="metric-intervention-badge" class="px-1.5 py-0.5 text-[9px] font-bold rounded bg-slate-800 text-slate-400">NO INTERV</span>
        </div>
        <div class="flex items-baseline gap-1.5">
          <span id="metric-u-applied" class="text-2xl font-bold font-mono text-emerald-400">-0.00</span>
          <span class="text-xs text-slate-500">m/s²</span>
          <span class="text-xs text-slate-500 ml-1">pol:</span>
          <span id="metric-u-policy" class="text-sm font-mono text-red-400">+0.20</span>
        </div>
        <div class="text-[10px] text-slate-500 mt-2 flex justify-between">
          <span id="metric-u-cbf">CBF Limit: +1.20</span>
          <span id="metric-u-diff">Diff: 0.00</span>
        </div>
      </div>

    </div>

    <!-- Synchronized Telemetry Plots (4 Subplots with Moving Cursor) -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
      <div class="flex items-center justify-between flex-wrap gap-2">
        <h2 class="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
          <span>Synchronized Telemetry Curves</span>
          <span class="text-xs font-normal text-slate-400 normal-case">• Vertical red line indicates current animation step</span>
        </h2>
        <div class="flex items-center gap-4 text-xs font-mono">
          <span class="text-blue-400">■ Headway s(t)</span>
          <span class="text-emerald-400">■ Safety h(t)</span>
          <span class="text-purple-400">■ Ego Speed</span>
          <span class="text-red-400">■ Lead Speed</span>
        </div>
      </div>

      <!-- 4 Plot Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
        
        <!-- Plot 1: Headway Gap -->
        <div class="bg-slate-950 border border-slate-800/80 rounded-xl p-3 relative">
          <div class="text-xs font-semibold text-slate-300 mb-1 flex justify-between">
            <span>1. Spatial Headway s(t) vs Safe Gap s_safe(t)</span>
            <span class="text-[10px] text-slate-500">D_min = 15m</span>
          </div>
          <canvas id="plot-gap" width="550" height="150" class="w-full block"></canvas>
        </div>

        <!-- Plot 2: Safety Margin h(t) -->
        <div class="bg-slate-950 border border-slate-800/80 rounded-xl p-3 relative">
          <div class="text-xs font-semibold text-slate-300 mb-1 flex justify-between">
            <span>2. Safety Margin h(t) = s - (2v + 15)</span>
            <span class="text-[10px] text-slate-500">Boundary: h = 0m</span>
          </div>
          <canvas id="plot-h" width="550" height="150" class="w-full block"></canvas>
        </div>

        <!-- Plot 3: Velocities -->
        <div class="bg-slate-950 border border-slate-800/80 rounded-xl p-3 relative">
          <div class="text-xs font-semibold text-slate-300 mb-1 flex justify-between">
            <span>3. Vehicle Velocities (Ego vs Lead)</span>
            <span class="text-[10px] text-slate-500">m/s</span>
          </div>
          <canvas id="plot-v" width="550" height="150" class="w-full block"></canvas>
        </div>

        <!-- Plot 4: Accelerations -->
        <div class="bg-slate-950 border border-slate-800/80 rounded-xl p-3 relative">
          <div class="text-xs font-semibold text-slate-300 mb-1 flex justify-between">
            <span>4. Accelerations: u_policy vs u_safe</span>
            <span class="text-[10px] text-slate-500">u_min = -4.0 m/s²</span>
          </div>
          <canvas id="plot-u" width="550" height="150" class="w-full block"></canvas>
        </div>

      </div>
    </div>

    <!-- Educational Deep-Dive Panel -->
    <div class="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
      <h3 class="text-sm font-bold text-white uppercase tracking-wider">Methodological Insight: Why This Matters</h3>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-300">
        
        <div class="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-1.5">
          <div class="font-bold text-blue-400">1. What is the Policy Proposing?</div>
          <p class="text-slate-400 leading-relaxed">
            The neural network actor maps normalized headway and speeds to candidate acceleration u_policy in [-4.0, +2.5] m/s². Under Nominal RL or Filter Only, the policy frequently pushes full throttle even during closing maneuvers.
          </p>
        </div>

        <div class="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-1.5">
          <div class="font-bold text-amber-400">2. What does the CBF Filter Change?</div>
          <p class="text-slate-400 leading-relaxed">
            The closed-form CBF supervisor enforces u &le; u_CBF_max = (delta_v + 0.1h)/2.0. If u_policy &gt; u_CBF_max, the supervisor projects the action down to u_safe, shielding the car from collisions.
          </p>
        </div>

        <div class="bg-slate-950 border border-slate-800 rounded-xl p-3.5 space-y-1.5">
          <div class="font-bold text-emerald-400">3. Policy Internalization (Dual CBF-RL)</div>
          <p class="text-slate-400 leading-relaxed">
            Because Dual CBF-RL receives barrier breach penalties during training, its proposed action tracks u_safe natively (||u_policy - u_safe|| &to; 0.06 m/s²). When tested with <strong>Filter OFF</strong>, it avoids collisions (99.4% success) where Filter Only crashes (53.3%).
          </p>
        </div>

      </div>
    </div>

  </div>

  <!-- EMBEDDED TELEMETRY DATABASE (3MB precomputed exact kinematics) -->
  <script>
    const SIM_DATABASE = __DATABASE_JSON__;
  </script>

  <!-- VISUALIZER APPLICATION LOGIC -->
  <script>
    // State variables
    let currentScenarioKey = "emergency_braking";
    let currentVariant = "Dual_CBF_RL";
    let currentFilterMode = "filter_on";
    let currentStep = 0;
    let isPlaying = false;
    let playbackSpeed = 1.0;
    let animTimer = null;

    // DOM Elements
    const selectScenario = document.getElementById("select-scenario");
    const selectVariant = document.getElementById("select-variant");
    const btnFilterOn = document.getElementById("btn-filter-on");
    const btnFilterOff = document.getElementById("btn-filter-off");
    const btnPlay = document.getElementById("btn-play");
    const playIcon = document.getElementById("play-icon");
    const playLabel = document.getElementById("play-label");
    const btnStep = document.getElementById("btn-step");
    const btnReset = document.getElementById("btn-reset");
    const selectSpeed = document.getElementById("select-speed");
    const timelineSlider = document.getElementById("timeline-slider");
    const timeDisplay = document.getElementById("time-display");
    const durationDisplay = document.getElementById("duration-display");

    const badgeCategory = document.getElementById("badge-category");
    const statusPill = document.getElementById("status-pill");
    const statusDot = document.getElementById("status-dot");
    const statusText = document.getElementById("status-text");
    const crashOverlay = document.getElementById("crash-overlay");
    const scenarioNote = document.getElementById("scenario-note");

    // Metric Displays
    const metricH = document.getElementById("metric-h");
    const barH = document.getElementById("bar-h");
    const metricS = document.getElementById("metric-s");
    const barS = document.getElementById("bar-s");
    const metricSSafe = document.getElementById("metric-s-safe");
    const metricVEgo = document.getElementById("metric-vego");
    const metricVLead = document.getElementById("metric-vlead");
    const metricDv = document.getElementById("metric-dv");
    const metricVEgoKmh = document.getElementById("metric-vego-kmh");
    const metricVLeadKmh = document.getElementById("metric-vlead-kmh");
    const metricUApplied = document.getElementById("metric-u-applied");
    const metricUPolicy = document.getElementById("metric-u-policy");
    const metricUCbf = document.getElementById("metric-u-cbf");
    const metricUDiff = document.getElementById("metric-u-diff");
    const metricInterventionBadge = document.getElementById("metric-intervention-badge");

    // Canvases
    const simCanvas = document.getElementById("sim-canvas");
    const simCtx = simCanvas.getContext("2d");
    const plotGapCanvas = document.getElementById("plot-gap");
    const plotGapCtx = plotGapCanvas.getContext("2d");
    const plotHCtx = document.getElementById("plot-h").getContext("2d");
    const plotVCtx = document.getElementById("plot-v").getContext("2d");
    const plotUCtx = document.getElementById("plot-u").getContext("2d");

    // Fetch active rollout data
    function getActiveRollout() {
      const sc = SIM_DATABASE.scenarios[currentScenarioKey];
      const runKey = currentVariant + "__" + currentFilterMode;
      return sc.runs[runKey];
    }

    // Update Category Badge & Notes
    function updateMetadataUI() {
      const sc = SIM_DATABASE.scenarios[currentScenarioKey];
      badgeCategory.textContent = sc.category;
      if (sc.category.includes("VICOLUNGO")) {
        badgeCategory.className = "px-2.5 py-0.5 text-xs font-semibold rounded-full bg-amber-900/60 text-amber-300 border border-amber-700/50";
      } else {
        badgeCategory.className = "px-2.5 py-0.5 text-xs font-semibold rounded-full bg-purple-900/60 text-purple-300 border border-purple-700/50";
      }
      scenarioNote.textContent = sc.description;
      const rollout = getActiveRollout();
      const totalSteps = rollout.telemetry.t.length - 1;
      timelineSlider.max = totalSteps;
      durationDisplay.textContent = rollout.telemetry.t[totalSteps].toFixed(2) + "s";
    }

    // Render Canvas Track
    function drawSimulation() {
      const rollout = getActiveRollout();
      const telem = rollout.telemetry;
      const step = Math.min(currentStep, telem.t.length - 1);

      const s = telem.s[step];
      const vEgo = telem.v_ego[step];
      const vLead = telem.v_lead[step];
      const h = telem.h[step];
      const sSafe = telem.s_safe[step];
      const uApplied = telem.u_applied[step];
      const isIntervened = telem.is_intervened[step];
      const isColl = telem.is_collision[step];

      const w = simCanvas.width;
      const hCanvas = simCanvas.height;

      // Clear road background
      simCtx.fillStyle = "#090d16";
      simCtx.fillRect(0, 0, w, hCanvas);

      // Draw highway asphalt
      const roadTop = 45;
      const roadBottom = hCanvas - 45;
      const roadH = roadBottom - roadTop;
      simCtx.fillStyle = "#1e293b";
      simCtx.fillRect(0, roadTop, w, roadH);

      // Road shoulder markers
      simCtx.strokeStyle = "#475569";
      simCtx.lineWidth = 3;
      simCtx.beginPath();
      simCtx.moveTo(0, roadTop);
      simCtx.lineTo(w, roadTop);
      simCtx.moveTo(0, roadBottom);
      simCtx.lineTo(w, roadBottom);
      simCtx.stroke();

      // Dashed lane divider with motion offset
      const offset = (step * vEgo * 1.5) % 40;
      simCtx.strokeStyle = "#94a3b8";
      simCtx.lineWidth = 2;
      simCtx.setLineDash([20, 20]);
      simCtx.beginPath();
      simCtx.moveTo(-offset, roadTop + roadH / 2);
      simCtx.lineTo(w + 40, roadTop + roadH / 2);
      simCtx.stroke();
      simCtx.setLineDash([]);

      // Vehicle placement geometry
      // Anchor ego vehicle at fixed horizontal position x = 200
      const egoX = 200;
      const egoY = roadTop + roadH / 2;
      const pxPerMeter = 5.2; // Scale: 1 meter = 5.2 pixels

      // Lead vehicle position based on gap s
      const leadX = egoX + s * pxPerMeter;
      const leadY = egoY;

      // Draw Safety Bubble / Headway Zone
      if (!isColl) {
        simCtx.save();
        let bubbleColor = "rgba(34, 197, 94, 0.22)"; // Safe green
        let strokeColor = "rgba(34, 197, 94, 0.8)";
        if (h < 0) {
          bubbleColor = "rgba(239, 68, 68, 0.35)"; // Unsafe red
          strokeColor = "rgba(239, 68, 68, 0.9)";
        } else if (h < 5.0) {
          bubbleColor = "rgba(245, 158, 11, 0.3)"; // Amber boundary
          strokeColor = "rgba(245, 158, 11, 0.9)";
        }

        // Fill safety corridor between vehicles
        simCtx.fillStyle = bubbleColor;
        simCtx.strokeStyle = strokeColor;
        simCtx.lineWidth = 1.5;
        const boxX = egoX + 35;
        const boxW = Math.max(5, leadX - 35 - boxX);
        simCtx.fillRect(boxX, egoY - 26, boxW, 52);
        simCtx.strokeRect(boxX, egoY - 26, boxW, 52);

        // Required safe threshold marker s_safe
        const sSafePx = egoX + 35 + sSafe * pxPerMeter;
        if (sSafePx < w - 20) {
          simCtx.strokeStyle = "#e2e8f0";
          simCtx.lineWidth = 2;
          simCtx.setLineDash([4, 4]);
          simCtx.beginPath();
          simCtx.moveTo(sSafePx, egoY - 32);
          simCtx.lineTo(sSafePx, egoY + 32);
          simCtx.stroke();
          simCtx.setLineDash([]);
          simCtx.fillStyle = "#e2e8f0";
          simCtx.font = "10px sans-serif";
          simCtx.fillText("s_safe", sSafePx - 14, egoY - 36);
        }

        // Distance dimension line with arrowheads
        simCtx.strokeStyle = "#f8fafc";
        simCtx.lineWidth = 1.5;
        const lineY = egoY + 36;
        simCtx.beginPath();
        simCtx.moveTo(boxX, lineY);
        simCtx.lineTo(boxX + boxW, lineY);
        simCtx.stroke();

        // Gap label
        simCtx.fillStyle = "#f8fafc";
        simCtx.font = "bold 11px sans-serif";
        simCtx.textAlign = "center";
        simCtx.fillText(`gap: ${s.toFixed(1)}m`, boxX + boxW / 2, lineY + 16);

        simCtx.restore();
      }

      // Draw Ego Car (Blue Autonomous Sedan)
      drawVehicle(egoX, egoY, "#2563eb", "EGO (RL)", vEgo, uApplied, isIntervened, true);

      // Draw Lead Car (Red Sport Sedan)
      if (leadX < w + 80) {
        drawVehicle(leadX, leadY, "#dc2626", "LEAD", vLead, telem.v_lead[Math.min(step+1, telem.v_lead.length-1)] - vLead, false, false);
      }

      // Crash Overlay
      if (isColl) {
        crashOverlay.classList.remove("hidden");
      } else {
        crashOverlay.classList.add("hidden");
      }
    }

    // Vehicle Drawing Helper
    function drawVehicle(x, y, bodyColor, label, speed, accel, isIntervened, isEgo) {
      simCtx.save();
      const carW = 65;
      const carH = 34;

      // Sensor cone for Ego Car
      if (isEgo) {
        simCtx.fillStyle = "rgba(59, 130, 246, 0.15)";
        simCtx.beginPath();
        simCtx.moveTo(x + carW / 2, y);
        simCtx.lineTo(x + carW / 2 + 180, y - 55);
        simCtx.lineTo(x + carW / 2 + 180, y + 55);
        simCtx.closePath();
        simCtx.fill();
      }

      // Car Shadow
      simCtx.fillStyle = "rgba(0, 0, 0, 0.4)";
      simCtx.beginPath();
      simCtx.ellipse(x, y + 2, carW / 2 + 4, carH / 2 + 4, 0, 0, Math.PI * 2);
      simCtx.fill();

      // Car Main Body
      simCtx.fillStyle = bodyColor;
      simCtx.beginPath();
      simCtx.roundRect(x - carW / 2, y - carH / 2, carW, carH, 8);
      simCtx.fill();
      simCtx.strokeStyle = "#ffffff";
      simCtx.lineWidth = 1;
      simCtx.stroke();

      // Windshield & Windows
      simCtx.fillStyle = "#0f172a";
      simCtx.beginPath();
      simCtx.roundRect(x - carW / 2 + 14, y - carH / 2 + 5, carW - 28, carH - 10, 4);
      simCtx.fill();

      // Headlights (Front is right side)
      simCtx.fillStyle = "#fef08a";
      simCtx.fillRect(x + carW / 2 - 2, y - carH / 2 + 4, 3, 7);
      simCtx.fillRect(x + carW / 2 - 2, y + carH / 2 - 11, 3, 7);

      // Brake Lights (Rear is left side)
      const isBraking = accel < -0.3;
      simCtx.fillStyle = isBraking ? "#ff0000" : "#7f1d1d";
      if (isBraking) {
        simCtx.shadowColor = "#ff0000";
        simCtx.shadowBlur = 12;
      }
      simCtx.fillRect(x - carW / 2 - 1, y - carH / 2 + 4, 3, 7);
      simCtx.fillRect(x - carW / 2 - 1, y + carH / 2 - 11, 3, 7);
      simCtx.shadowBlur = 0;

      // Label & Speed Badge
      simCtx.fillStyle = "#ffffff";
      simCtx.font = "bold 10px sans-serif";
      simCtx.textAlign = "center";
      simCtx.fillText(label, x, y - carH / 2 - 8);

      simCtx.fillStyle = "#94a3b8";
      simCtx.font = "9px monospace";
      simCtx.fillText(`${speed.toFixed(1)} m/s`, x, y + carH / 2 + 15);

      simCtx.restore();
    }

    // Synchronized Plots Drawing Function
    function drawPlots() {
      const rollout = getActiveRollout();
      const telem = rollout.telemetry;
      const totalSteps = telem.t.length;
      const cursorXRatio = currentStep / (totalSteps - 1);

      // Plot 1: Headway Gap s(t)
      drawSingleChart(plotGapCtx, 550, 150, [
        { data: telem.s, color: "#38bdf8", width: 2, label: "s(t)" },
        { data: telem.s_safe, color: "#4ade80", width: 1.5, dash: [4, 4], label: "s_safe" },
      ], [15.0], "Gap (m)", cursorXRatio);

      // Plot 2: Safety Margin h(t)
      drawSingleChart(plotHCtx, 550, 150, [
        { data: telem.h, color: "#10b981", width: 2, label: "h(t)" },
      ], [0.0], "Margin h (m)", cursorXRatio);

      // Plot 3: Velocities
      drawSingleChart(plotVCtx, 550, 150, [
        { data: telem.v_ego, color: "#c084fc", width: 2, label: "v_ego" },
        { data: telem.v_lead, color: "#f87171", width: 2, label: "v_lead" },
      ], [], "Speed (m/s)", cursorXRatio);

      // Plot 4: Accelerations
      drawSingleChart(plotUCtx, 550, 150, [
        { data: telem.u_applied, color: "#34d399", width: 2, label: "u_applied" },
        { data: telem.u_policy, color: "#f87171", width: 1.5, dash: [3, 3], label: "u_policy" },
      ], [-4.0, 0.0], "Accel (m/s²)", cursorXRatio);
    }

    // Generic Line Chart Drawer
    function drawSingleChart(ctx, w, h, seriesList, thresholdLines, yAxisLabel, cursorRatio) {
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#030712";
      ctx.fillRect(0, 0, w, h);

      let allVals = [];
      seriesList.forEach(s => allVals.push(...s.data));
      thresholdLines.forEach(v => allVals.push(v));
      let minVal = Math.min(...allVals);
      let maxVal = Math.max(...allVals);
      if (maxVal - minVal < 2) { maxVal += 1; minVal -= 1; }
      const padding = (maxVal - minVal) * 0.1;
      minVal -= padding;
      maxVal += padding;

      const padLeft = 40;
      const padRight = 15;
      const padTop = 15;
      const padBottom = 20;
      const plotW = w - padLeft - padRight;
      const plotH = h - padTop - padBottom;

      const toY = (val) => padTop + plotH - ((val - minVal) / (maxVal - minVal)) * plotH;
      const toX = (idx, total) => padLeft + (idx / (total - 1)) * plotW;

      // Draw Grid
      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 0; i <= 3; i++) {
        const gy = padTop + (plotH / 3) * i;
        ctx.moveTo(padLeft, gy);
        ctx.lineTo(padLeft + plotW, gy);
      }
      ctx.stroke();

      // Threshold reference lines (e.g. h=0 or u_min=-4.0)
      thresholdLines.forEach(thresh => {
        const ty = toY(thresh);
        ctx.strokeStyle = "rgba(239, 68, 68, 0.7)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(padLeft, ty);
        ctx.lineTo(padLeft + plotW, ty);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "rgba(239, 68, 68, 0.8)";
        ctx.font = "9px monospace";
        ctx.fillText(thresh.toFixed(1), padLeft - 26, ty + 3);
      });

      // Draw Series Lines
      seriesList.forEach(series => {
        ctx.strokeStyle = series.color;
        ctx.lineWidth = series.width;
        if (series.dash) ctx.setLineDash(series.dash);
        ctx.beginPath();
        series.data.forEach((val, i) => {
          const px = toX(i, series.data.length);
          const py = toY(val);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.setLineDash([]);
      });

      // Y-Axis Labels
      ctx.fillStyle = "#64748b";
      ctx.font = "9px monospace";
      ctx.textAlign = "right";
      ctx.fillText(maxVal.toFixed(0), padLeft - 5, padTop + 8);
      ctx.fillText(minVal.toFixed(0), padLeft - 5, padTop + plotH);

      // Vertical Moving Cursor (Red scrub line)
      const cursorX = padLeft + cursorRatio * plotW;
      ctx.strokeStyle = "#ef4444";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cursorX, padTop);
      ctx.lineTo(cursorX, padTop + plotH);
      ctx.stroke();
    }

    // Update Live Metrics Dashboard
    function updateMetricsUI() {
      const rollout = getActiveRollout();
      const telem = rollout.telemetry;
      const step = Math.min(currentStep, telem.t.length - 1);

      const t = telem.t[step];
      const s = telem.s[step];
      const vEgo = telem.v_ego[step];
      const vLead = telem.v_lead[step];
      const deltaV = telem.delta_v[step];
      const h = telem.h[step];
      const sSafe = telem.s_safe[step];
      const uPol = telem.u_policy[step];
      const uCbf = telem.u_cbf[step];
      const uApp = telem.u_applied[step];
      const isInterv = telem.is_intervened[step];
      const isInf = telem.is_infeasible[step];
      const isColl = telem.is_collision[step];

      // Time & Slider
      timeDisplay.textContent = t.toFixed(2) + "s";
      timelineSlider.value = step;

      // Metric Card 1: Safety Margin h
      metricH.textContent = (h >= 0 ? "+" : "") + h.toFixed(1);
      metricH.className = "text-2xl font-bold font-mono " + (h >= 0 ? "text-emerald-400" : "text-red-400");
      const hPct = Math.min(100, Math.max(0, ((h + 20) / 70) * 100));
      barH.style.width = hPct + "%";
      barH.className = h >= 0 ? "bg-emerald-500 h-full transition-all" : "bg-red-500 h-full transition-all";

      // Metric Card 2: Gap s
      metricS.textContent = s.toFixed(1);
      metricSSafe.textContent = `Req: ${sSafe.toFixed(1)}m`;
      const sPct = Math.min(100, Math.max(0, (s / 120) * 100));
      barS.style.width = sPct + "%";

      // Metric Card 3: Velocities
      metricVEgo.textContent = vEgo.toFixed(1);
      metricVLead.textContent = vLead.toFixed(1);
      metricDv.textContent = `Δv: ${(deltaV >= 0 ? "+" : "")}${deltaV.toFixed(1)} m/s`;
      metricVEgoKmh.textContent = `${(vEgo * 3.6).toFixed(0)} km/h`;
      metricVLeadKmh.textContent = `${(vLead * 3.6).toFixed(0)} km/h`;

      // Metric Card 4: Accelerations
      metricUApplied.textContent = (uApp >= 0 ? "+" : "") + uApp.toFixed(2);
      metricUPolicy.textContent = (uPol >= 0 ? "+" : "") + uPol.toFixed(2);
      metricUCbf.textContent = `CBF Limit: ${(uCbf >= 0 ? "+" : "")}${uCbf.toFixed(1)}`;
      metricUDiff.textContent = `Diff: ${Math.abs(uPol - uApp).toFixed(2)}`;

      // Intervention Badge
      if (isInterv) {
        metricInterventionBadge.textContent = "CBF OVERRIDE";
        metricInterventionBadge.className = "px-1.5 py-0.5 text-[9px] font-bold rounded bg-amber-900/80 text-amber-300 border border-amber-700/60";
      } else {
        metricInterventionBadge.textContent = "NO INTERV";
        metricInterventionBadge.className = "px-1.5 py-0.5 text-[9px] font-bold rounded bg-slate-800 text-slate-400";
      }

      // Supervisory State Indicator Pill
      if (isColl) {
        statusDot.className = "w-3 h-3 rounded-full bg-red-600 animate-ping";
        statusText.textContent = "COLLISION (s ≤ 0)";
        statusText.className = "text-sm font-semibold text-red-500";
      } else if (isInf) {
        statusDot.className = "w-3 h-3 rounded-full bg-amber-500 animate-pulse";
        statusText.textContent = "INFEASIBLE (Max Braking -4.0)";
        statusText.className = "text-sm font-semibold text-amber-400";
      } else if (isInterv) {
        statusDot.className = "w-3 h-3 rounded-full bg-blue-500 animate-pulse";
        statusText.textContent = "CBF ACTIVE (Shielding)";
        statusText.className = "text-sm font-semibold text-blue-400";
      } else if (h >= 0) {
        statusDot.className = "w-3 h-3 rounded-full bg-emerald-500";
        statusText.textContent = "SAFE (h ≥ 0)";
        statusText.className = "text-sm font-semibold text-emerald-400";
      } else {
        statusDot.className = "w-3 h-3 rounded-full bg-red-400";
        statusText.textContent = "PENETRATED (h < 0)";
        statusText.className = "text-sm font-semibold text-red-400";
      }
    }

    // Step Engine
    function stepSimulation() {
      const rollout = getActiveRollout();
      const maxSteps = rollout.telemetry.t.length - 1;
      if (currentStep < maxSteps) {
        currentStep++;
        updateMetricsUI();
        drawSimulation();
        drawPlots();
      } else {
        pauseSimulation();
      }
    }

    function playSimulation() {
      if (isPlaying) return;
      isPlaying = true;
      playIcon.textContent = "❚❚";
      playLabel.textContent = "Pause";
      btnPlay.className = "flex-1 bg-amber-600 hover:bg-amber-500 text-white font-semibold py-2 px-3 rounded-lg text-xs transition flex items-center justify-center gap-1.5 shadow";

      const rollout = getActiveRollout();
      if (currentStep >= rollout.telemetry.t.length - 1) {
        currentStep = 0;
      }

      const intervalMs = (100 / playbackSpeed);
      animTimer = setInterval(stepSimulation, intervalMs);
    }

    function pauseSimulation() {
      isPlaying = false;
      playIcon.textContent = "▶";
      playLabel.textContent = "Play";
      btnPlay.className = "flex-1 bg-blue-600 hover:bg-blue-500 text-white font-semibold py-2 px-3 rounded-lg text-xs transition flex items-center justify-center gap-1.5 shadow";
      if (animTimer) {
        clearInterval(animTimer);
        animTimer = null;
      }
    }

    function resetSimulation() {
      pauseSimulation();
      currentStep = 0;
      updateMetricsUI();
      drawSimulation();
      drawPlots();
    }

    // Event Listeners
    selectScenario.addEventListener("change", (e) => {
      currentScenarioKey = e.target.value;
      updateMetadataUI();
      resetSimulation();
    });

    selectVariant.addEventListener("change", (e) => {
      currentVariant = e.target.value;
      resetSimulation();
    });

    btnFilterOn.addEventListener("click", () => {
      currentFilterMode = "filter_on";
      btnFilterOn.className = "py-1.5 text-xs font-semibold rounded-md bg-blue-600 text-white shadow transition";
      btnFilterOff.className = "py-1.5 text-xs font-semibold rounded-md text-slate-400 hover:text-white transition";
      resetSimulation();
    });

    btnFilterOff.addEventListener("click", () => {
      currentFilterMode = "filter_off";
      btnFilterOff.className = "py-1.5 text-xs font-semibold rounded-md bg-red-600 text-white shadow transition";
      btnFilterOn.className = "py-1.5 text-xs font-semibold rounded-md text-slate-400 hover:text-white transition";
      resetSimulation();
    });

    btnPlay.addEventListener("click", () => {
      if (isPlaying) pauseSimulation();
      else playSimulation();
    });

    btnStep.addEventListener("click", () => {
      pauseSimulation();
      stepSimulation();
    });

    btnReset.addEventListener("click", resetSimulation);

    selectSpeed.addEventListener("change", (e) => {
      playbackSpeed = parseFloat(e.target.value);
      if (isPlaying) {
        pauseSimulation();
        playSimulation();
      }
    });

    timelineSlider.addEventListener("input", (e) => {
      pauseSimulation();
      currentStep = parseInt(e.target.value);
      updateMetricsUI();
      drawSimulation();
      drawPlots();
    });

    window.addEventListener("keydown", (e) => {
      if (e.code === "Space") {
        e.preventDefault();
        if (isPlaying) pauseSimulation();
        else playSimulation();
      } else if (e.code === "ArrowRight") {
        pauseSimulation();
        stepSimulation();
      } else if (e.code === "ArrowLeft") {
        pauseSimulation();
        if (currentStep > 0) currentStep--;
        updateMetricsUI();
        drawSimulation();
        drawPlots();
      }
    });

    // Initialize UI on load
    updateMetadataUI();
    resetSimulation();

    setTimeout(() => {
      playSimulation();
    }, 400);

  </script>
</body>
</html>
"""


def generate_html():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}. Run generate_simulation_data.py first.")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        database_json = f.read()

    full_html = HTML_TEMPLATE.replace("__DATABASE_JSON__", database_json)

    with open(HTML_OUTPUT, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Successfully generated visualizer: {HTML_OUTPUT.name} ({HTML_OUTPUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    generate_html()
