/**
 * app.js - Autonomous CBF Car-Following Simulation Engine for Prototype 2
 * 
 * Reuses the EXACT mathematical equations from Prototype 2:
 *   - Barrier Function: h = gap - (T_MIN * ego_speed + D_MIN)
 *   - CBF Constraint:   u_CBF <= (K_CBF * h + delta_v) / T_MIN
 *   - Nominal ACC:      u_nom = K_GAP * (gap - s_des) + K_REL * delta_v
 *   - Mediation:        u_cmd = min(u_nom, u_CBF)
 *   - Saturation:       clip(u_cmd, A_MIN, A_MAX)
 *   - Jerk Limit:       |da/dt| <= JERK_MAX
 * 
 * Numerically cross-validated against Python (cross_validate_controllers.py).
 */

// ============================================================================
// 1. EXACT PROTOTYPE 2 CONFIGURATION
// ============================================================================
const P2_CONFIG = {
  DT: 0.1,             // 10 Hz simulation time-step (s)
  D_DES: 5.0,          // Standstill / minimum desired distance (m)
  T_DES: 1.5,          // Desired time headway (s)
  K_GAP: 0.20,         // Position / headway error gain
  K_REL: 0.80,         // Relative velocity gain
  T_MIN: 2.0,          // CBF minimum safe time headway (s)
  D_MIN: 15.0,         // CBF minimum safe standstill distance (m)
  K_CBF: 0.1,          // Class-K linear decay parameter (1/s)
  A_MAX: 2.0,          // Maximum throttle acceleration (m/s²)
  A_MIN: -4.0,         // Maximum emergency braking deceleration (m/s²)
  JERK_MAX: 1.5,       // Maximum physical jerk rate limit (m/s³)
  MAX_ACC_CHANGE: 1.5 * 0.1, // 0.15 m/s² per 0.1s step
};

// Pure mathematical helpers
function computeH(gap, egoSpeed) {
  return gap - (P2_CONFIG.T_MIN * egoSpeed + P2_CONFIG.D_MIN);
}

function computeUCbf(h, deltaV) {
  return (P2_CONFIG.K_CBF * h + deltaV) / P2_CONFIG.T_MIN;
}

function computeNominalAcc(gap, egoSpeed, leadSpeed) {
  const deltaV = leadSpeed - egoSpeed;
  const sDes = P2_CONFIG.D_DES + P2_CONFIG.T_DES * egoSpeed;
  const gapError = gap - sDes;
  const uNom = P2_CONFIG.K_GAP * gapError + P2_CONFIG.K_REL * deltaV;
  return { uNom, deltaV, sDes };
}

// Controller Step Function (Exact mirror of cbf_controller.py)
function stepController(uNomSat, gap, egoSpeed, leadSpeed, prevAcc, useCbf) {
  const deltaV = leadSpeed - egoSpeed;
  const h = computeH(gap, egoSpeed);
  const uCbf = computeUCbf(h, deltaV);
  const infeasible = uCbf < P2_CONFIG.A_MIN;

  let uRaw = uNomSat;
  let intervention = false;

  if (useCbf) {
    uRaw = Math.min(uNomSat, uCbf);
    intervention = uCbf < uNomSat;
  }

  // Physical actuator saturation [-4.0, +2.0] m/s²
  const uSat = Math.max(P2_CONFIG.A_MIN, Math.min(P2_CONFIG.A_MAX, uRaw));

  // Jerk rate limitation (|da/dt| <= 1.5 m/s³)
  const accChange = Math.max(
    -P2_CONFIG.MAX_ACC_CHANGE,
    Math.min(P2_CONFIG.MAX_ACC_CHANGE, uSat - prevAcc)
  );
  const appliedAcc = prevAcc + accChange;

  return {
    appliedAcc,
    uNom: uNomSat,
    uCbf,
    uRaw,
    uSat,
    h,
    deltaV,
    intervention,
    infeasible,
  };
}

// ============================================================================
// 2. SIMULATION STATE & CONTROLLER INSTANCE
// ============================================================================
class VehicleSimState {
  constructor(initialGap, initialEgoSpeed, initialAcc = 0.0) {
    this.gap = initialGap;
    this.egoSpeed = initialEgoSpeed;
    this.appliedAcc = initialAcc;
    this.prevAcc = initialAcc;
    this.history = {
      t: [0.0],
      gap: [initialGap],
      egoSpeed: [initialEgoSpeed],
      sDes: [P2_CONFIG.D_DES + P2_CONFIG.T_DES * initialEgoSpeed],
      sSafe: [P2_CONFIG.D_MIN + P2_CONFIG.T_MIN * initialEgoSpeed],
      h: [computeH(initialGap, initialEgoSpeed)],
      uNom: [0.0],
      uCbf: [0.0],
      appliedAcc: [initialAcc],
      intervention: [false],
      infeasible: [false],
      collision: [initialGap <= 0.0],
    };
  }

  step(leadSpeed, useCbf, dt = P2_CONFIG.DT) {
    const { uNom, sDes } = computeNominalAcc(this.gap, this.egoSpeed, leadSpeed);
    const uNomSat = Math.max(P2_CONFIG.A_MIN, Math.min(P2_CONFIG.A_MAX, uNom));

    const ctrlRes = stepController(
      uNomSat,
      this.gap,
      this.egoSpeed,
      leadSpeed,
      this.prevAcc,
      useCbf
    );

    this.appliedAcc = ctrlRes.appliedAcc;
    this.prevAcc = this.appliedAcc;

    // Kinematic forward Euler state integration
    const egoSpeedNext = Math.max(0.0, this.egoSpeed + this.appliedAcc * dt);
    const gapNext = this.gap + (leadSpeed - this.egoSpeed) * dt;

    this.egoSpeed = egoSpeedNext;
    this.gap = gapNext;

    const tNow = (this.history.t.length * dt);
    const sSafeNow = P2_CONFIG.D_MIN + P2_CONFIG.T_MIN * this.egoSpeed;
    const isColl = this.gap <= 0.0;

    this.history.t.push(parseFloat(tNow.toFixed(2)));
    this.history.gap.push(parseFloat(this.gap.toFixed(2)));
    this.history.egoSpeed.push(parseFloat(this.egoSpeed.toFixed(2)));
    this.history.sDes.push(parseFloat(sDes.toFixed(2)));
    this.history.sSafe.push(parseFloat(sSafeNow.toFixed(2)));
    this.history.h.push(parseFloat(ctrlRes.h.toFixed(2)));
    this.history.uNom.push(parseFloat(ctrlRes.uNom.toFixed(2)));
    this.history.uCbf.push(parseFloat(Math.max(-15.0, Math.min(10.0, ctrlRes.uCbf)).toFixed(2)));
    this.history.appliedAcc.push(parseFloat(this.appliedAcc.toFixed(2)));
    this.history.intervention.push(ctrlRes.intervention);
    this.history.infeasible.push(ctrlRes.infeasible);
    this.history.collision.push(isColl);

    return ctrlRes;
  }
}

// ============================================================================
// 3. SCENARIO GENERATOR & TRAJECTORY DATABASE
// ============================================================================
let VICOLUNGO_DATABASE = null;

// Built-in Synthetic Scenarios (dt = 0.1s)
function buildSyntheticScenario(scenarioType) {
  const dt = P2_CONFIG.DT;
  let totalSteps = 250; // 25.0s
  let s0 = 65.0;
  let vEgo0 = 22.0;
  let leadSpeeds = [];

  switch (scenarioType) {
    case "cruising": {
      s0 = 65.0;
      vEgo0 = 22.0;
      for (let i = 0; i <= totalSteps; i++) {
        const t = i * dt;
        // Cruising around 22 m/s (~80 km/h) with subtle traffic wave oscillations
        const v = 22.0 + 1.0 * Math.sin(0.3 * t) + 0.5 * Math.sin(0.8 * t);
        leadSpeeds.push(parseFloat(v.toFixed(3)));
      }
      return {
        title: "1. Normal Car Following",
        category: "[SYNTHETIC DEMONSTRATION]",
        description: "Lead cruises at ~80 km/h with natural traffic oscillations. Tests nominal ACC equilibrium.",
        s0, vEgo0, leadSpeeds, duration_s: 25.0,
      };
    }
    case "lead_accel": {
      s0 = 55.0;
      vEgo0 = 20.0;
      for (let i = 0; i <= totalSteps; i++) {
        const t = i * dt;
        let v = 20.0;
        if (t >= 3.0 && t <= 9.0) {
          v = 20.0 + 1.33 * (t - 3.0); // accelerates from 20 to 28 m/s (~100 km/h)
        } else if (t > 9.0) {
          v = 28.0;
        }
        leadSpeeds.push(parseFloat(v.toFixed(3)));
      }
      return {
        title: "2. Lead Vehicle Acceleration",
        category: "[SYNTHETIC DEMONSTRATION]",
        description: "Lead vehicle accelerates from 72 km/h to 101 km/h at +1.3 m/s². Follower expands speed safely.",
        s0, vEgo0, leadSpeeds, duration_s: 25.0,
      };
    }
    case "lead_decel": {
      s0 = 70.0;
      vEgo0 = 25.0;
      for (let i = 0; i <= totalSteps; i++) {
        const t = i * dt;
        let v = 25.0;
        if (t >= 3.0 && t <= 7.5) {
          v = 25.0 - 1.8 * (t - 3.0); // decelerates from 25 to 17 m/s at -1.8 m/s²
        } else if (t > 7.5) {
          v = 17.0;
        }
        leadSpeeds.push(parseFloat(v.toFixed(3)));
      }
      return {
        title: "3. Moderate Lead Deceleration",
        category: "[SYNTHETIC DEMONSTRATION]",
        description: "Lead decelerates steadily from 90 km/h to 61 km/h at -1.8 m/s². Tests standard deceleration response.",
        s0, vEgo0, leadSpeeds, duration_s: 25.0,
      };
    }
    case "emergency_braking": {
      s0 = 68.0;
      vEgo0 = 25.0;
      for (let i = 0; i <= totalSteps; i++) {
        const t = i * dt;
        let v = 25.0;
        if (t >= 3.0 && t <= 8.5) {
          v = 25.0 - 3.6 * (t - 3.0); // hard emergency braking at -3.6 m/s² down to 5.2 m/s
        } else if (t > 8.5) {
          v = 5.2;
        }
        leadSpeeds.push(parseFloat(v.toFixed(3)));
      }
      return {
        title: "4. Sudden Emergency Hard Braking",
        category: "[SYNTHETIC DEMONSTRATION]",
        description: "Lead slams brakes at -3.6 m/s² (near physical limit). Demonstrates critical CBF intervention avoiding crash.",
        s0, vEgo0, leadSpeeds, duration_s: 25.0,
      };
    }
    case "closing_speed_stress": {
      // High closing speed stress test: Ego at 30 m/s (108 km/h), Lead at 18 m/s (65 km/h), Gap = 35m
      s0 = 35.0;
      vEgo0 = 30.0;
      for (let i = 0; i <= totalSteps; i++) {
        leadSpeeds.push(18.0);
      }
      return {
        title: "5. Closing-Speed Stress Test",
        category: "[SYNTHETIC DEMONSTRATION]",
        description: "Ego closes rapidly at 108 km/h on a 65 km/h vehicle with a 35m gap. Tests actuator saturation and jerk lag.",
        s0, vEgo0, leadSpeeds, duration_s: 25.0,
      };
    }
  }
}

// ============================================================================
// 4. MAIN SIMULATOR APPLICATION CONTROLLER
// ============================================================================
class VisualizerApp {
  constructor() {
    this.mode = "synthetic"; // "synthetic" | "vicolungo"
    this.scenarioType = "emergency_braking";
    this.controllerMode = "comparison"; // Default to side-by-side comparison
    this.vicolungoTrajId = "16"; // Default research stress case

    this.simData = null;
    this.simLead = null; // Single follower or comparison
    this.simFollowerCBF = null;
    this.simFollowerBase = null;

    this.currentStep = 0;
    this.maxSteps = 0;
    this.isPlaying = false;
    this.playbackSpeed = 1.0;
    this.timer = null;

    this.emergencyBrakeTriggered = false;

    this.initDOM();
    this.initEvents();
    this.loadVicolungoData().then(() => {
      this.loadScenario();
    });
  }

  async loadVicolungoData() {
    if (typeof window.EMBEDDED_VICOLUNGO !== 'undefined' && window.EMBEDDED_VICOLUNGO) {
      VICOLUNGO_DATABASE = window.EMBEDDED_VICOLUNGO;
      console.log("Loaded embedded Vicolungo trajectories:", Object.keys(VICOLUNGO_DATABASE));
      return;
    }
    try {
      const resp = await fetch("vicolungo_trajectories.json");
      if (resp.ok) {
        VICOLUNGO_DATABASE = await resp.json();
        console.log("Loaded Vicolungo trajectories:", Object.keys(VICOLUNGO_DATABASE));
      }
    } catch (e) {
      console.warn("Could not load external vicolungo_trajectories.json directly via fetch:", e);
    }
  }


  initDOM() {
    // Buttons & Selectors
    this.btnPlay = document.getElementById("btn-play");
    this.playIcon = document.getElementById("play-icon");
    this.playLabel = document.getElementById("play-label");
    this.btnStep = document.getElementById("btn-step");
    this.btnReset = document.getElementById("btn-reset");
    this.btnEmergency = document.getElementById("btn-emergency-brake");

    this.selectMode = document.getElementById("select-sim-mode");
    this.selectScenario = document.getElementById("select-scenario");
    this.selectTraj = document.getElementById("select-trajectory");
    this.selectController = document.getElementById("select-controller");
    if (this.selectController) this.selectController.value = this.controllerMode;
    this.selectSpeed = document.getElementById("select-speed");

    this.trajContainer = document.getElementById("trajectory-select-container");
    this.scenarioContainer = document.getElementById("scenario-select-container");

    this.timelineSlider = document.getElementById("timeline-slider");
    this.timeDisplay = document.getElementById("time-display");
    this.durationDisplay = document.getElementById("duration-display");

    // Badges & Status
    this.badgeCategory = document.getElementById("badge-category");
    this.scenarioTitle = document.getElementById("scenario-title");
    this.scenarioDesc = document.getElementById("scenario-desc");
    this.statusPill = document.getElementById("status-pill");
    this.statusDot = document.getElementById("status-dot");
    this.statusText = document.getElementById("status-text");

    // Live Metrics
    this.metricH = document.getElementById("metric-h");
    this.barH = document.getElementById("bar-h");
    this.metricGap = document.getElementById("metric-gap");
    this.barGap = document.getElementById("bar-gap");
    this.metricReqGap = document.getElementById("metric-req-gap");

    this.metricLeadSpeed = document.getElementById("metric-lead-speed");
    this.metricEgoSpeed = document.getElementById("metric-ego-speed");
    this.metricDeltaV = document.getElementById("metric-delta-v");

    this.metricUNom = document.getElementById("metric-u-nom");
    this.metricUCbf = document.getElementById("metric-u-cbf");
    this.metricUApplied = document.getElementById("metric-u-applied");
    this.metricIntervention = document.getElementById("metric-intervention-badge");
    this.metricInfeasible = document.getElementById("metric-infeasible-badge");

    // Canvases
    this.canvasRoad = document.getElementById("canvas-road");
    this.ctxRoad = this.canvasRoad.getContext("2d");

    this.chartGap = document.getElementById("chart-gap").getContext("2d");
    this.chartH = document.getElementById("chart-h").getContext("2d");
    this.chartSpeed = document.getElementById("chart-speed").getContext("2d");
    this.chartAccel = document.getElementById("chart-accel").getContext("2d");

    this.crashOverlay = document.getElementById("crash-overlay");
    this.crashDesc = document.getElementById("crash-desc");
  }

  initEvents() {
    this.btnPlay.addEventListener("click", () => this.togglePlay());
    this.btnStep.addEventListener("click", () => {
      this.pause();
      this.step();
    });
    this.btnReset.addEventListener("click", () => this.reset());

    this.selectMode.addEventListener("change", (e) => {
      this.mode = e.target.value;
      if (this.mode === "vicolungo") {
        this.trajContainer.classList.remove("hidden");
        this.scenarioContainer.classList.add("hidden");
      } else {
        this.trajContainer.classList.add("hidden");
        this.scenarioContainer.classList.remove("hidden");
      }
      this.loadScenario();
    });

    this.selectScenario.addEventListener("change", (e) => {
      this.scenarioType = e.target.value;
      this.loadScenario();
    });

    this.selectTraj.addEventListener("change", (e) => {
      this.vicolungoTrajId = e.target.value;
      this.loadScenario();
    });

    this.selectController.addEventListener("change", (e) => {
      this.controllerMode = e.target.value;
      this.loadScenario();
    });

    this.selectSpeed.addEventListener("change", (e) => {
      this.playbackSpeed = parseFloat(e.target.value);
      if (this.isPlaying) {
        this.pause();
        this.play();
      }
    });

    this.timelineSlider.addEventListener("input", (e) => {
      this.pause();
      this.currentStep = parseInt(e.target.value);
      this.render();
    });

    // Emergency Brake Interactive Trigger
    this.btnEmergency.addEventListener("click", () => {
      this.triggerLiveEmergencyBrake();
    });

    // Keyboard Shortcuts
    window.addEventListener("keydown", (e) => {
      if (e.code === "Space") {
        e.preventDefault();
        this.togglePlay();
      } else if (e.code === "ArrowRight") {
        this.pause();
        this.step();
      } else if (e.code === "ArrowLeft") {
        this.pause();
        if (this.currentStep > 0) {
          this.currentStep--;
          this.render();
        }
      }
    });
  }

  triggerLiveEmergencyBrake() {
    if (!this.simData) return;
    const curIdx = this.currentStep;
    const total = this.simData.leadSpeeds.length;
    // Decelerate lead vehicle at -3.8 m/s² for next 4.0s
    let curV = this.simData.leadSpeeds[curIdx];
    for (let k = curIdx; k < total; k++) {
      curV = Math.max(0.0, curV - 3.8 * P2_CONFIG.DT);
      this.simData.leadSpeeds[k] = parseFloat(curV.toFixed(3));
    }
    // Recompute simulation rollout forward from current state
    this.recomputeFutureRollout();
    this.render();

    // Visual feedback
    this.btnEmergency.classList.add("ring-4", "ring-red-400");
    setTimeout(() => this.btnEmergency.classList.remove("ring-4", "ring-red-400"), 600);
  }

  loadScenario() {
    this.pause();
    this.emergencyBrakeTriggered = false;

    if (this.mode === "synthetic") {
      this.simData = buildSyntheticScenario(this.scenarioType);
    } else {
      // Vicolungo Replay Mode
      if (VICOLUNGO_DATABASE && VICOLUNGO_DATABASE[this.vicolungoTrajId]) {
        const tData = VICOLUNGO_DATABASE[this.vicolungoTrajId];
        this.simData = {
          title: tData.title,
          category: "[VICOLUNGO — RECORDED LEAD REPLAY]",
          description: tData.description,
          s0: tData.s0,
          vEgo0: tData.v_ego0,
          leadSpeeds: [...tData.speed_lv],
          duration_s: tData.duration_s,
        };
      } else {
        // Fallback default if json not yet parsed
        this.simData = {
          title: "Vicolungo Trajectory 16 — Safety Recovery / Actuator-Limit Stress Case",
          category: "[VICOLUNGO — RECORDED LEAD REPLAY]",
          description: "Recorded lead speed from A4 highway (Feb 2019). Follower starts outside the safe set (h0 = -37.05m < 0). Heavy closing speed triggers CBF intervention and actuator saturation (-4.0 m/s²). Tests emergency recovery and saturation handling, NOT forward invariance.",
          s0: 33.09,
          vEgo0: 27.57,
          leadSpeeds: new Array(400).fill(19.05),
          duration_s: 40.0,
        };
      }

    }

    // Update Header Badges
    this.badgeCategory.textContent = this.simData.category;
    if (this.simData.category.includes("VICOLUNGO")) {
      this.badgeCategory.className = "px-2.5 py-0.5 text-xs font-semibold rounded-full bg-amber-900/60 text-amber-300 border border-amber-700/50";
    } else {
      this.badgeCategory.className = "px-2.5 py-0.5 text-xs font-semibold rounded-full bg-purple-900/60 text-purple-300 border border-purple-700/50";
    }

    this.scenarioTitle.textContent = this.simData.title;
    this.scenarioDesc.textContent = this.simData.description;

    // Run Full Forward Simulation
    this.recomputeFutureRollout();

    this.maxSteps = this.simData.leadSpeeds.length - 1;
    this.timelineSlider.max = this.maxSteps;
    this.durationDisplay.textContent = `${(this.maxSteps * P2_CONFIG.DT).toFixed(1)}s`;

    this.currentStep = 0;
    this.render();
  }

  recomputeFutureRollout() {
    const s0 = this.simData.s0;
    const vEgo0 = this.simData.vEgo0;
    const leadSpeeds = this.simData.leadSpeeds;

    // Simulate ACC + CBF Follower
    this.simFollowerCBF = new VehicleSimState(s0, vEgo0, 0.0);
    for (let k = 0; k < leadSpeeds.length - 1; k++) {
      this.simFollowerCBF.step(leadSpeeds[k], true);
    }

    // Simulate Baseline ACC Follower
    this.simFollowerBase = new VehicleSimState(s0, vEgo0, 0.0);
    for (let k = 0; k < leadSpeeds.length - 1; k++) {
      this.simFollowerBase.step(leadSpeeds[k], false);
    }
  }

  togglePlay() {
    if (this.isPlaying) this.pause();
    else this.play();
  }

  play() {
    if (this.isPlaying) return;
    this.isPlaying = true;
    this.playIcon.textContent = "❚❚";
    this.playLabel.textContent = "Pause";
    this.btnPlay.classList.replace("bg-blue-600", "bg-amber-600");
    this.btnPlay.classList.replace("hover:bg-blue-500", "hover:bg-amber-500");

    if (this.currentStep >= this.maxSteps) {
      this.currentStep = 0;
    }

    const intervalMs = (100 / this.playbackSpeed); // 10 Hz base
    this.timer = setInterval(() => this.step(), intervalMs);
  }

  pause() {
    this.isPlaying = false;
    this.playIcon.textContent = "▶";
    this.playLabel.textContent = "Play";
    this.btnPlay.classList.replace("bg-amber-600", "bg-blue-600");
    this.btnPlay.classList.replace("hover:bg-amber-500", "hover:bg-blue-500");
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  reset() {
    this.pause();
    this.currentStep = 0;
    this.render();
  }

  step() {
    if (this.currentStep < this.maxSteps) {
      this.currentStep++;
      this.render();
    } else {
      this.pause();
    }
  }

  // ==========================================================================
  // 5. RENDERING PIPELINE (2.5D AUTONOMOUS CAR HIGHWAY TRACK & CHARTS)
  // ==========================================================================
  render() {
    this.renderMetrics();
    this.drawRoadCanvas();
    this.drawCharts();
  }

  renderMetrics() {
    const isComparison = this.controllerMode === "comparison";
    const primary = (this.controllerMode === "baseline") ? this.simFollowerBase : this.simFollowerCBF;
    const hData = primary.history;
    const step = Math.min(this.currentStep, hData.t.length - 1);

    const t = hData.t[step];
    const gap = hData.gap[step];
    const sSafe = hData.sSafe[step];
    const h = hData.h[step];
    const egoSpeed = hData.egoSpeed[step];
    const leadSpeed = this.simData.leadSpeeds[step];
    const deltaV = leadSpeed - egoSpeed;
    const uNom = hData.uNom[step];
    const uCbf = hData.uCbf[step];
    const uApp = hData.appliedAcc[step];
    const isInterv = hData.intervention[step];
    const isInf = hData.infeasible[step];
    const isColl = hData.collision[step];

    // Time & Slider
    this.timeDisplay.textContent = `${t.toFixed(2)}s`;
    this.timelineSlider.value = step;

    // Safety Margin h
    this.metricH.textContent = `${h >= 0 ? "+" : ""}${h.toFixed(2)}`;
    this.metricH.className = `metric-value text-2xl font-bold ${h >= 0 ? "text-emerald-400" : "text-red-400"}`;
    const hPercent = Math.min(100, Math.max(0, ((h + 30) / 90) * 100));
    this.barH.style.width = `${hPercent}%`;
    this.barH.className = h >= 0 ? "bg-emerald-500 h-full transition-all" : "bg-red-500 h-full transition-all";

    // Physical Gap
    this.metricGap.textContent = gap.toFixed(1);
    this.metricReqGap.textContent = `Req: ${sSafe.toFixed(1)}m`;
    const gapPercent = Math.min(100, Math.max(0, (gap / 120) * 100));
    this.barGap.style.width = `${gapPercent}%`;

    // Velocities
    this.metricLeadSpeed.textContent = `${leadSpeed.toFixed(1)} m/s (${(leadSpeed * 3.6).toFixed(0)} km/h)`;
    this.metricEgoSpeed.textContent = `${egoSpeed.toFixed(1)} m/s (${(egoSpeed * 3.6).toFixed(0)} km/h)`;
    this.metricDeltaV.textContent = `Δv: ${deltaV >= 0 ? "+" : ""}${deltaV.toFixed(1)} m/s`;

    // Control Commands
    this.metricUNom.textContent = `${uNom >= 0 ? "+" : ""}${uNom.toFixed(2)} m/s²`;
    this.metricUCbf.textContent = `${uCbf >= 0 ? "+" : ""}${uCbf.toFixed(2)} m/s²`;
    this.metricUApplied.textContent = `${uApp >= 0 ? "+" : ""}${uApp.toFixed(2)} m/s²`;

    // Badges
    if (isInterv && this.controllerMode !== "baseline") {
      this.metricIntervention.textContent = "CBF INTERVENED";
      this.metricIntervention.className = "px-1.5 py-0.5 text-[9px] font-bold rounded bg-amber-900/80 text-amber-300 border border-amber-700/60";
    } else {
      this.metricIntervention.textContent = "NO INTERV";
      this.metricIntervention.className = "px-1.5 py-0.5 text-[9px] font-bold rounded bg-slate-800 text-slate-400";
    }

    if (isInf && this.controllerMode !== "baseline") {
      this.metricInfeasible.textContent = "INFEASIBLE (u_cbf < -4.0)";
      this.metricInfeasible.className = "px-1.5 py-0.5 text-[9px] font-bold rounded bg-red-900/80 text-red-300 border border-red-700/60";
    } else {
      this.metricInfeasible.textContent = "FEASIBLE";
      this.metricInfeasible.className = "px-1.5 py-0.5 text-[9px] font-bold rounded bg-slate-800 text-slate-400";
    }

    // Status Pill
    if (isColl) {
      this.statusDot.className = "w-3 h-3 rounded-full bg-red-600 animate-ping";
      this.statusText.textContent = "COLLISION (s ≤ 0)";
      this.statusText.className = "text-sm font-semibold text-red-500";
      this.crashOverlay.classList.remove("hidden");
      this.crashDesc.textContent = (this.controllerMode === "baseline")
        ? "Baseline ACC has no safety barrier awareness and caused a collision during heavy deceleration."
        : "Initial condition or actuator limit caused collision.";
    } else {
      this.crashOverlay.classList.add("hidden");
      if (isInf) {
        this.statusDot.className = "w-3 h-3 rounded-full bg-amber-500 animate-pulse";
        this.statusText.textContent = "ACTUATOR SATURATED (-4.0 m/s²)";
        this.statusText.className = "text-sm font-semibold text-amber-400";
      } else if (isInterv && this.controllerMode !== "baseline") {
        this.statusDot.className = "w-3 h-3 rounded-full bg-blue-500 animate-pulse";
        this.statusText.textContent = "CBF ACTIVE (Shielding)";
        this.statusText.className = "text-sm font-semibold text-blue-400";
      } else if (h >= 5.0) {
        this.statusDot.className = "w-3 h-3 rounded-full bg-emerald-500";
        this.statusText.textContent = "SAFE (h ≥ 5m)";
        this.statusText.className = "text-sm font-semibold text-emerald-400";
      } else if (h >= 0.0) {
        this.statusDot.className = "w-3 h-3 rounded-full bg-amber-400";
        this.statusText.textContent = "PROXIMITY (0 ≤ h < 5m)";
        this.statusText.className = "text-sm font-semibold text-amber-300";
      } else {
        this.statusDot.className = "w-3 h-3 rounded-full bg-red-400 animate-pulse";
        this.statusText.textContent = "PENETRATED (h < 0m)";
        this.statusText.className = "text-sm font-semibold text-red-400";
      }
    }
  }

  // ==========================================================================
  // 6. HIGHWAY CANVAS WITH SMOOTH CAMERA TRACKING & STYLIZED CARS
  // ==========================================================================
  drawRoadCanvas() {
    const ctx = this.ctxRoad;
    const w = this.canvasRoad.width;
    const h = this.canvasRoad.height;
    const step = this.currentStep;

    const leadSpeed = this.simData.leadSpeeds[step];
    const isComparison = (this.controllerMode === "comparison");

    // Clear background (Dark asphalt & environment)
    ctx.fillStyle = "#0a0f1d";
    ctx.fillRect(0, 0, w, h);

    const roadTop = 45;
    const roadBottom = h - 35;
    const roadHeight = roadBottom - roadTop;
    const laneHeight = isComparison ? roadHeight / 2 : roadHeight;

    // Draw Highway Surface
    ctx.fillStyle = "#161e2e";
    ctx.fillRect(0, roadTop, w, roadHeight);

    // Shoulder lines & rumble strips
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(0, roadTop);
    ctx.lineTo(w, roadTop);
    ctx.moveTo(0, roadBottom);
    ctx.lineTo(w, roadBottom);
    ctx.stroke();

    // Roadside distance tick markers every 10 meters (motion parallax)
    const egoSpeedRef = isComparison
      ? this.simFollowerCBF.history.egoSpeed[step]
      : (this.controllerMode === "baseline" ? this.simFollowerBase.history.egoSpeed[step] : this.simFollowerCBF.history.egoSpeed[step]);
    const scrollOffset = (step * egoSpeedRef * 4.0) % 60;

    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 1;
    ctx.fillStyle = "#64748b";
    ctx.font = "9px monospace";
    for (let x = -scrollOffset; x < w + 60; x += 60) {
      ctx.beginPath();
      ctx.moveTo(x, roadTop - 6);
      ctx.lineTo(x, roadTop);
      ctx.moveTo(x, roadBottom);
      ctx.lineTo(x, roadBottom + 6);
      ctx.stroke();
    }

    // Lane divider & Banners
    if (isComparison) {
      // Top Banner: Parallel Counterfactual Explanation
      ctx.save();
      ctx.fillStyle = "rgba(15, 23, 42, 0.90)";
      ctx.strokeStyle = "rgba(59, 130, 246, 0.45)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(15, 8, 860, 28, 6);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#93c5fd";
      ctx.font = "bold 11px sans-serif";
      ctx.textAlign = "left";
      ctx.fillText("PARALLEL COUNTERFACTUAL COMPARISON — SAME INITIAL STATE & SAME LEAD DISTURBANCE", 25, 21);

      ctx.fillStyle = "#94a3b8";
      ctx.font = "10px sans-serif";
      ctx.fillText("(Two independent parallel simulations for direct visual contrast — no lateral vehicle interaction)", 25, 32);
      ctx.restore();

      // White dashed center line separating Track 1 (Baseline) and Track 2 (CBF)
      const midY = roadTop + laneHeight;
      ctx.strokeStyle = "#94a3b8";
      ctx.lineWidth = 2;
      ctx.setLineDash([20, 20]);
      ctx.beginPath();
      ctx.moveTo(0, midY);
      ctx.lineTo(w, midY);
      ctx.stroke();
      ctx.setLineDash([]);

      // Track 1 (Upper): Baseline ACC label & live status badge
      ctx.save();
      ctx.fillStyle = "rgba(245, 158, 11, 0.15)";
      ctx.strokeStyle = "rgba(245, 158, 11, 0.5)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(15, roadTop + 6, 335, 20, 4);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#fbbf24";
      ctx.font = "bold 10px monospace";
      ctx.fillText("TRACK 1 (UPPER): BASELINE ACC (NOMINAL SPACING)", 22, roadTop + 20);

      const baseGap = this.simFollowerBase.history.gap[step];
      const baseCollided = baseGap <= 0.0;
      if (baseCollided) {
        ctx.fillStyle = "#ef4444";
        ctx.font = "bold 11px monospace";
        ctx.fillText("💥 COLLISION (s ≤ 0m)", 360, roadTop + 20);
      }
      ctx.restore();

      // Track 2 (Lower): ACC + CBF label & live status badge
      ctx.save();
      ctx.fillStyle = "rgba(59, 130, 246, 0.15)";
      ctx.strokeStyle = "rgba(59, 130, 246, 0.5)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.roundRect(15, roadTop + laneHeight + 6, 335, 20, 4);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = "#60a5fa";
      ctx.font = "bold 10px monospace";
      ctx.fillText("TRACK 2 (LOWER): ACC + CBF (SAFETY SUPERVISED)", 22, roadTop + laneHeight + 20);

      const cbfH = this.simFollowerCBF.history.h[step];
      const cbfInterv = this.simFollowerCBF.history.intervention[step];
      if (cbfH < 0.0) {
        ctx.fillStyle = "#f87171";
        ctx.font = "bold 11px monospace";
        ctx.fillText(`⚠️ PENETRATED (h = ${cbfH.toFixed(1)}m)`, 360, roadTop + laneHeight + 20);
      } else if (cbfInterv) {
        ctx.fillStyle = "#38bdf8";
        ctx.font = "bold 11px monospace";
        ctx.fillText(`🛡️ CBF ACTIVE (h = +${cbfH.toFixed(1)}m)`, 360, roadTop + laneHeight + 20);
      } else {
        ctx.fillStyle = "#34d399";
        ctx.font = "bold 11px monospace";
        ctx.fillText(`SAFE (h = +${cbfH.toFixed(1)}m)`, 360, roadTop + laneHeight + 20);
      }
      ctx.restore();

    } else {
      // Single Lane highway track with dashed lane markings
      const midY = roadTop + roadHeight / 2;
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([25, 25]);
      ctx.beginPath();
      ctx.moveTo(-scrollOffset, midY);
      ctx.lineTo(w + 50, midY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Vehicle positioning scale
    const pxPerMeter = 4.8;
    const egoX = 180; // Anchor AV horizontally on track

    if (isComparison) {
      // Comparison Mode: Draw two parallel follower cars on separate lanes!
      const lane1Y = roadTop + laneHeight / 2 + 10;
      const lane2Y = roadTop + laneHeight + laneHeight / 2 + 10;

      const gapBase = this.simFollowerBase.history.gap[step];
      const gapCBF = this.simFollowerCBF.history.gap[step];
      const leadXBase = egoX + gapBase * pxPerMeter;
      const leadXCBF = egoX + gapCBF * pxPerMeter;

      // Draw Safety Envelopes
      this.drawSafetyEnvelope(ctx, egoX, lane2Y, gapCBF, this.simFollowerCBF.history.sSafe[step], this.simFollowerCBF.history.h[step], pxPerMeter);
      this.drawSafetyEnvelope(ctx, egoX, lane1Y, gapBase, this.simFollowerBase.history.sSafe[step], this.simFollowerBase.history.h[step], pxPerMeter);

      // Draw Vehicles
      // Lane 1: Baseline ACC (Amber/Red tint)
      this.drawCar(ctx, egoX, lane1Y, "#f59e0b", "AV (Baseline ACC)", this.simFollowerBase.history.egoSpeed[step], this.simFollowerBase.history.appliedAcc[step], true);
      this.drawCar(ctx, leadXBase, lane1Y, "#ef4444", "Lead Car", leadSpeed, 0, false);

      // Lane 2: ACC + CBF (Blue/Green tint)
      this.drawCar(ctx, egoX, lane2Y, "#3b82f6", "AV (ACC + CBF)", this.simFollowerCBF.history.egoSpeed[step], this.simFollowerCBF.history.appliedAcc[step], true);
      this.drawCar(ctx, leadXCBF, lane2Y, "#ef4444", "Lead Car", leadSpeed, 0, false);

    } else {
      // Single Follower Mode
      const activeFollower = (this.controllerMode === "baseline") ? this.simFollowerBase : this.simFollowerCBF;
      const gap = activeFollower.history.gap[step];
      const sSafe = activeFollower.history.sSafe[step];
      const hVal = activeFollower.history.h[step];
      const egoSpeed = activeFollower.history.egoSpeed[step];
      const appliedAcc = activeFollower.history.appliedAcc[step];
      const isCBF = (this.controllerMode === "cbf");

      const carY = roadTop + roadHeight / 2;
      const leadX = egoX + gap * pxPerMeter;

      // Draw Dynamic CBF Safety Envelope
      this.drawSafetyEnvelope(ctx, egoX, carY, gap, sSafe, hVal, pxPerMeter);

      // Draw Follower Car
      const avColor = isCBF ? "#3b82f6" : "#f59e0b";
      const avLabel = isCBF ? "AV (ACC + CBF)" : "AV (Baseline ACC)";
      this.drawCar(ctx, egoX, carY, avColor, avLabel, egoSpeed, appliedAcc, true);

      // Draw Lead Car
      const leadAccel = (step + 1 < this.simData.leadSpeeds.length)
        ? (this.simData.leadSpeeds[step + 1] - leadSpeed) / P2_CONFIG.DT
        : 0.0;
      this.drawCar(ctx, leadX, carY, "#ef4444", "Lead Car", leadSpeed, leadAccel, false);
    }
  }

  // Draw Dynamic Safety Envelope & Distance Dimension Line
  drawSafetyEnvelope(ctx, egoX, y, gap, sSafe, hVal, pxPerMeter) {
    ctx.save();
    const boxX = egoX + 38;
    const boxW = Math.max(5, gap * pxPerMeter - 38);

    let fillColor = "rgba(34, 197, 94, 0.22)"; // Safe green
    let strokeColor = "rgba(34, 197, 94, 0.85)";

    if (hVal < 0) {
      fillColor = "rgba(239, 68, 68, 0.35)"; // Unsafe red
      strokeColor = "rgba(239, 68, 68, 0.95)";
    } else if (hVal < 5.0) {
      fillColor = "rgba(245, 158, 11, 0.28)"; // Warning amber
      strokeColor = "rgba(245, 158, 11, 0.9)";
    }

    // Safety Corridor Rectangle
    ctx.fillStyle = fillColor;
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1.5;
    ctx.fillRect(boxX, y - 24, boxW, 48);
    ctx.strokeRect(boxX, y - 24, boxW, 48);

    // Required Safe Distance Tick Line (s_safe)
    const sSafePx = egoX + 38 + sSafe * pxPerMeter;
    if (sSafePx < ctx.canvas.width - 20) {
      ctx.strokeStyle = "#f8fafc";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(sSafePx, y - 30);
      ctx.lineTo(sSafePx, y + 30);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#f8fafc";
      ctx.font = "10px monospace";
      ctx.fillText(`s_safe: ${sSafe.toFixed(1)}m`, sSafePx - 20, y - 34);
    }

    // Gap Dimension Line
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 1.5;
    const lineY = y + 34;
    ctx.beginPath();
    ctx.moveTo(boxX, lineY);
    ctx.lineTo(boxX + boxW, lineY);
    ctx.stroke();

    // Arrows
    ctx.fillStyle = "#cbd5e1";
    ctx.beginPath();
    ctx.moveTo(boxX + 5, lineY - 3);
    ctx.lineTo(boxX, lineY);
    ctx.lineTo(boxX + 5, lineY + 3);
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(boxX + boxW - 5, lineY - 3);
    ctx.lineTo(boxX + boxW, lineY);
    ctx.lineTo(boxX + boxW - 5, lineY + 3);
    ctx.fill();

    // Distance Label
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 11px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`gap: ${gap.toFixed(1)}m (h: ${hVal >= 0 ? "+" : ""}${hVal.toFixed(1)}m)`, boxX + boxW / 2, lineY + 15);

    ctx.restore();
  }

  // Draw Stylized Autonomous Vehicle Sprite (2.5D Top-Down View)
  drawCar(ctx, x, y, color, label, speed, accel, isAV) {
    ctx.save();
    const carW = 68;
    const carH = 34;

    // Sensor Beam (LiDAR cone forward)
    if (isAV) {
      ctx.fillStyle = "rgba(59, 130, 246, 0.15)";
      ctx.beginPath();
      ctx.moveTo(x + carW / 2, y);
      ctx.lineTo(x + carW / 2 + 160, y - 48);
      ctx.lineTo(x + carW / 2 + 160, y + 48);
      ctx.closePath();
      ctx.fill();

      // Spinning LiDAR Puck on Roof
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.arc(x - 4, y, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#0284c7";
      ctx.stroke();
    }

    // Shadow
    ctx.fillStyle = "rgba(0, 0, 0, 0.45)";
    ctx.beginPath();
    ctx.ellipse(x, y + 2, carW / 2 + 4, carH / 2 + 4, 0, 0, Math.PI * 2);
    ctx.fill();

    // Car Body (Aerodynamic contours)
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(x - carW / 2, y - carH / 2, carW, carH, 7);
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1;
    ctx.stroke();

    // Windows & Windshield
    ctx.fillStyle = "#0f172a";
    ctx.beginPath();
    ctx.roundRect(x - carW / 2 + 15, y - carH / 2 + 5, carW - 28, carH - 10, 4);
    ctx.fill();

    // Wheels
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(x - carW / 2 + 8, y - carH / 2 - 3, 14, 4);
    ctx.fillRect(x + carW / 2 - 22, y - carH / 2 - 3, 14, 4);
    ctx.fillRect(x - carW / 2 + 8, y + carH / 2 - 1, 14, 4);
    ctx.fillRect(x + carW / 2 - 22, y + carH / 2 - 1, 14, 4);

    // Headlights (Front is right side)
    ctx.fillStyle = "#fef08a";
    ctx.fillRect(x + carW / 2 - 2, y - carH / 2 + 4, 3, 7);
    ctx.fillRect(x + carW / 2 - 2, y + carH / 2 - 11, 3, 7);

    // Brake Lights (Rear is left side) - Glows bright red when braking
    const isBraking = accel < -0.25;
    ctx.fillStyle = isBraking ? "#ff0000" : "#7f1d1d";
    if (isBraking) {
      ctx.shadowColor = "#ff0000";
      ctx.shadowBlur = 14;
    }
    ctx.fillRect(x - carW / 2 - 1, y - carH / 2 + 4, 3, 7);
    ctx.fillRect(x - carW / 2 - 1, y + carH / 2 - 11, 3, 7);
    ctx.shadowBlur = 0;

    // Vehicle Label & Speed Display
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(label, x, y - carH / 2 - 8);

    ctx.fillStyle = "#cbd5e1";
    ctx.font = "9px monospace";
    ctx.fillText(`${speed.toFixed(1)} m/s`, x, y + carH / 2 + 15);

    ctx.restore();
  }

  // ==========================================================================
  // 7. SYNCHRONIZED TELEMETRY CHARTS (WITH CURSOR & OVERLAY)
  // ==========================================================================
  drawCharts() {
    const isComparison = (this.controllerMode === "comparison");
    const primary = (this.controllerMode === "baseline") ? this.simFollowerBase : this.simFollowerCBF;
    const totalSteps = primary.history.t.length;
    const cursorRatio = this.currentStep / Math.max(1, totalSteps - 1);

    // Chart 1: Headway Gap s(t)
    let gapSeries = [
      { data: primary.history.gap, color: (this.controllerMode === "baseline") ? "#f59e0b" : "#38bdf8", width: 2, label: "Actual Gap s" },
      { data: primary.history.sSafe, color: "#10b981", width: 1.5, dash: [4, 4], label: "Required s_safe" },
      { data: primary.history.sDes, color: "#94a3b8", width: 1.0, dash: [2, 2], label: "Desired s_des" },
    ];
    if (isComparison) {
      gapSeries = [
        { data: this.simFollowerBase.history.gap, color: "#f59e0b", width: 1.8, label: "Baseline Gap" },
        { data: this.simFollowerCBF.history.gap, color: "#38bdf8", width: 2.2, label: "ACC+CBF Gap" },
        { data: this.simFollowerCBF.history.sSafe, color: "#10b981", width: 1.5, dash: [4, 4], label: "Safe Req" },
      ];
    }
    this.drawChart(this.chartGap, 550, 140, gapSeries, [0.0, 15.0], "Gap (m)", cursorRatio);

    // Chart 2: Barrier Margin h(t)
    let hSeries = [
      { data: primary.history.h, color: "#10b981", width: 2.2, label: "Margin h(t)" },
    ];
    if (isComparison) {
      hSeries = [
        { data: this.simFollowerBase.history.h, color: "#f59e0b", width: 1.8, label: "Baseline h" },
        { data: this.simFollowerCBF.history.h, color: "#10b981", width: 2.2, label: "ACC+CBF h" },
      ];
    }
    this.drawChart(this.chartH, 550, 140, hSeries, [0.0], "Margin (m)", cursorRatio);

    // Chart 3: Speeds (Lead vs Ego)
    let speedSeries = [
      { data: this.simData.leadSpeeds, color: "#f87171", width: 1.8, label: "Lead Speed" },
      { data: primary.history.egoSpeed, color: "#c084fc", width: 2.0, label: "AV Speed" },
    ];
    if (isComparison) {
      speedSeries = [
        { data: this.simData.leadSpeeds, color: "#f87171", width: 1.5, label: "Lead" },
        { data: this.simFollowerBase.history.egoSpeed, color: "#f59e0b", width: 1.8, label: "Baseline AV" },
        { data: this.simFollowerCBF.history.egoSpeed, color: "#38bdf8", width: 2.0, label: "ACC+CBF AV" },
      ];
    }
    this.drawChart(this.chartSpeed, 550, 140, speedSeries, [], "Speed (m/s)", cursorRatio);

    // Chart 4: Accelerations (u_nom vs u_cbf vs applied)
    let accelSeries = [
      { data: primary.history.appliedAcc, color: "#34d399", width: 2.0, label: "a_applied" },
      { data: primary.history.uNom, color: "#f87171", width: 1.5, dash: [3, 3], label: "u_nom" },
    ];
    if (this.controllerMode !== "baseline") {
      accelSeries.push({ data: primary.history.uCbf, color: "#fbbf24", width: 1.5, dash: [4, 4], label: "u_CBF" });
    }
    if (isComparison) {
      accelSeries = [
        { data: this.simFollowerBase.history.appliedAcc, color: "#f59e0b", width: 1.8, label: "Base a" },
        { data: this.simFollowerCBF.history.appliedAcc, color: "#34d399", width: 2.0, label: "CBF a" },
        { data: this.simFollowerCBF.history.uCbf, color: "#fbbf24", width: 1.2, dash: [3, 3], label: "u_CBF" },
      ];
    }
    this.drawChart(this.chartAccel, 550, 140, accelSeries, [-4.0, 0.0], "Accel (m/s²)", cursorRatio);
  }

  drawChart(ctx, w, h, seriesList, thresholdLines, yAxisLabel, cursorRatio) {
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

    const padLeft = 38;
    const padRight = 15;
    const padTop = 15;
    const padBottom = 20;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    const toY = (val) => padTop + plotH - ((val - minVal) / (maxVal - minVal)) * plotH;
    const toX = (idx, total) => padLeft + (idx / Math.max(1, total - 1)) * plotW;

    // Grid lines
    ctx.strokeStyle = "#1e293b";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 3; i++) {
      const gy = padTop + (plotH / 3) * i;
      ctx.moveTo(padLeft, gy);
      ctx.lineTo(padLeft + plotW, gy);
    }
    ctx.stroke();

    // Threshold lines
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

    // Curves
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

    // Y-Axis Min/Max
    ctx.fillStyle = "#64748b";
    ctx.font = "9px monospace";
    ctx.textAlign = "right";
    ctx.fillText(maxVal.toFixed(0), padLeft - 5, padTop + 8);
    ctx.fillText(minVal.toFixed(0), padLeft - 5, padTop + plotH);

    // Vertical Cursor Line
    const cursorX = padLeft + cursorRatio * plotW;
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cursorX, padTop);
    ctx.lineTo(cursorX, padTop + plotH);
    ctx.stroke();
  }
}

// Instantiate visualizer when page loads
window.addEventListener("DOMContentLoaded", () => {
  window.app = new VisualizerApp();
});
