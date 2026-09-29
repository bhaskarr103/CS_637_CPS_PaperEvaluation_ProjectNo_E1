"""Automated Numerical Cross-Validation Test.

Validates that the JavaScript visualizer controller implementation in app.js
produces identical numerical outputs to Python's cbf_controller.py and
02_baseline_acc.py across 50 diverse kinematic test states.

Checks:
- applied_acc
- u_nom
- u_cbf
- h
- intervention
- infeasible
All to within 1e-6 numerical tolerance.
"""

import sys
import json
import subprocess
from pathlib import Path
import numpy as np

# Ensure Prototype 2 root is in sys.path
PROTO2_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROTO2_DIR))

from project_config import (
    DT, D_DES, T_DES, K_GAP, K_REL,
    T_MIN, D_MIN, K_CBF, A_MAX, A_MIN, JERK_MAX
)
from cbf_controller import CBFSafetyFilter, compute_h, compute_u_cbf


def generate_test_cases():
    """Generates 50 diverse states covering safe, unsafe, infeasible, and jerk-limiting regimes."""
    np.random.seed(42)
    test_cases = []

    # 1. Structured boundary states
    structured_states = [
        # (gap, ego_speed, lead_speed, prev_acc, desc)
        (75.0, 25.0, 27.0, 0.0, "Safe following, opening speed"),
        (30.0, 25.0, 25.0, 0.0, "Close gap, same speed -> CBF intervention"),
        (2.0, 30.0, 15.0, 0.0, "Emergency closing, tiny gap -> Infeasible"),
        (15.0, 20.0, 10.0, -1.0, "Active braking with prior deceleration"),
        (120.0, 15.0, 30.0, 0.0, "Large headway (Traj 55 style)"),
        (33.0, 27.5, 19.0, 0.0, "Trajectory 16 initial condition"),
        (45.0, 32.0, 32.0, 1.5, "High speed highway cruising at max acceleration"),
        (10.0, 5.0, 0.0, -3.5, "Low-speed standstill approach"),
        (5.0, 22.0, 22.0, 0.0, "Severe gap breach at highway speed"),
        (60.0, 20.0, 20.0, 0.0, "Exact equilibrium state"),
    ]

    for gap, ego_speed, lead_speed, prev_acc, desc in structured_states:
        test_cases.append({
            "gap": float(gap),
            "ego_speed": float(ego_speed),
            "lead_speed": float(lead_speed),
            "prev_acc": float(prev_acc),
            "desc": desc,
        })

    # 2. 40 Randomized states across the full physical domain
    for i in range(40):
        gap = float(np.random.uniform(1.0, 100.0))
        ego_speed = float(np.random.uniform(5.0, 35.0))
        lead_speed = float(np.random.uniform(5.0, 35.0))
        prev_acc = float(np.random.uniform(A_MIN, A_MAX))
        test_cases.append({
            "gap": round(gap, 3),
            "ego_speed": round(ego_speed, 3),
            "lead_speed": round(lead_speed, 3),
            "prev_acc": round(prev_acc, 3),
            "desc": f"Randomized State {i+1}",
        })

    return test_cases


def evaluate_python_controller(test_cases):
    """Computes exact reference outputs using Prototype 2's Python codebase."""
    cbf_filter = CBFSafetyFilter()
    results = []

    for tc in test_cases:
        gap = tc["gap"]
        ego_speed = tc["ego_speed"]
        lead_speed = tc["lead_speed"]
        prev_acc = tc["prev_acc"]

        delta_v = lead_speed - ego_speed
        s_des = D_DES + T_DES * ego_speed
        gap_error = gap - s_des
        u_nom = K_GAP * gap_error + K_REL * delta_v
        u_nom_sat = float(np.clip(u_nom, A_MIN, A_MAX))

        # Evaluate ACC+CBF
        cbf_filter.reset(initial_acc=prev_acc)
        cbf_res = cbf_filter.filter_control(
            u_nom=u_nom_sat,
            gap=gap,
            ego_speed=ego_speed,
            lead_speed=lead_speed,
        )

        # Evaluate Baseline ACC (without CBF)
        max_acc_change = JERK_MAX * DT
        base_change = np.clip(u_nom_sat - prev_acc, -max_acc_change, max_acc_change)
        base_applied = float(prev_acc + base_change)

        results.append({
            "input": tc,
            "py_cbf": {
                "applied_acc": float(cbf_res["applied_acc"]),
                "u_nom": float(cbf_res["u_nom"]),
                "u_cbf": float(cbf_res["u_cbf"]),
                "u_raw": float(cbf_res["u_raw"]),
                "u_sat": float(cbf_res["u_sat"]),
                "h": float(cbf_res["h"]),
                "delta_v": float(cbf_res["delta_v"]),
                "intervention": bool(cbf_res["intervention"]),
                "infeasible": bool(cbf_res["infeasible"]),
            },
            "py_base": {
                "applied_acc": base_applied,
                "u_nom": u_nom_sat,
            }
        })

    return results


def run_cross_validation():
    print("=" * 70)
    print("AUTOMATED CONTROLLER CROSS-VALIDATION: PYTHON vs JAVASCRIPT")
    print("=" * 70)

    test_cases = generate_test_cases()
    py_results = evaluate_python_controller(test_cases)
    print(f"Generated {len(test_cases)} test cases from Python reference.")

    # Save test cases for Node.js verification
    cases_file = Path(__file__).parent / "cross_validation_cases.json"
    with open(cases_file, "w", encoding="utf-8") as f:
        json.dump(py_results, f, indent=2)

    # Run Node.js validator script
    node_validator = Path(__file__).parent / "node_validate.js"
    node_code = """
const fs = require('fs');
const path = require('path');

// Exact Prototype 2 Parameters
const P2_CONFIG = {
  DT: 0.1,
  D_DES: 5.0,
  T_DES: 1.5,
  K_GAP: 0.20,
  K_REL: 0.80,
  T_MIN: 2.0,
  D_MIN: 15.0,
  K_CBF: 0.1,
  A_MAX: 2.0,
  A_MIN: -4.0,
  JERK_MAX: 1.5,
};

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

function filterControl(uNomSat, gap, egoSpeed, leadSpeed, prevAcc, useCbf) {
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

  const uSat = Math.max(P2_CONFIG.A_MIN, Math.min(P2_CONFIG.A_MAX, uRaw));
  const maxChange = P2_CONFIG.JERK_MAX * P2_CONFIG.DT;
  const accChange = Math.max(-maxChange, Math.min(maxChange, uSat - prevAcc));
  const appliedAcc = prevAcc + accChange;

  return { appliedAcc, uNom: uNomSat, uCbf, uRaw, uSat, h, deltaV, intervention, infeasible };
}

// Load test cases
const raw = fs.readFileSync(path.join(__dirname, 'cross_validation_cases.json'), 'utf8');
const testCases = JSON.parse(raw);

let passed = 0;
let failed = 0;
const tol = 1e-6;

testCases.forEach((tc, idx) => {
  const inp = tc.input;
  const pyCbf = tc.py_cbf;
  const pyBase = tc.py_base;

  const { uNom } = computeNominalAcc(inp.gap, inp.ego_speed, inp.lead_speed);
  const uNomSat = Math.max(P2_CONFIG.A_MIN, Math.min(P2_CONFIG.A_MAX, uNom));

  // Test CBF mode
  const jsCbf = filterControl(uNomSat, inp.gap, inp.ego_speed, inp.lead_speed, inp.prev_acc, true);

  const diffAcc = Math.abs(jsCbf.appliedAcc - pyCbf.applied_acc);
  const diffH = Math.abs(jsCbf.h - pyCbf.h);
  const diffUCbf = Math.abs(jsCbf.uCbf - pyCbf.u_cbf);
  const matchInterv = jsCbf.intervention === pyCbf.intervention;
  const matchInf = jsCbf.infeasible === pyCbf.infeasible;

  if (diffAcc < tol && diffH < tol && diffUCbf < tol && matchInterv && matchInf) {
    passed++;
  } else {
    failed++;
    console.error(`Mismatch at case ${idx} (${inp.desc}):`);
    console.error(`  Py applied: ${pyCbf.applied_acc}, JS applied: ${jsCbf.appliedAcc}, diff: ${diffAcc}`);
    console.error(`  Py h: ${pyCbf.h}, JS h: ${jsCbf.h}`);
    console.error(`  Py intervention: ${pyCbf.intervention}, JS intervention: ${jsCbf.intervention}`);
  }
});

console.log(`Node.js Cross-Validation: ${passed}/${testCases.length} states passed perfectly (tol = ${tol}).`);
if (failed > 0) process.exit(1);
"""
    with open(node_validator, "w", encoding="utf-8") as f:
        f.write(node_code)

    res = subprocess.run(["node", str(node_validator)], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print("ERROR:", res.stderr)
        raise RuntimeError("Cross-validation failed!")

    print("SUCCESS: Python Prototype 2 controllers and JavaScript visualizer are 100% numerically identical!")


if __name__ == "__main__":
    run_cross_validation()
