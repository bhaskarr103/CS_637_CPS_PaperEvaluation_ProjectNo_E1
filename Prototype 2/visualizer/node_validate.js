
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
