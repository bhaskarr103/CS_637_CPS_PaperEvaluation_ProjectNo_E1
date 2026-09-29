# Lightweight Interactive Car-Following Visualizer: Prototype 3

**Author:** Antigravity Research Engineering  
**Context:** CBF-RL Autonomous Car-Following (Yang et al., arXiv:2510.14959)  
**File Location:** [`E:\CPS Project\Prototype 3\visualizer.html`](file:///E:/CPS%20Project/Prototype%203/visualizer.html)  
**Launcher:** [`E:\CPS Project\Prototype 3\launch_visualizer.py`](file:///E:/CPS%20Project/Prototype%203/launch_visualizer.py)  

---

## 1. Framework Architecture & Design Rationale

### Why This Architecture Was Chosen:
1. **Lightweight & Low-Overhead:**  
   Unlike heavy 3D game engines (Unreal Engine 5 / Cosys-AirSim) that require minutes to load and gigabytes of GPU VRAM, this visualizer uses a **custom HTML5 Canvas + Tailwind CSS** architecture. It renders at a solid 60 FPS in any browser with instant startup.
2. **Zero External Server / CORS Dependency:**  
   The complete 10 Hz simulation rollouts across all scenarios and ablations are precomputed with 100% mathematical fidelity using Prototype 3's PyTorch actor-critic network and embedded directly into `visualizer.html`. It can be launched directly via `file:///` without local HTTP server blockers.
3. **Exact Mathematical Fidelity:**  
   The visualization does **not** approximate car physics. Every frame directly displays the state vectors computed by [`cbf_core.py`](file:///E:/CPS%20Project/Prototype%203/cbf_core.py) and [`env.py`](file:///E:/CPS%20Project/Prototype%203/env.py).

---

## 2. Reused Mathematical Equations & Hyperparameters

The simulation strictly enforces the research formulation of Prototype 3:

| Mathematical Concept | Exact Formula Reused from `cbf_core.py` | Parameter Values |
| :--- | :--- | :--- |
| **Control Barrier Function** | $h(s, v_{\text{ego}}) = s - (T_{\min} \cdot v_{\text{ego}} + D_{\min})$ | $T_{\min} = 2.0\,\text{s}, \; D_{\min} = 15.0\,\text{m}$ |
| **Discrete Barrier Limit** | $u_{\text{CBF}}^{\max} = \frac{\Delta v + \alpha \cdot h}{T_{\min}}$ | $\alpha = 0.1\,\text{s}^{-1}, \; \Delta v = v_{\text{lead}} - v_{\text{ego}}$ |
| **Closed-Form Projection** | $u_{\text{safe}} = \text{clip}\left(\min(u_{\text{policy}}, u_{\text{CBF}}^{\max}),\, u_{\min},\, u_{\max}\right)$ | $u_{\min} = -4.0\,\text{m/s}^2, \; u_{\max} = +2.5\,\text{m/s}^2$ |
| **Kinematic Forward Euler** | $v_{\text{ego}}[k+1] = \max(0,\, v_{\text{ego}}[k] + u_{\text{applied}}[k] \cdot \Delta t)$<br/>$s[k+1] = \max(0,\, s[k] + (v_{\text{lead}}[k] - v_{\text{ego}}[k]) \cdot \Delta t)$ | $\Delta t = 0.1\,\text{s}$ (10 Hz) |

---

## 3. Demonstration Scenarios

| Scenario ID | Name | Category Label | Description |
| :---: | :--- | :--- | :--- |
| **1** | Normal Highway Cruising | `[SYNTHETIC DEMONSTRATION]` | Lead cruises at ~80 km/h with subtle wave noise. Follower maintains smooth headway equilibrium. |
| **2** | Moderate Lead Braking | `[SYNTHETIC DEMONSTRATION]` | Lead decelerates steadily from 86 km/h to 58 km/h at $-2.0\,\text{m/s}^2$. Tests standard deceleration tracking. |
| **3** | Sudden Emergency Braking | `[SYNTHETIC DEMONSTRATION]` | Lead slams brakes at $-3.8\,\text{m/s}^2$ (near physical limit). Severe collision risk if unshielded or uninternalized. |
| **4** | Close Initial Cut-In | `[SYNTHETIC DEMONSTRATION]` | Lead cuts in at close range ($s_0 = 26\,\text{m}$ at 80 km/h, $h_0 = -33\,\text{m}$). Tests recovery from initial penetration under $u_{\min}$. |
| **5** | Vicolungo Trajectory 3 | `[VICOLUNGO REAL-DATA REPLAY]` | Real recorded human lead speed from Italian A4 highway (Feb 2019). Starts safe ($h_0 = +51.3\,\text{m}$). Tests Forward Invariance preservation. |
| **6** | Vicolungo Trajectory 1 | `[VICOLUNGO REAL-DATA REPLAY]` | Real recorded human lead speed from Italian A4 highway (Feb 2019). Starts penetrated ($h_0 = -34.1\,\text{m}$). Tests Safe Recovery under actuator saturation. |

---

## 4. Policy Ablation Comparison in the Visualizer

For every scenario, the viewer can toggle between the 4 trained policy ablations:
1. **Dual CBF-RL (Shielded + CBF Reward Penalty):**  
   The policy internalizes safety during training. Its proposed action $u_{\text{policy}}$ tracks $u_{\text{safe}}$ smoothly. Even with **Filter OFF**, it avoids collisions across all emergency braking and cut-in scenarios.
2. **Filter Only (Shielded in Training, No CBF Reward):**  
   The policy was shielded during training but never penalized for unsafe action proposals. When tested with **Filter ON**, it is safe; when toggled to **Filter OFF**, it proposes excessive throttle and causes physical collisions.
3. **Reward Only (CBF Penalty, No Shield in Training):**  
   Soft penalties encourage safety, but lack hard boundary guarantees during aggressive closing transients.
4. **Nominal RL (Unconstrained PPO Baseline):**  
   Trades off headway error without barrier awareness. Collides when lead brakes abruptly under Filter OFF.

---

## 5. One-Click Launch Instructions

To launch the visualizer in your default browser:

```powershell
& "E:\CPS Project\venv\Scripts\python.exe" "E:\CPS Project\Prototype 3\launch_visualizer.py"
```

Or open directly in any browser:
```text
file:///E:/CPS Project/Prototype 3/visualizer.html
```

### Keyboard Shortcuts in the Visualizer:
- **`Spacebar`:** Play / Pause simulation playback.
- **`Right Arrow`:** Step forward one frame (0.1s).
- **`Left Arrow`:** Step backward one frame.
- **Timeline Slider:** Click or drag to scrub across the episode.
