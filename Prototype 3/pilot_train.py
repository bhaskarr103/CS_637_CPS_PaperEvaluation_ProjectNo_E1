"""Phase 2 Minimal Pilot Runner for Prototype 3 (CBF-RL).

Executes a small, rigorous sanity pilot:
- 1 seed (seed=42)
- 15,000 environment steps per variant
- 4 variants: Nominal, Filter Only, Reward Only, Dual CBF-RL
- Comprehensive metric logging per episode:
    * episode_return, episode_len, collision, min_h, min_gap, pct_h_safe
    * intervention_rate, mean_correction, infeasible_count, mean_v_ego
    * mean_u_policy, mean_u_safe, mean_u_diff (for Filter Only & Dual)
- Small Vicolungo Evaluation:
    * 5 selected valid trajectories
    * Evaluated under both Filter ON and Filter OFF (deterministic actions)
- Saves CSV results in results/ directory
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch

# Ensure Prototype 3 directory is in Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from env import CarFollowingEnv
from ppo import PPOAgent, RolloutBuffer


def train_variant(
    variant_name: str,
    use_filter: bool,
    use_reward: bool,
    total_steps: int = 15000,
    rollout_size: int = 500,
    seed: int = 42,
    config: Optional[Config] = None,
) -> Tuple[PPOAgent, pd.DataFrame]:
    """Trains a single PPO agent variant for total_steps in synthetic mode."""
    cfg = config if config is not None else Config()

    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = CarFollowingEnv(
        config=cfg,
        mode="synthetic",
        use_cbf_filter=use_filter,
        use_cbf_reward=use_reward,
    )

    agent = PPOAgent(
        obs_dim=env.obs_dim,
        act_dim=env.act_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64,
        device="cpu",
    )

    buffer = RolloutBuffer(capacity=rollout_size, obs_dim=env.obs_dim, act_dim=env.act_dim)

    # Logging structures
    episode_records: List[Dict[str, Any]] = []

    # Episode accumulators
    obs, info = env.reset(seed=seed)
    ep_return = 0.0
    ep_len = 0
    ep_collisions = 0
    ep_h_vals = [info["h"]]
    ep_gaps = [info["s"]]
    ep_v_egos = [info["v_ego"]]
    ep_interventions = 0
    ep_corrections = []
    ep_infeasibles = 0
    ep_u_policies = []
    ep_u_safes = []

    global_step = 0
    update_count = 0
    start_time = time.time()

    while global_step < total_steps:
        # 1. Query agent
        action, log_prob, value = agent.select_action(obs, deterministic=False)

        # 2. Step environment
        # Note: action is the raw policy sample in [-1, 1]
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        # 3. Store into rollout buffer
        # Crucial: Store the policy action 'action' and its true log_prob!
        # Do NOT store u_safe!
        buffer.add(
            obs=obs,
            action=action,
            log_prob=log_prob,
            reward=reward,
            value=value,
            done=done,
        )

        # Telemetry accumulation
        ep_return += reward
        ep_len += 1
        ep_h_vals.append(step_info["h"])
        ep_gaps.append(step_info["s"])
        ep_v_egos.append(step_info["v_ego"])
        ep_u_policies.append(step_info["u_policy"])
        ep_u_safes.append(step_info["u_safe"])

        if step_info["is_intervened"]:
            ep_interventions += 1
            ep_corrections.append(abs(step_info["u_policy"] - step_info["u_safe"]))
        else:
            ep_corrections.append(0.0)

        if step_info["is_infeasible"]:
            ep_infeasibles += 1

        if step_info["is_collision"]:
            ep_collisions += 1

        global_step += 1
        obs = next_obs

        # Handle episode termination
        if done:
            h_arr = np.array(ep_h_vals)
            ep_record = {
                "variant": variant_name,
                "global_step": global_step,
                "episode": len(episode_records) + 1,
                "ep_return": float(ep_return),
                "ep_len": ep_len,
                "collisions": ep_collisions,
                "min_h": float(h_arr.min()),
                "min_gap": float(np.min(ep_gaps)),
                "pct_h_safe": float(np.mean(h_arr >= 0.0) * 100.0),
                "intervention_rate": float((ep_interventions / max(1, ep_len)) * 100.0),
                "mean_correction": float(np.mean(ep_corrections)),
                "infeasible_count": ep_infeasibles,
                "mean_v_ego": float(np.mean(ep_v_egos)),
                "mean_u_policy": float(np.mean(ep_u_policies)),
                "mean_u_safe": float(np.mean(ep_u_safes)),
                "mean_u_diff": float(np.mean(np.array(ep_u_policies) - np.array(ep_u_safes))),
            }
            episode_records.append(ep_record)

            # Reset accumulators for next episode
            obs, info = env.reset()
            ep_return = 0.0
            ep_len = 0
            ep_collisions = 0
            ep_h_vals = [info["h"]]
            ep_gaps = [info["s"]]
            ep_v_egos = [info["v_ego"]]
            ep_interventions = 0
            ep_corrections = []
            ep_infeasibles = 0
            ep_u_policies = []
            ep_u_safes = []

        # Buffer update check
        if buffer.ptr >= buffer.capacity:
            # Estimate value of final state for GAE bootstrapping
            if done:
                last_val = 0.0
            else:
                _, _, last_val = agent.select_action(obs, deterministic=True)

            buffer.compute_gae(last_val, gamma=cfg.gamma if hasattr(cfg, "gamma") else 0.99)
            loss_info = agent.update(buffer)
            buffer.reset()
            update_count += 1

    train_time = time.time() - start_time
    df_records = pd.DataFrame(episode_records)
    print(
        f"  Completed [{variant_name}] in {train_time:.1f}s | "
        f"Episodes: {len(df_records)} | Updates: {update_count} | "
        f"Last 5 Return: {df_records['ep_return'].tail(5).mean():.2f}"
    )

    return agent, df_records


def evaluate_vicolungo_pilot(
    agents: Dict[str, PPOAgent],
    trajectories: List[int] = [1, 2, 3, 4, 5],
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """Evaluates trained agents on 5 selected Vicolungo trajectories.

    Tests both Filter ON and Filter OFF modes using deterministic actions.
    """
    cfg = config if config is not None else Config()
    eval_records: List[Dict[str, Any]] = []

    for name, agent in agents.items():
        # Evaluate under both deployment regimes
        for filter_mode in [True, False]:
            mode_label = "Filter_ON" if filter_mode else "Filter_OFF"

            for tid in trajectories:
                env = CarFollowingEnv(
                    config=cfg,
                    mode="vicolungo",
                    use_cbf_filter=filter_mode,
                    use_cbf_reward=False,  # Evaluation does not apply reward
                )

                obs, info = env.reset(seed=100 + tid, options={"traj_id": tid})
                h_vals = [info["h"]]
                gaps = [info["s"]]
                interventions = 0
                infeasibles = 0
                corrections = []
                collisions = 0
                step_count = 0

                done = False
                while not done:
                    # Deterministic evaluation action
                    action, _, _ = agent.select_action(obs, deterministic=True)
                    obs, reward, terminated, truncated, step_info = env.step(action)
                    done = terminated or truncated

                    h_vals.append(step_info["h"])
                    gaps.append(step_info["s"])
                    step_count += 1

                    if step_info["is_intervened"]:
                        interventions += 1
                        corrections.append(abs(step_info["u_policy"] - step_info["u_safe"]))
                    else:
                        corrections.append(0.0)

                    if step_info["is_infeasible"]:
                        infeasibles += 1

                    if step_info["is_collision"]:
                        collisions += 1

                h_arr = np.array(h_vals)
                eval_records.append({
                    "variant": name,
                    "deploy_mode": mode_label,
                    "traj_id": tid,
                    "steps": step_count,
                    "collision": collisions,
                    "min_h": float(h_arr.min()),
                    "min_gap": float(np.min(gaps)),
                    "pct_h_safe": float(np.mean(h_arr >= 0.0) * 100.0),
                    "intervention_rate": float((interventions / max(1, step_count)) * 100.0),
                    "mean_correction": float(np.mean(corrections)),
                    "infeasible_count": infeasibles,
                })

    return pd.DataFrame(eval_records)


def run_pilot():
    """Main orchestration for Phase 2 Minimal Pilot."""
    print("=" * 70)
    print("STARTING PHASE 2 — PPO MINIMAL PILOT")
    print("1 Seed | 15,000 steps per variant | 4 Variants | Synthetic Train")
    print("=" * 70)

    cfg = Config()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    variant_specs = [
        ("Nominal", False, False),
        ("Filter_Only", True, False),
        ("Reward_Only", False, True),
        ("Dual_CBF_RL", True, True),
    ]

    trained_agents: Dict[str, PPOAgent] = {}
    all_train_dfs: List[pd.DataFrame] = []

    pilot_start = time.time()

    for name, use_filt, use_rew in variant_specs:
        print(f"\nTraining Variant: {name} (Filter={use_filt}, Reward={use_rew})...")
        agent, df_train = train_variant(
            variant_name=name,
            use_filter=use_filt,
            use_reward=use_rew,
            total_steps=15000,
            rollout_size=500,
            seed=42,
            config=cfg,
        )
        trained_agents[name] = agent
        all_train_dfs.append(df_train)

    total_train_time = time.time() - pilot_start
    print(f"\nAll 4 variants trained in {total_train_time:.1f} seconds total.")

    # Combine and save training logs
    df_all_train = pd.concat(all_train_dfs, ignore_index=True)
    train_log_path = cfg.results_dir / "pilot_training_log.csv"
    df_all_train.to_csv(train_log_path, index=False)
    print(f"Saved training log: {train_log_path}")

    # Small Vicolungo Evaluation on 5 trajectories
    print("\nRunning Small Vicolungo Evaluation on 5 trajectories (Filter ON & OFF)...")
    df_eval = evaluate_vicolungo_pilot(trained_agents, trajectories=[1, 2, 3, 4, 5], config=cfg)
    eval_log_path = cfg.results_dir / "pilot_vicolungo_eval.csv"
    df_eval.to_csv(eval_log_path, index=False)
    print(f"Saved Vicolungo evaluation log: {eval_log_path}")

    # Summary table
    print("\n" + "=" * 70)
    print("PILOT TRAINING SUMMARY (Last 10 Episodes Averages)")
    print("=" * 70)
    summary_rows = []
    for name in ["Nominal", "Filter_Only", "Reward_Only", "Dual_CBF_RL"]:
        sub = df_all_train[df_all_train["variant"] == name].tail(10)
        summary_rows.append({
            "Variant": name,
            "Episodes": len(df_all_train[df_all_train["variant"] == name]),
            "Mean Return": f"{sub['ep_return'].mean():.2f}",
            "Min h (m)": f"{sub['min_h'].mean():.2f}",
            "Min Gap (m)": f"{sub['min_gap'].mean():.2f}",
            "% h >= 0": f"{sub['pct_h_safe'].mean():.1f}%",
            "Intervention Rate": f"{sub['intervention_rate'].mean():.1f}%",
            "Mean Correction": f"{sub['mean_correction'].mean():.3f}",
            "Collisions": int(sub['collisions'].sum()),
        })
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("VICOLUNGO PILOT EVALUATION SUMMARY (5 Trajectories)")
    print("=" * 70)
    eval_summary = df_eval.groupby(["variant", "deploy_mode"])[
        ["collision", "min_h", "min_gap", "pct_h_safe", "intervention_rate", "mean_correction"]
    ].mean().reset_index()
    print(eval_summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("PILOT COMPLETED SUCCESSFULLY. STOPPING AS REQUIRED.")
    print("=" * 70)


if __name__ == "__main__":
    run_pilot()
