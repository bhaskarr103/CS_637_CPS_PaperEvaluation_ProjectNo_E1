"""Phase 3 Full Experiment: CBF-RL Multi-Seed Training & 60-Trajectory Vicolungo Benchmark.

Executes:
- 4 Ablations: Nominal, Filter Only, Reward Only, Dual CBF-RL
- 3 Random Seeds: 42, 100, 2024
- 150,000 steps per run (total 1,800,000 steps across 12 runs)
- Training domain: 100% Synthetic procedural generator
- Model checkpointing: Saves model_state_dict to results/checkpoints/
- Full Evaluation Benchmark:
    * 60 valid real-world trajectories from OpenACC Vicolungo (A4 Highway, Feb 2019)
    * Both Filter ON and Filter OFF modes (internalization test)
    * Disaggregated analysis: Subset A (N=9, Initially Safe) vs Subset B (N=51, Initially Penetrated)
- Saves full results to results/phase3_training_log.csv and results/phase3_vicolungo_eval.csv
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
import torch

# Ensure Prototype 3 is in sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from env import CarFollowingEnv
from ppo import PPOAgent, RolloutBuffer
from cbf_core import compute_h


def train_single_agent(
    variant_name: str,
    use_filter: bool,
    use_reward: bool,
    seed: int,
    total_steps: int = 150000,
    rollout_size: int = 500,
    config: Optional[Config] = None,
) -> Tuple[PPOAgent, pd.DataFrame]:
    """Trains a single PPO agent variant for total_steps in synthetic mode."""
    cfg = config if config is not None else Config()

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

    episode_records: List[Dict[str, Any]] = []

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
    last_print_step = 0
    start_time = time.time()

    while global_step < total_steps:
        action, log_prob, value = agent.select_action(obs, deterministic=False)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated

        # Rollout buffer strictly stores policy action and its true log-prob
        buffer.add(
            obs=obs,
            action=action,
            log_prob=log_prob,
            reward=reward,
            value=value,
            done=done,
        )

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

        if done:
            h_arr = np.array(ep_h_vals)
            ep_record = {
                "variant": variant_name,
                "seed": seed,
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

        if buffer.ptr >= buffer.capacity:
            last_val = 0.0 if done else agent.select_action(obs, deterministic=True)[2]
            buffer.compute_gae(last_val, gamma=0.99)
            agent.update(buffer)
            buffer.reset()
            update_count += 1

        # Periodic progress report every 25,000 steps
        if global_step - last_print_step >= 25000 or global_step == total_steps:
            last_print_step = global_step
            recent = episode_records[-10:] if episode_records else []
            r_mean = np.mean([r["ep_return"] for r in recent]) if recent else 0.0
            inv_mean = np.mean([r["intervention_rate"] for r in recent]) if recent else 0.0
            safe_mean = np.mean([r["pct_h_safe"] for r in recent]) if recent else 0.0
            print(
                f"    [{variant_name} | Seed {seed}] Step {global_step:6d}/{total_steps} | "
                f"Return: {r_mean:6.2f} | Safe%: {safe_mean:5.1f}% | Interv%: {inv_mean:5.1f}% | "
                f"Elapsed: {time.time() - start_time:.1f}s"
            )

    train_time = time.time() - start_time
    df_records = pd.DataFrame(episode_records)

    # Save model checkpoint
    ckpt_dir = cfg.results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / f"{variant_name}_seed{seed}.pt"
    torch.save(
        {
            "variant": variant_name,
            "seed": seed,
            "global_step": global_step,
            "model_state_dict": agent.ac.state_dict(),
        },
        ckpt_file,
    )
    print(f"  --> Saved checkpoint: {ckpt_file.name} (Time: {train_time:.1f}s)")

    return agent, df_records


def evaluate_agent_on_vicolungo(
    agent: PPOAgent,
    variant_name: str,
    seed: int,
    valid_trajectory_ids: List[int],
    config: Optional[Config] = None,
) -> List[Dict[str, Any]]:
    """Evaluates a single trained agent checkpoint on all 60 Vicolungo trajectories.

    Runs both Filter ON and Filter OFF modes with deterministic actions.
    """
    cfg = config if config is not None else Config()
    eval_records: List[Dict[str, Any]] = []

    for filter_mode in [True, False]:
        mode_label = "Filter_ON" if filter_mode else "Filter_OFF"

        for tid in valid_trajectory_ids:
            env = CarFollowingEnv(
                config=cfg,
                mode="vicolungo",
                use_cbf_filter=filter_mode,
                use_cbf_reward=False,
            )

            obs, info = env.reset(seed=1000 + tid, options={"traj_id": tid})
            s_0 = float(info["s"])
            v_ego_0 = float(info["v_ego"])
            h_0 = float(info["h"])
            is_initially_safe = bool(h_0 >= 0.0)

            h_vals = [h_0]
            gaps = [s_0]
            interventions = 0
            infeasibles = 0
            corrections = []
            collisions = 0
            jerks = []
            step_count = 0
            recovered_step: Optional[int] = 0 if is_initially_safe else None

            done = False
            while not done:
                action, _, _ = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated

                curr_h = step_info["h"]
                h_vals.append(curr_h)
                gaps.append(step_info["s"])
                step_count += 1

                # Check recovery point for initially unsafe trajectories
                if not is_initially_safe and recovered_step is None and curr_h >= 0.0:
                    recovered_step = step_count

                if step_info["is_intervened"]:
                    interventions += 1
                    corrections.append(abs(step_info["u_policy"] - step_info["u_safe"]))
                else:
                    corrections.append(0.0)

                if step_info["is_infeasible"]:
                    infeasibles += 1

                if step_info["is_collision"]:
                    collisions += 1

                # Longitudinal comfort jerk: |u - u_prev| / dt
                jerks.append(abs(step_info["u_applied"] - env.u_prev) / cfg.dt)

            h_arr = np.array(h_vals)
            min_gap_val = float(np.min(gaps))
            recovery_time = (
                0.0
                if is_initially_safe
                else (float(recovered_step * cfg.dt) if recovered_step is not None else np.nan)
            )

            eval_records.append({
                "variant": variant_name,
                "seed": seed,
                "deploy_mode": mode_label,
                "traj_id": tid,
                "is_initially_safe": is_initially_safe,
                "s_0": s_0,
                "v_ego_0": v_ego_0,
                "h_0": h_0,
                "steps": step_count,
                "duration_s": float(step_count * cfg.dt),
                "collision": 1 if min_gap_val <= 0.0 else 0,
                "min_h": float(h_arr.min()),
                "min_gap": min_gap_val,
                "mean_h": float(np.mean(h_arr)),
                "pct_h_safe": float(np.mean(h_arr >= 0.0) * 100.0),
                "recovery_time_s": recovery_time,
                "intervention_rate": float((interventions / max(1, step_count)) * 100.0),
                "mean_correction": float(np.mean(corrections)),
                "max_correction": float(np.max(corrections) if corrections else 0.0),
                "infeasible_count": infeasibles,
                "mean_jerk": float(np.mean(jerks) if jerks else 0.0),
            })

    return eval_records


def run_phase3():
    """Executes the complete Phase 3 experimental pipeline."""
    print("=" * 80)
    print("PHASE 3: FULL MULTI-SEED CBF-RL EXPERIMENT & VICOLUNGO BENCHMARK")
    print("4 Ablations | 3 Seeds (42, 100, 2024) | 150,000 steps/run | 60 Vicolungo Trajectories")
    print("=" * 80)

    cfg = Config()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 100, 2024]
    variant_specs = [
        ("Nominal", False, False),
        ("Filter_Only", True, False),
        ("Reward_Only", False, True),
        ("Dual_CBF_RL", True, True),
    ]

    # Pre-fetch valid Vicolungo trajectory IDs
    dummy_env = CarFollowingEnv(config=cfg, mode="vicolungo")
    valid_ids = dummy_env.get_valid_trajectory_ids()
    print(f"\nLoaded Vicolungo benchmark with {len(valid_ids)} causally valid trajectories.")

    all_train_dfs: List[pd.DataFrame] = []
    all_eval_records: List[Dict[str, Any]] = []

    total_experiment_start = time.time()

    # 1. Training Loop across 4 Variants x 3 Seeds
    print("\n" + "#" * 80)
    print("STAGE 1: PPO POLICY TRAINING (100% Synthetic Domain)")
    print("#" * 80)

    for var_idx, (name, use_filt, use_rew) in enumerate(variant_specs, 1):
        print(f"\n--- Variant {var_idx}/4: {name} (Filter={use_filt}, CBF_Reward={use_rew}) ---")
        for seed_idx, seed in enumerate(seeds, 1):
            print(f"\n  Starting Run [Seed {seed}] ({seed_idx}/3 for {name})...")
            agent, df_train = train_single_agent(
                variant_name=name,
                use_filter=use_filt,
                use_reward=use_rew,
                seed=seed,
                total_steps=150000,
                rollout_size=500,
                config=cfg,
            )
            all_train_dfs.append(df_train)

            # Evaluate this checkpoint on Vicolungo immediately
            print(f"  Evaluating {name} (Seed {seed}) on all 60 Vicolungo trajectories (Filter ON & OFF)...")
            eval_recs = evaluate_agent_on_vicolungo(
                agent=agent,
                variant_name=name,
                seed=seed,
                valid_trajectory_ids=valid_ids,
                config=cfg,
            )
            all_eval_records.extend(eval_recs)

    # 2. Save Consolidated Datasets
    print("\n" + "#" * 80)
    print("SAVING CONSOLIDATED RESULTS")
    print("#" * 80)

    combined_train_df = pd.concat(all_train_dfs, ignore_index=True)
    train_log_path = cfg.results_dir / "phase3_training_log.csv"
    combined_train_df.to_csv(train_log_path, index=False)
    print(f"Saved full training logs: {train_log_path} ({len(combined_train_df)} episode rows)")

    combined_eval_df = pd.DataFrame(all_eval_records)
    eval_log_path = cfg.results_dir / "phase3_vicolungo_eval.csv"
    combined_eval_df.to_csv(eval_log_path, index=False)
    print(f"Saved full evaluation logs: {eval_log_path} ({len(combined_eval_df)} rollout rows)")

    total_time = time.time() - total_experiment_start
    print(f"\nALL 12 RUNS & 1,440 EVALUATIONS COMPLETED IN {total_time / 60:.2f} MINUTES!")
    print("=" * 80)


if __name__ == "__main__":
    run_phase3()
