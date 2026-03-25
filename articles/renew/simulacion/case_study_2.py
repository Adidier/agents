"""
Case Study 2 — HESS + MGA + 5 xMG Energy Trading
====================================================
Replicates Section 5.2 of the paper.

Scenario:
  - All Case Study 1 components (3 ESS agents)
  - MGA agent (idx 3): decides sell volume + reserve price for next hour
  - 5 xMG agents (idx 4–8): bid volume + bid price for energy from MGA
  - 200 episodes × 168 steps = 33,600 total steps
  - Mixed cooperative-competitive MAS:
      ESS agents cooperate to minimise primary microgrid costs
      MGA competes with xMGs to maximise revenue
      xMGs compete with each other to buy cheapest energy

Agent index layout (case_study=2):
  0 = LIB, 1 = VRB, 2 = SC  → ESS charge/discharge  (1 action each)
  3 = MGA                    → sell volume + reserve price (2 actions)
  4–8 = xMG 1–5             → bid volume + bid price     (2 actions each)
  Total: 3 + 2 + 5×2 = 15 actions

Algorithms compared (only DDPG-family; paper excludes MADQN/RBM for CS2):
  Single-agent: DDPG, D3PG, TD3
  Multi-agent:  MADDPG, MAD3PG, MATD3

Usage
-----
    from articles.renew.simulacion.case_study_2 import run_case_study_2
    results = run_case_study_2("maddpg", total_episodes=200)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch

import sim_config  # noqa: F401 — path bootstrap

from maddpg_microgrid.environment import HESSMicrogridEnv, SyntheticDataGenerator
from maddpg_microgrid.config import (
    DDPGConfig, D3PGConfig, TD3Config, SimConfig,
    N_XMG,
)
from maddpg_microgrid.ddpg import DDPGAgent
from maddpg_microgrid.d3pg import D3PGAgent
from maddpg_microgrid.td3 import TD3Agent
from maddpg_microgrid.maddpg import MADDPGSystem

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Case study 2 action breakdown
_N_ESS    = 3          # LIB, VRB, SC  — 1 action each
_N_MGA    = 1          # 1 MGA agent   — 2 actions
_N_XMG    = N_XMG      # 5 xMG agents  — 2 actions each
_SGC_ACTIONS = _N_ESS + 2            # 5 total actions for single global controller
_MAS_ACTION_DIMS = [1]*3 + [2] + [2]*N_XMG   # per-agent action dims


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _env(seed: int, total_episodes: int) -> HESSMicrogridEnv:
    eval_start   = max(1, total_episodes // 2)
    total_steps  = total_episodes * 168
    rand_steps   = min(1000, max(64, total_steps // 10))
    learn_steps  = min(500,  max(32, rand_steps // 2))
    sim_cfg = SimConfig(
        total_episodes=total_episodes,
        eval_start_episode=eval_start,
        random_action_steps=rand_steps,
        learning_start_steps=learn_steps,
    )
    data_gen = SyntheticDataGenerator(seed=seed)
    return HESSMicrogridEnv(case_study=2, sim_cfg=sim_cfg, data_generator=data_gen, seed=seed)


def _make_sgc(algo: str, obs_dim: int) -> Any:
    """Single global controller: outputs 5 actions (3 ESS + MGA sell vol + reserve price)."""
    cfg_map = {"ddpg": DDPGConfig(), "d3pg": D3PGConfig(), "td3": TD3Config()}
    if algo not in cfg_map:
        raise ValueError(f"Unknown SGC algo for CS2: {algo!r}")
    cls_map = {"ddpg": DDPGAgent, "d3pg": D3PGAgent, "td3": TD3Agent}
    return cls_map[algo](obs_dim, _SGC_ACTIONS, cfg_map[algo], device=DEVICE)


def _make_mas(algo: str, env: HESSMicrogridEnv) -> MADDPGSystem:
    obs_dim_ess = env.obs_size_ess()
    obs_dim_xmg = env.obs_size_xmg()
    obs_dims = [obs_dim_ess] * (_N_ESS + _N_MGA) + [obs_dim_xmg] * _N_XMG
    base = algo.replace("ma", "").replace("matd3", "td3").replace("mad3pg", "d3pg").replace("maddpg", "ddpg")
    cfg = TD3Config() if "td3" in algo else D3PGConfig() if "d3pg" in algo else DDPGConfig()
    return MADDPGSystem(obs_dims, _MAS_ACTION_DIMS, algorithm=base, cfg=cfg, device=DEVICE)


def _savings_from_r_sum(r_sum: float, n_agents: int) -> float:
    return r_sum / (0.01 * n_agents)


def _compile(
    algo: str,
    eval_savings: List[float],
    ess_losses: List[float],
    mga_revenues: List[float],
) -> Dict[str, Any]:
    total_raw    = sum(eval_savings)
    total_ess    = sum(ess_losses)
    total_mga    = sum(mga_revenues)
    return {
        "algo": algo,
        "episode_savings": eval_savings,
        "total_savings_gbp": total_raw,
        "total_ess_loss_gbp": total_ess,
        "adjusted_savings_gbp": total_raw - total_ess,
        "ess_loss_pct": (total_ess / total_raw * 100) if total_raw > 0 else 0.0,
        "total_mga_revenue_gbp": total_mga,
    }


# -------------------------------------------------------------------------
# SGC training loop
# -------------------------------------------------------------------------

def _run_sgc(
    algo: str,
    env: HESSMicrogridEnv,
    sim_cfg: SimConfig,
    verbose: bool,
) -> Dict[str, Any]:
    obs_dim = env.obs_size_ess()
    agent = _make_sgc(algo, obs_dim)

    global_step = 0
    episode_savings: List[float] = []
    eval_savings:    List[float] = []
    ess_losses:      List[float] = []
    mga_revenues:    List[float] = []

    for episode in range(sim_cfg.total_episodes):
        obs_list = env.reset()
        obs = obs_list[0]   # SGC uses single ESS observation
        total_r_sum   = 0.0
        total_ess_cost = 0.0
        total_mga_rev  = 0.0

        for _ in range(sim_cfg.steps_per_episode):
            global_step += 1

            # SGC outputs 5 actions (ESS×3 + MGA×2); xMG actions are random
            if global_step < sim_cfg.random_action_steps:
                sgc_action = np.random.uniform(-1, 1, size=_SGC_ACTIONS)
            else:
                sgc_action = agent.select_action(obs)

            xmg_actions = list(np.random.uniform(-1, 1, size=2 * _N_XMG))
            full_actions = list(sgc_action) + xmg_actions

            next_obs_list, rewards, done, info = env.step(full_actions)
            next_obs = next_obs_list[0]
            reward   = sum(rewards[:_N_ESS + 1])   # ESS + MGA reward for SGC

            agent.push(obs, sgc_action, reward, next_obs, float(done))
            if global_step >= sim_cfg.learning_start_steps:
                agent.update()

            obs = next_obs
            total_r_sum    += info["R_sum"]
            total_ess_cost += info["R_CPC"] + info["R_SDC"]
            total_mga_rev  += info["R_MGA"]
            if done:
                break

        savings_raw = _savings_from_r_sum(total_r_sum, env.n_agents)
        episode_savings.append(savings_raw)

        if episode >= sim_cfg.eval_start_episode:
            eval_savings.append(savings_raw)
            ess_losses.append(total_ess_cost)
            mga_revenues.append(total_mga_rev)

        if verbose and episode % 20 == 0:
            m10 = np.mean(episode_savings[-10:])
            print(f"  [{algo.upper()} SGC ep {episode:3d}] "
                  f"savings=£{savings_raw:,.0f}  MGA=£{total_mga_rev:,.0f}  mean10=£{m10:,.0f}/week")

    return _compile(algo, eval_savings, ess_losses, mga_revenues)


# -------------------------------------------------------------------------
# MAS training loop
# -------------------------------------------------------------------------

def _run_mas(
    algo: str,
    env: HESSMicrogridEnv,
    sim_cfg: SimConfig,
    verbose: bool,
) -> Dict[str, Any]:
    system = _make_mas(algo, env)

    global_step = 0
    episode_savings: List[float] = []
    eval_savings:    List[float] = []
    ess_losses:      List[float] = []
    mga_revenues:    List[float] = []

    for episode in range(sim_cfg.total_episodes):
        obs_list = env.reset()
        total_r_sum   = 0.0
        total_ess_cost = 0.0
        total_mga_rev  = 0.0

        for _ in range(sim_cfg.steps_per_episode):
            global_step += 1
            if global_step < sim_cfg.random_action_steps:
                actions = [np.random.uniform(-1, 1, size=d) for d in _MAS_ACTION_DIMS]
            else:
                actions = system.select_actions(obs_list)

            # Flatten to list of floats for environment
            flat: List[float] = []
            for a in actions:
                if np.ndim(a) == 0:
                    flat.append(float(a))
                else:
                    flat.extend(a.tolist() if hasattr(a, "tolist") else list(a))

            next_obs_list, rewards, done, info = env.step(flat)

            system.push(obs_list, actions, rewards, next_obs_list, done)
            if global_step >= sim_cfg.learning_start_steps:
                system.update()

            obs_list = next_obs_list
            total_r_sum    += info["R_sum"]
            total_ess_cost += info["R_CPC"] + info["R_SDC"]
            total_mga_rev  += info["R_MGA"]
            if done:
                break

        savings_raw = _savings_from_r_sum(total_r_sum, env.n_agents)
        episode_savings.append(savings_raw)

        if episode >= sim_cfg.eval_start_episode:
            eval_savings.append(savings_raw)
            ess_losses.append(total_ess_cost)
            mga_revenues.append(total_mga_rev)

        if verbose and episode % 20 == 0:
            m10 = np.mean(episode_savings[-10:])
            print(f"  [{algo.upper()} ep {episode:3d}] "
                  f"savings=£{savings_raw:,.0f}  MGA=£{total_mga_rev:,.0f}  mean10=£{m10:,.0f}/week")

    return _compile(algo, eval_savings, ess_losses, mga_revenues)


# -------------------------------------------------------------------------
# Public entry point
# -------------------------------------------------------------------------

def run_case_study_2(
    algo: str = "maddpg",
    total_episodes: int = 200,
    seed: int = 42,
    verbose: bool = True,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Run Case Study 2 for one algorithm.

    Parameters
    ----------
    algo           : ddpg | d3pg | td3 | maddpg | mad3pg | matd3
    total_episodes : 200 (paper default)
    seed           : random seed
    verbose        : print per-episode progress
    save           : save results JSON to results/
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    algo = algo.lower()
    env = _env(seed, total_episodes)
    sim_cfg = env.sim_cfg   # use scaled sim_cfg from env

    print(f"\n{'='*60}")
    print(f"Case Study 2 — HESS + MGA + 5 xMG Energy Trading")
    print(f"Algorithm  : {algo.upper()}")
    print(f"Agents     : {env.n_agents}  (3 ESS + 1 MGA + 5 xMG)")
    print(f"Episodes   : {total_episodes}  (eval from ep {sim_cfg.eval_start_episode})")
    print(f"Device     : {DEVICE}")
    print(f"{'='*60}")

    t0 = time.time()
    cs2_algos = ["ddpg", "d3pg", "td3", "maddpg", "mad3pg", "matd3"]
    if algo not in cs2_algos:
        raise ValueError(f"Case Study 2 only supports: {cs2_algos}. Got: {algo!r}")

    if algo in ("ddpg", "d3pg", "td3"):
        results = _run_sgc(algo, env, sim_cfg, verbose)
    else:
        results = _run_mas(algo, env, sim_cfg, verbose)

    elapsed = time.time() - t0
    results["elapsed_s"] = elapsed
    results["seed"] = seed

    print(f"\n{'─'*60}")
    print(f"  Total savings (eval):   £{results['total_savings_gbp']:>10,.0f}")
    print(f"  ESS loss:               £{results['total_ess_loss_gbp']:>10,.0f}  "
          f"({results['ess_loss_pct']:.1f} %)")
    print(f"  Adjusted savings:       £{results['adjusted_savings_gbp']:>10,.0f}")
    print(f"  MGA revenue:            £{results['total_mga_revenue_gbp']:>10,.0f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'─'*60}\n")

    if save:
        path = os.path.join(sim_config.RESULTS_DIR, f"cs2_{algo}_ep{total_episodes}.json")
        with open(path, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"  Results saved → {path}")

    return results


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Case Study 2: HESS + MGA + xMG trading")
    parser.add_argument("--algo",     type=str, default="maddpg",
                        choices=["ddpg","d3pg","td3","maddpg","mad3pg","matd3"])
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--quiet",    action="store_true")
    parser.add_argument("--no-save",  action="store_true")
    args = parser.parse_args()

    run_case_study_2(
        algo=args.algo,
        total_episodes=args.episodes,
        seed=args.seed,
        verbose=not args.quiet,
        save=not args.no_save,
    )
