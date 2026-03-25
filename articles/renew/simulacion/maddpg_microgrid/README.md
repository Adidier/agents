# maddpg_microgrid

Python implementation of the microgrid MADDPG paper, faithfully following
all equations and parameters from the article.

## Structure

```
src/maddpg_microgrid/
├── __init__.py          # public API
├── config.py            # all paper constants (Table 1, Table 2, Eq. params)
├── environment.py       # HESSMicrogridEnv — ESS dynamics, auction, rewards
├── networks.py          # Actor, Critic, DistributionalCritic, MADDPGCritic
├── replay_buffer.py     # off-policy experience replay buffer
├── ddpg.py              # DDPGAgent  (Section 3.2.1)
├── d3pg.py              # D3PGAgent  (Section 3.2.2 — distributional)
├── td3.py               # TD3Agent   (Section 3.2.3 — twin delayed)
├── maddpg.py            # MADDPGSystem (Section 3.3 — multi-agent)
└── train.py             # full training loop (Section 4.3)
```

## Quick start

### Run from the command line

```bash
# Case study 2 (ESS + xMG trading) with MATD3 — best result in the paper
python -m src.maddpg_microgrid.train --algo matd3 --case 2 --episodes 200

# Available algorithms: ddpg, d3pg, td3, maddpg, mad3pg, matd3
# Case study 1 (HESS control only):
python -m src.maddpg_microgrid.train --algo maddpg --case 1 --episodes 200
```

### Run from Python

```python
from src.maddpg_microgrid.train import run

results = run(algo="matd3", case_study=2, total_episodes=200, seed=42)
print(f"Total eval savings: £{results['total_eval_savings_gbp']:,.0f}")
```

### Manual environment loop

```python
import numpy as np
from src.maddpg_microgrid import (
    HESSMicrogridEnv, MADDPGSystem, TD3Config, SimConfig
)

env = HESSMicrogridEnv(case_study=2)

obs_dims = [env.obs_size_ess()] * 3 + [env.obs_size_ess()] + [env.obs_size_xmg()] * 5
act_dims = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2]

system = MADDPGSystem(obs_dims, act_dims, algorithm="td3", cfg=TD3Config())

obs_list = env.reset()
for step in range(168):            # one episode = 1 week
    actions = system.select_actions(obs_list)

    # flatten to environment format
    flat = []
    for a in actions:
        flat.extend(a.tolist() if a.ndim > 0 else [float(a)])

    next_obs, rewards, done, info = env.step(flat)
    system.push(obs_list, actions, rewards, next_obs, done)
    system.update()
    obs_list = next_obs
    if done:
        break
```

## Paper → code mapping

| Paper section | File | Key symbol |
|---|---|---|
| Eq. 28 (ESS charge update) | `environment.py` `ESSUnit.step()` | `c_t = x_t√η_RTE + c_{t-1}·η_SDC` |
| Eq. 29 (CPC cost) | `environment.py` `ESSUnit.step()` | `R_CPC = 0.5·P_CPC·(Δc/C_max)²` |
| Eq. 30 (wind curve) | `environment.py` `wind_power_mw()` | `X_WT` |
| Eq. 21–23 (reward) | `environment.py` `HESSMicrogridEnv.step()` | `R_in, X_in, X_dc` |
| Eq. 27 (norm reward) | `environment.py` | `r_t = 0.01·N·R_sum` |
| Alg. 2 (MGA auction) | `environment.py` `mga_auction()` | `R_MGA` |
| Eq. y_i, L_i, ∇J_i | `ddpg.py` `DDPGAgent.update()` | DDPG update |
| KL distributional loss | `d3pg.py` `D3PGAgent.update()` | D3PG loss |
| min(Q̂₁, Q̂₂) | `td3.py` `TD3Agent.update()` | TD3 target |
| Centralised critic | `maddpg.py` `MADDPGCritic` | global (s,a) → Q |
| Marginal contribution | `environment.py` `HESSMicrogridEnv.step()` | per-agent reward |
| Table 1 (ESS params) | `config.py` `LIB_CONFIG` etc. | — |
| Table 2 (hyperparams) | `config.py` `DDPGConfig` | — |

## Notes

- **Synthetic data**: the environment uses `SyntheticDataGenerator` as a
  placeholder for the real Keele University campus data. Replace it by
  subclassing and overriding `get_step(t)`.
- **NoisyNet**: actors use NoisyLinear output layers instead of an
  external noise process (as stated in the paper).
- **Marginal-contribution reward**: each ESS agent is rewarded based on
  the counterfactual "what would costs be if only this ESS had remained idle",
  following game-theoretic marginal contribution (Section 3.4.3).
