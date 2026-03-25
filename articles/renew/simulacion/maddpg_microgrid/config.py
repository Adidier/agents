"""
Configuration — MADDPG Microgrid (Paper-exact parameters)
==========================================================
All constants taken directly from the article:
  - Table 1: ESS properties (LIB, VRB, SC)
  - Table 2: DDPG hyperparameters
  - Section 4.1: environment parameters
"""

from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# ESS parameters  (Table 1)
# ---------------------------------------------------------------------------
@dataclass
class ESSConfig:
    name: str
    C_max: float        # MWh — maximum capacity
    X_max: float        # MW  — maximum charge/discharge power per step
    eta_SDC: float      # self-discharge efficiency per hour  (fraction kept)
    eta_RTE: float      # round-trip efficiency  (fraction, not percent)
    capacity_cost: float  # £/kWh
    lifecycles: int
    P_CPC: float        # £ per cycle (capacity_cost * C_max*1000 / lifecycles / 2)


LIB_CONFIG = ESSConfig(
    name="LIB", C_max=2.0, X_max=1.0,
    eta_SDC=0.9999, eta_RTE=0.95,
    capacity_cost=100.0, lifecycles=5_000,
    P_CPC=40.0,   # 100 * 2000 kWh / 5000 / 2  ≈ £40 per cycle-half
)

VRB_CONFIG = ESSConfig(
    name="VRB", C_max=2.0, X_max=1.0,
    eta_SDC=1.0000, eta_RTE=0.80,
    capacity_cost=200.0, lifecycles=10_000,
    P_CPC=40.0,
)

SC_CONFIG = ESSConfig(
    name="SC", C_max=2.0, X_max=1.0,
    eta_SDC=0.9900, eta_RTE=0.95,
    capacity_cost=300.0, lifecycles=100_000,
    P_CPC=6.0,
)

ESS_CONFIGS: Dict[str, ESSConfig] = {
    "LIB": LIB_CONFIG,
    "VRB": VRB_CONFIG,
    "SC":  SC_CONFIG,
}


# ---------------------------------------------------------------------------
# Market / price parameters  (Section 4.1.4)
# ---------------------------------------------------------------------------
P_MIN: float = 16.0    # £/MWh — fixed feed-in tariff (sell to grid)
P_MAX: float = 144.0   # £/MWh — cap on wholesale buy price


# ---------------------------------------------------------------------------
# Wind turbine parameters  (Section 4.1.2, Eq. 30)
# ---------------------------------------------------------------------------
@dataclass
class WindConfig:
    v_ci: float = 3.0    # m/s — cut-in speed
    v_r:  float = 12.0   # m/s — rated speed
    v_co: float = 25.0   # m/s — cut-out speed
    rho:  float = 1.225  # kg/m³ — air density
    r:    float = 30.0   # m — blade radius
    cp:   float = 0.40   # power coefficient
    X_rated: float = 1.0 # MW — rated power per turbine
    n_turbines: int = 2  # total turbines → 2 MW max


WIND_CONFIG = WindConfig()

# PV max capacity (Section 4.1.2)
PV_MAX_MW: float = 5.0


# ---------------------------------------------------------------------------
# xMG parameters  (Section 4.2.1)
# ---------------------------------------------------------------------------
N_XMG: int = 5                    # number of external microgrids
XMG_DEMAND_SCALE: float = 0.05    # each xMG demand = 5% of primary + noise
XMG_NOISE_STD: float = 0.01       # Gaussian noise std on xMG demand
X_MAX_XMG: float = 0.5            # MW — max xMG bid volume (assumed)

# MGA max sell volume per step
X_MAX_MGA: float = 2.0            # MW (assumption — must cover all xMGs)


# ---------------------------------------------------------------------------
# Inverter / transformer efficiencies  (Section 4.1.3)
# ---------------------------------------------------------------------------
ETA_INV: float = 0.95      # inverter efficiency (simplified flat model)
ETA_MW_TRANSFORMER: float = 0.98   # WT transformer efficiency
ETA_GRID_TRANSFORMER: float = 0.98  # primary microgrid ↔ grid transformer


# ---------------------------------------------------------------------------
# DDPG Hyperparameters  (Table 2)
# ---------------------------------------------------------------------------
@dataclass
class DDPGConfig:
    actor_lr: float  = 5e-4
    critic_lr: float = 1e-3
    hidden_dim: int  = 256
    batch_size: int  = 64
    tau: float       = 0.005    # soft target update
    gamma: float     = 0.99
    buffer_size: int = 100_000


@dataclass
class D3PGConfig(DDPGConfig):
    V_min: float = -10.0
    V_max: float =  10.0
    n_atoms: int = 51


@dataclass
class TD3Config(DDPGConfig):
    actor_delay: int = 2        # update actor every N critic updates
    policy_noise: float = 0.2   # noise added to target actions
    noise_clip: float = 0.5     # clamp for target noise


# ---------------------------------------------------------------------------
# Simulation setup  (Section 4.3)
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    steps_per_episode: int = 168    # 1 week (168 hours)
    total_episodes: int    = 200
    eval_start_episode: int = 100   # evaluation begins at episode 100
    random_action_steps: int = 1000 # purely random actions for first N steps
    learning_start_steps: int = 500 # begin training after N steps in buffer
    reward_scale: float = 0.01      # r_t = reward_scale * N_agents * R_sum
