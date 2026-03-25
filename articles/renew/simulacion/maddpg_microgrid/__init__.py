"""
maddpg_microgrid
================
Python implementation of the paper:
  "MADDPG for mixed cooperative-competitive MAS microgrid scenario
   for RES integration and energy arbitrage"

Public API
----------
    from src.maddpg_microgrid import HESSMicrogridEnv
    from src.maddpg_microgrid import DDPGAgent, D3PGAgent, TD3Agent
    from src.maddpg_microgrid import MADDPGSystem
    from src.maddpg_microgrid.train import run
"""

from .environment   import HESSMicrogridEnv, SyntheticDataGenerator
from .ddpg          import DDPGAgent
from .d3pg          import D3PGAgent
from .td3           import TD3Agent
from .maddpg        import MADDPGSystem
from .replay_buffer import ReplayBuffer
from .config        import (
    DDPGConfig, D3PGConfig, TD3Config, SimConfig,
    LIB_CONFIG, VRB_CONFIG, SC_CONFIG,
    P_MIN, P_MAX,
)

__all__ = [
    "HESSMicrogridEnv",
    "SyntheticDataGenerator",
    "DDPGAgent",
    "D3PGAgent",
    "TD3Agent",
    "MADDPGSystem",
    "ReplayBuffer",
    "DDPGConfig",
    "D3PGConfig",
    "TD3Config",
    "SimConfig",
    "LIB_CONFIG",
    "VRB_CONFIG",
    "SC_CONFIG",
    "P_MIN",
    "P_MAX",
]
