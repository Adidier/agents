"""
HESS Microgrid Environment
===========================
Implements the microgrid described in the paper, including:

  - Three ESS types: LIB, VRB, SC  (Section 2.1.1, Eq. 28–29)
  - PV and WT renewable generation  (Section 4.1.2, Eq. 30)
  - Wholesale energy market with asymmetric pricing  (Section 4.1.4)
  - MGA energy auction with 5 xMGs  (Section 4.2, Algorithm 2)
  - State / action / reward formulation  (Section 3.4)

All equation references point to the paper:
  "MADDPG for mixed cooperative-competitive MAS microgrid scenario"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import math

from .config import (
    ESS_CONFIGS, ESSConfig,
    P_MIN, P_MAX,
    WIND_CONFIG, PV_MAX_MW,
    N_XMG, XMG_DEMAND_SCALE, XMG_NOISE_STD, X_MAX_XMG, X_MAX_MGA,
    ETA_INV, ETA_MW_TRANSFORMER, ETA_GRID_TRANSFORMER,
    SimConfig,
)


# ---------------------------------------------------------------------------
# Synthetic data generation (placeholder for real Keele University data)
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Generates realistic synthetic time-series for demand, price, PV, WT, wind.
    Replace with real data by overriding `get_step`.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def get_step(self, t: int) -> Dict[str, float]:
        """Return environment data for absolute hour t."""
        hour_of_day = t % 24
        day_of_week = (t // 24) % 7

        # Demand (MW): campus-like profile
        base_demand = 3.0 + 2.0 * math.sin(math.pi * (hour_of_day - 6) / 12)
        demand = max(0.5, base_demand + self.rng.normal(0, 0.2))

        # Wholesale price (£/MWh): peaks in morning and evening
        price_base = 60 + 40 * math.sin(math.pi * (hour_of_day - 6) / 12) + \
                     20 * math.sin(math.pi * (hour_of_day - 17) / 6)
        price = float(np.clip(price_base + self.rng.normal(0, 8), P_MIN, P_MAX))

        # Solar irradiance (W/m²): bell curve around noon
        if 6 <= hour_of_day <= 20:
            irr = 1000 * math.sin(math.pi * (hour_of_day - 6) / 14) ** 2
        else:
            irr = 0.0

        # Wind speed (m/s)
        wind = float(np.clip(self.rng.normal(8, 3), 0, 30))

        return {
            "demand": demand,
            "grid_price": price,
            "irradiance": irr,
            "wind_speed": wind,
        }


# ---------------------------------------------------------------------------
# ESS dynamics
# ---------------------------------------------------------------------------

class ESSUnit:
    """
    Models one ESS with charge update (Eq. 28) and CPC cost (Eq. 29).
    """

    def __init__(self, cfg: ESSConfig):
        self.cfg = cfg
        self.charge: float = cfg.C_max * 0.5   # start at 50 % SoC

    def reset(self):
        self.charge = self.cfg.C_max * 0.5

    def step(self, action_mw: float) -> Dict[str, float]:
        """
        Apply one-step charge/discharge.

        Parameters
        ----------
        action_mw : float
            Positive = charge, negative = discharge (MW).
            Clipped to [-X_max, +X_max].

        Returns
        -------
        dict with 'charge', 'R_CPC', 'R_SDC', 'R_cap', 'dc_energy'
        """
        cfg = self.cfg
        x = float(np.clip(action_mw, -cfg.X_max, cfg.X_max))

        # Theoretical charge after action (Eq. 28)
        c_dot = x * math.sqrt(cfg.eta_RTE) + self.charge * cfg.eta_SDC

        # Capacity penalty (Eq. 24)
        if c_dot < 0:
            R_cap = cfg.X_max * c_dot ** 2 / cfg.X_max  # simplification: P_max = X_max
        elif c_dot > cfg.C_max:
            R_cap = cfg.X_max * (c_dot - cfg.C_max) ** 2 / cfg.X_max
        else:
            R_cap = 0.0

        # Clamp to physical limits
        c_new = float(np.clip(c_dot, 0.0, cfg.C_max))

        # CPC cost (Eq. 29) — half-cycle per step
        R_CPC = 0.5 * cfg.P_CPC * ((c_new - self.charge) / cfg.C_max) ** 2

        # Self-discharge penalty (Eq. 25)
        R_SDC = cfg.X_max * (self.charge / cfg.C_max) * (1.0 - cfg.eta_SDC)

        # DC energy contribution (positive = discharging into DC bus, negative = charging)
        dc_energy = -x * cfg.eta_RTE  # discharge provides energy to bus

        old_charge = self.charge
        self.charge = c_new

        return {
            "charge": c_new,
            "prev_charge": old_charge,
            "R_CPC": R_CPC,
            "R_SDC": R_SDC,
            "R_cap": R_cap,
            "dc_energy": dc_energy,   # net MW from ESS into DC bus
        }


# ---------------------------------------------------------------------------
# Wind turbine (Eq. 30)
# ---------------------------------------------------------------------------

def wind_power_mw(wind_speed: float) -> float:
    """Power output of one turbine in MW (Eq. 30)."""
    cfg = WIND_CONFIG
    v = wind_speed
    if v < cfg.v_ci or v > cfg.v_co:
        return 0.0
    if v >= cfg.v_r:
        return cfg.X_rated
    power_w = 0.5 * cfg.rho * math.pi * cfg.r ** 2 * cfg.cp * v ** 3
    return min(power_w / 1e6, cfg.X_rated)     # convert W → MW, cap at rated


# ---------------------------------------------------------------------------
# MGA auction  (Algorithm 2)
# ---------------------------------------------------------------------------

def mga_auction(
    X_MGA: float,
    P_MGA_reserve: float,
    xmg_volumes: np.ndarray,  # shape (N_XMG,)
    xmg_prices:  np.ndarray,  # shape (N_XMG,)
) -> Tuple[float, np.ndarray]:
    """
    MGA bidding step (Algorithm 2 in paper).

    Returns
    -------
    R_MGA : float   — reward for the MGA agent
    allocated : np.ndarray shape (N_XMG,) — energy allocated to each xMG
    """
    remaining = float(X_MGA)
    prices = xmg_prices.copy().astype(float)
    volumes = xmg_volumes.copy().astype(float)
    allocated = np.zeros(N_XMG)
    R_MGA = 0.0

    while remaining > 1e-6 and np.any(volumes > 1e-6):
        i = int(np.argmax(prices))
        if prices[i] <= P_MGA_reserve or volumes[i] <= 1e-6:
            break
        x_bid = min(remaining, volumes[i])
        R_MGA += 0.8 * prices[i] * x_bid          # MGA earns 80% of bid price
        allocated[i] += x_bid
        remaining -= x_bid
        volumes[i] = 0.0
        prices[i] = 0.0

    # Unsold energy returned to grid at P_MIN
    R_MGA += P_MIN * remaining

    return R_MGA, allocated


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class HESSMicrogridEnv:
    """
    Microgrid environment implementing:
      - Case study 1: ESS control only
      - Case study 2: ESS + MGA trading with 5 xMGs

    Agent indices (case study 2):
      0 = LIB, 1 = VRB, 2 = SC, 3 = MGA, 4–8 = xMG1–xMG5

    For case study 1, only indices 0–2 are used (ESS only).
    """

    AGENT_LIB = 0
    AGENT_VRB = 1
    AGENT_SC  = 2
    AGENT_MGA = 3
    AGENT_XMG_START = 4

    def __init__(
        self,
        case_study: int = 2,        # 1 = ESS only, 2 = ESS + xMG trading
        sim_cfg: Optional[SimConfig] = None,
        data_generator: Optional[SyntheticDataGenerator] = None,
        seed: int = 42,
    ):
        self.case_study = case_study
        self.sim_cfg = sim_cfg or SimConfig()
        self.data_gen = data_generator or SyntheticDataGenerator(seed=seed)

        # ESS units
        self.ess: List[ESSUnit] = [
            ESSUnit(ESS_CONFIGS["LIB"]),
            ESSUnit(ESS_CONFIGS["VRB"]),
            ESSUnit(ESS_CONFIGS["SC"]),
        ]

        # Determine agent count
        if case_study == 1:
            self.n_agents = 3
        else:
            self.n_agents = 3 + 1 + N_XMG   # ESS + MGA + xMGs

        self.t = 0                  # absolute step counter
        self.episode_step = 0       # step within current episode
        self._init_state()

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _init_state(self):
        """Pre-fetch data for current step."""
        self._data = self.data_gen.get_step(self.t)
        self._prev_data = self._data

        # xMG demands
        base_d = self._data["demand"]
        self._xmg_demand = np.array([
            max(0.0, XMG_DEMAND_SCALE * base_d + np.random.normal(0, XMG_NOISE_STD))
            for _ in range(N_XMG)
        ])

        # MGA last step sell volume and reserve price (observable by ESS/MGA)
        self._mga_volume = 0.0
        self._mga_price  = (P_MAX + P_MIN) / 2.0

    def _ess_obs(self) -> np.ndarray:
        """
        State for ESS / MGA agents (Eq. 14) — 15 features.
          [c_LIB, c_SC, c_VRB,
           P_grid, P_grid_pred,
           P_MGA, X_MGA,
           X_D, X_D_pred,
           X_PV, X_PV_pred,
           X_WT, X_WT_pred,
           H_day, H_week]
        All values normalised to [0, 1].
        """
        d = self._data
        d_next = self.data_gen.get_step(self.t + 1)

        pv = (d["irradiance"] / 1000.0) * PV_MAX_MW
        pv_next = (d_next["irradiance"] / 1000.0) * PV_MAX_MW
        wt = wind_power_mw(d["wind_speed"]) * WIND_CONFIG.n_turbines
        wt_next = wind_power_mw(d_next["wind_speed"]) * WIND_CONFIG.n_turbines

        raw = np.array([
            self.ess[0].charge / ESS_CONFIGS["LIB"].C_max,  # c_LIB
            self.ess[2].charge / ESS_CONFIGS["SC"].C_max,   # c_SC
            self.ess[1].charge / ESS_CONFIGS["VRB"].C_max,  # c_VRB
            (d["grid_price"] - P_MIN) / (P_MAX - P_MIN),    # P_grid
            (d_next["grid_price"] - P_MIN) / (P_MAX - P_MIN),
            (self._mga_price - P_MIN) / (P_MAX - P_MIN),    # P_MGA
            self._mga_volume / X_MAX_MGA,                   # X_MGA
            d["demand"] / 10.0,                             # X_D (normalise by 10 MW)
            d_next["demand"] / 10.0,
            pv / PV_MAX_MW,
            pv_next / PV_MAX_MW,
            wt / (WIND_CONFIG.X_rated * WIND_CONFIG.n_turbines),
            wt_next / (WIND_CONFIG.X_rated * WIND_CONFIG.n_turbines),
            (self.t % 24) / 23.0,                           # H_day
            (self.t % (24 * 7)) / (24 * 7 - 1),            # H_week
        ], dtype=np.float32)
        return np.clip(raw, 0.0, 1.0)

    def _xmg_obs(self, xmg_idx: int) -> np.ndarray:
        """
        State for xMG agents (Eq. 15) — 5 features.
          [P_MGA, X_MGA, X_xMG_D, H_day, H_week]
        """
        raw = np.array([
            (self._mga_price - P_MIN) / (P_MAX - P_MIN),
            self._mga_volume / X_MAX_MGA,
            self._xmg_demand[xmg_idx] / (XMG_DEMAND_SCALE * 10.0 * 2),
            (self.t % 24) / 23.0,
            (self.t % (24 * 7)) / (24 * 7 - 1),
        ], dtype=np.float32)
        return np.clip(raw, 0.0, 1.0)

    def obs_size_ess(self) -> int:
        return 15

    def obs_size_xmg(self) -> int:
        return 5

    # ------------------------------------------------------------------
    # Reset / Step
    # ------------------------------------------------------------------

    def reset(self) -> List[np.ndarray]:
        """Reset episode. Returns list of observations per agent."""
        for ess in self.ess:
            ess.reset()
        self.episode_step = 0
        self._init_state()

        obs = [self._ess_obs()] * 3   # LIB, VRB, SC see same ESS obs
        if self.case_study == 2:
            obs.append(self._ess_obs())   # MGA also sees ESS obs
            for i in range(N_XMG):
                obs.append(self._xmg_obs(i))
        return obs

    def step(self, actions: List[float]) -> Tuple[List[np.ndarray], List[float], bool, Dict]:
        """
        Execute one time step.

        Parameters
        ----------
        actions : list of floats, one per agent, each in [-1, 1]
            [a_LIB, a_VRB, a_SC]  for case_study=1
            [a_LIB, a_VRB, a_SC, a_MGA_X, a_MGA_P, a_xMG1_X, a_xMG1_P, ...] for case_study=2

        Returns
        -------
        obs_list, rewards, done, info
        """
        d = self._data
        pv_mw = (d["irradiance"] / 1000.0) * PV_MAX_MW
        wt_mw = wind_power_mw(d["wind_speed"]) * WIND_CONFIG.n_turbines
        P_grid = d["grid_price"]

        # ---- ESS actions (Eq. 16) ----
        ess_results = []
        X_dc = 0.0   # net DC bus energy from all ESSs (positive = discharged into bus)
        for idx, ess in enumerate(self.ess):
            a_raw = float(actions[idx])                        # in [-1, 1]
            x_mw = ess.cfg.X_max * float(np.clip(a_raw, -1, 1))  # MW
            result = ess.step(x_mw)
            ess_results.append(result)
            X_dc += result["dc_energy"]

        # Subtract PV from DC bus (PV on DC side)
        X_dc -= pv_mw

        # ---- Grid energy (Eq. 21–22) ----
        # DC line (Eq. 21): using RTE already applied in step(); here X_dc is net
        # X_in = (demand + X_dc * eta_inv - wt * eta_MW_tra) * eta_grid_tra
        X_in = (d["demand"] + X_dc * ETA_INV - wt_mw * ETA_MW_TRANSFORMER) * ETA_GRID_TRANSFORMER

        # Baseline: all ESS idle, no MGA (Eq. 26 baseline)
        X_in_base = (d["demand"] - pv_mw * ETA_INV - wt_mw * ETA_MW_TRANSFORMER) * ETA_GRID_TRANSFORMER

        # ---- Grid cost reward (Eq. 23) ----
        R_in = -X_in * P_grid
        R_base = -X_in_base * P_grid

        # ---- ESS penalty rewards ----
        R_CPC_total = sum(r["R_CPC"] for r in ess_results)
        R_SDC_total = sum(r["R_SDC"] for r in ess_results)
        R_cap_total = sum(r["R_cap"] for r in ess_results)

        # ---- MGA auction (Case study 2) ----
        R_MGA = 0.0
        xmg_allocated = np.zeros(N_XMG)

        if self.case_study == 2:
            # MGA actions (Eq. 17–18)
            a_mga_x = float(actions[3])
            a_mga_p = float(actions[4])
            self._mga_volume = 0.5 * X_MAX_MGA * a_mga_x + 0.5 * X_MAX_MGA
            self._mga_price  = 0.5 * (P_MAX - P_MIN) * a_mga_p + 0.5 * (P_MAX + P_MIN)

            xmg_volumes = np.zeros(N_XMG)
            xmg_prices  = np.zeros(N_XMG)
            for i in range(N_XMG):
                a_x = float(actions[5 + 2 * i])
                a_p = float(actions[5 + 2 * i + 1])
                # xMG actions (Eq. 19–20)
                xmg_volumes[i] = 0.5 * X_MAX_XMG * a_x + 0.5 * X_MAX_XMG
                xmg_prices[i]  = 0.5 * (P_MAX - P_MIN) * a_p + 0.5 * (P_MAX + P_MIN)

            R_MGA, xmg_allocated = mga_auction(
                self._mga_volume, self._mga_price, xmg_volumes, xmg_prices
            )

        # ---- Total reward (Eq. 26) ----
        R_sum = R_in + R_MGA - R_CPC_total - R_SDC_total - R_cap_total - R_base

        # ---- Per-agent marginal-contribution rewards ----
        # Each ESS is rewarded on global savings if that ESS had remained idle,
        # rather than if all ESSs were idle. (marginal contribution, Section 3.4.3)
        agent_rewards = []
        for idx, ess in enumerate(self.ess):
            result_i = ess_results[idx]
            # Counterfactual: this ESS at zero (no contribution)
            dc_without_i = X_dc - result_i["dc_energy"]
            x_in_without_i = (d["demand"] + dc_without_i * ETA_INV - wt_mw * ETA_MW_TRANSFORMER) * ETA_GRID_TRANSFORMER
            r_in_without_i = -x_in_without_i * P_grid
            r_marginal = R_in - r_in_without_i - result_i["R_CPC"] - result_i["R_SDC"] - result_i["R_cap"]
            agent_rewards.append(r_marginal)

        # MGA and xMG rewards (case study 2)
        if self.case_study == 2:
            agent_rewards.append(R_MGA)
            for i in range(N_XMG):
                # xMG reward: savings vs buying everything from grid at P_MAX
                got = xmg_allocated[i]
                shortfall = max(0.0, self._xmg_demand[i] - got)
                xmg_cost = got * self._mga_price + shortfall * P_MAX
                xmg_baseline = self._xmg_demand[i] * P_MAX
                agent_rewards.append(xmg_baseline - xmg_cost)

        # ---- Normalise reward (Eq. 27) ----
        scale = 0.01 * self.n_agents
        rewards_norm = [r * scale for r in agent_rewards]

        # ---- Advance time ----
        self.t += 1
        self.episode_step += 1
        self._prev_data = self._data
        self._data = self.data_gen.get_step(self.t)
        base_d = self._data["demand"]
        self._xmg_demand = np.array([
            max(0.0, XMG_DEMAND_SCALE * base_d + np.random.normal(0, XMG_NOISE_STD))
            for _ in range(N_XMG)
        ])

        done = self.episode_step >= self.sim_cfg.steps_per_episode

        # ---- Next observations ----
        next_obs = [self._ess_obs()] * 3
        if self.case_study == 2:
            next_obs.append(self._ess_obs())
            for i in range(N_XMG):
                next_obs.append(self._xmg_obs(i))

        info = {
            "R_in": R_in,
            "R_MGA": R_MGA,
            "R_CPC": R_CPC_total,
            "R_SDC": R_SDC_total,
            "R_cap": R_cap_total,
            "R_base": R_base,
            "R_sum": R_sum,
            "X_in": X_in,
            "grid_price": P_grid,
        }

        return next_obs, rewards_norm, done, info
