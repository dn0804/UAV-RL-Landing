"""
filename: manager.py
description: Placeholder manager for handling domain randomization effects in the simulation.
"""

import numpy as np

from .wind import WindModel
from .sensor_noise import add_noise
from .battery import BatteryModel


class DomainRandomizationManager:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

        # Placeholder modules
        self.wind = WindModel(rng)
        self.battery = BatteryModel(rng)

        # Config placeholders
        self.enable_wind = True
        self.enable_noise = True
        self.enable_battery = True

        self.noise_std = 0.0  # start at 0 → no effect

    # ── Episode-level ─────────────────────────────
    def reset(self, env):
        """Called once per episode"""
        if self.enable_wind:
            self.wind.reset()
        if self.enable_battery:
            self.battery.reset()

    # ── Action-level ──────────────────────────────
    def apply_action_noise(self, action: np.ndarray):
        if not self.enable_battery:
            return action

        action = action.copy()

        # Placeholder - very small effect
        action[:3] = self.battery.apply(action[:3])

        return action

    # ── State-level ───────────────────────────────
    def apply_state_perturbation(self, state: dict):
        if not self.enable_wind:
            return state

        state = state.copy()

        wind = self.wind.step()

        # Small placeholder disturbance
        state["vx"] += wind[0]
        state["vy"] += wind[1]
        state["vz"] += wind[2]

        return state

    # ── Observation-level ─────────────────────────
    def apply_observation_noise(self, obs: np.ndarray):
        if not self.enable_noise or self.noise_std == 0.0:
            return obs

        return add_noise(obs, std=self.noise_std)