"""
filename: battery.py
description: Placeholder battery model for domain randomization. Simulates a simple scaling effect on the action to represent battery degradation.
"""

class BatteryModel:
    def __init__(self, rng):
        self.rng = rng
        self.scale = 1.0

    def reset(self):
        self.scale = self.rng.uniform(0.85, 1.0)

    def apply(self, action):
        return action * self.scale