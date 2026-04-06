"""
filename: wind.py
description: Placeholder wind model for domain randomization. Generates a random wind vector at the start of each episode and applies small random changes at each step.
"""

import numpy as np

class WindModel:
    def __init__(self, rng, strength=0.2):
        self.rng = rng
        self.strength = strength
        self.wind = np.zeros(3)

    def reset(self):
        self.wind = self.rng.uniform(-self.strength, self.strength, size=3)

    def step(self):
        self.wind += self.rng.normal(0, 0.01, size=3)
        return self.wind