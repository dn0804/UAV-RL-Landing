"""
filename: sensor_noise.py
description: Placeholder sensor noise model for domain randomization. Adds small Gaussian noise to observations.
"""


import numpy as np

def add_noise(obs, std=0.01):
    return obs + np.random.normal(0, std, size=obs.shape)