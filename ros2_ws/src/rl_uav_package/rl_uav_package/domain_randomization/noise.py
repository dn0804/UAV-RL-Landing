import numpy as np
from rl_uav_package.config import constants


def add_noise(obs):
    noise = np.random.normal(
        0,
        constants.OBS_NOISE_STD,
        size=len(obs)
    )

    return obs + noise