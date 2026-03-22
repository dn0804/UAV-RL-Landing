import numpy as np
from rl_uav_package.config import constants


def wind():
    return np.random.normal(
        0,
        constants.WIND_STD,
        size=3
    )