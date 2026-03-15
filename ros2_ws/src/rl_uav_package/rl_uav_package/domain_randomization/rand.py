from .noise import add_noise
from .wind import wind

class DomainRandomizer:

    def __init__(self, enabled=True):
        self.enabled = enabled

    def get_wind(self):
        if not self.enabled:
            return None
        return wind()

    def apply_noise(self, obs):
        if not self.enabled:
            return obs
        return add_noise(obs)