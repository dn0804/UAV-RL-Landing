"""
Exponential Moving Average (EMA) velocity filter.

Smooths noisy velocity readings before they enter the observation vector.
This exact filter runs during both training and deployment (filter-in-the-loop).
The agent learns to control the drone *through* the filter's lag, so the
smoothing parameter must be identical in both contexts.

v_filtered = α × v_raw + (1 − α) × v_previous

At α = 0.4 and 10 Hz, the effective time constant is ~0.25 s (settles in 2-3 steps).
"""

import numpy as np

from rl_uav_package.config.constants import EMA_ALPHA


class EMAFilter:
    """Per-axis exponential moving average filter.

    Parameters
    ----------
    n_channels : int
        Number of independent channels to filter (e.g. 3 for vx, vy, vz).
    alpha : float, optional
        Smoothing factor. Defaults to the project-wide constant.
    """

    def __init__(self, n_channels: int = 3, alpha: float = EMA_ALPHA):
        self.alpha = alpha
        self.n_channels = n_channels
        self._value = np.zeros(n_channels, dtype=np.float64)
        self._initialized = False

    def reset(self, initial_value: np.ndarray) -> None:
        """Re-seed the filter with a known ground-truth value.

        Must be called at every episode boundary with the drone's actual
        velocity at the new spawn position.  Never reset to zero — that
        introduces a startup transient where the filter ramps from zero
        to the true value over several steps, feeding the agent
        systematically wrong data at the start of every episode.
        """
        self._value = np.asarray(initial_value, dtype=np.float64).copy()
        self._initialized = True

    def update(self, raw: np.ndarray) -> np.ndarray:
        """Apply one EMA step and return the filtered value.

        Parameters
        ----------
        raw : array-like, shape (n_channels,)
            Unfiltered sensor reading for this timestep.

        Returns
        -------
        filtered : np.ndarray, shape (n_channels,), dtype float32
            Smoothed value suitable for the observation vector.
        """
        raw = np.asarray(raw, dtype=np.float64)

        if not self._initialized:
            # First call without an explicit reset — seed from this reading.
            self._value = raw.copy()
            self._initialized = True
        else:
            self._value = self.alpha * raw + (1.0 - self.alpha) * self._value

        return self._value.astype(np.float32)

    @property
    def value(self) -> np.ndarray:
        """Current filtered value (read-only)."""
        return self._value.astype(np.float32)