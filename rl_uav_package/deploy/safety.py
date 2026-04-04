"""
Safety systems for real-hardware deployment.

Three layers of protection:
    1. ActionLimiter  — clamps RC channel magnitudes before sending
    2. Geofence       — triggers auto-land if drone leaves a safe volume
    3. KillSwitch     — background thread listening for keyboard abort

All three are independent; the runner composes them in order.
"""

import math
import sys
import threading
from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Configuration
# ============================================================

@dataclass
class SafetyConfig:
    """All safety-related tunables in one place."""

    # ── Action limiter ───────────────────────────────────────
    # Maximum absolute value sent on any single RC channel.
    # Tello RC range is -100..100; 40 keeps things gentle.
    max_rc: int = 40

    # ── Geofence ─────────────────────────────────────────────
    # Horizontal radius (meters) from the marker center.
    # Anything outside this triggers auto-land.
    geofence_radius: float = 3.0

    # Altitude bounds (meters, from ToF sensor).
    max_altitude: float = 2.0
    min_altitude: float = 0.10  # below this → already on a surface

    # ── Timing ───────────────────────────────────────────────
    # Maximum episode duration in seconds.  Auto-lands after this
    # regardless of policy output.
    max_duration_s: float = 60.0

    # Maximum consecutive seconds without a marker detection
    # before triggering auto-land.  Prevents the drone from
    # wandering blind indefinitely.
    max_dropout_s: float = 5.0


# ============================================================
# Action Limiter
# ============================================================

def clamp_rc(lr: int, fb: int, ud: int, yaw: int,
             max_rc: int) -> tuple[int, int, int, int]:
    """Clamp each RC channel to [-max_rc, +max_rc].

    This is the innermost safety layer — it runs on every single
    command, even if the geofence and kill switch are both happy.

    Parameters
    ----------
    lr, fb, ud, yaw : int
        Raw RC channel values from action scaling.
    max_rc : int
        Maximum absolute value on any channel (e.g. 40 out of 100).

    Returns
    -------
    tuple[int, int, int, int]
        Clamped (lr, fb, ud, yaw).
    """
    def _c(v):
        return max(-max_rc, min(max_rc, int(v)))
    return _c(lr), _c(fb), _c(ud), _c(yaw)


# ============================================================
# Geofence
# ============================================================

# Reason codes returned by GeofenceMonitor.check()
SAFE = "safe"
OUTSIDE_RADIUS = "outside_geofence_radius"
TOO_HIGH = "above_max_altitude"
TOO_LOW = "below_min_altitude"
EPISODE_TIMEOUT = "episode_timeout"
DROPOUT_TIMEOUT = "marker_dropout_timeout"


class GeofenceMonitor:
    """Checks whether the drone is within the safe operating volume.

    The volume is a vertical cylinder centered on the marker with
    floor and ceiling planes.  Call ``check()`` every control tick;
    it returns a (safe, reason) tuple.

    Parameters
    ----------
    config : SafetyConfig
        Safety parameters (radius, altitudes, timeouts).
    marker_world_xy : tuple[float, float]
        World-frame (x, y) of the marker center.  The geofence
        cylinder is centered here.
    control_rate_hz : float
        Expected control loop frequency (used for dropout timing).
    """

    def __init__(self, config: SafetyConfig,
                 marker_world_xy: tuple[float, float],
                 control_rate_hz: float = 10.0):
        self._cfg = config
        self._mx, self._my = marker_world_xy
        self._hz = control_rate_hz

        # Timing state
        self._elapsed_s = 0.0
        self._dropout_s = 0.0

    def reset(self) -> None:
        """Reset timers for a new episode / attempt."""
        self._elapsed_s = 0.0
        self._dropout_s = 0.0

    def check(self, world_x: float, world_y: float,
              altitude: float, marker_visible: bool
              ) -> tuple[bool, str]:
        """Evaluate all geofence conditions.

        Parameters
        ----------
        world_x, world_y : float
            Estimated drone position in world frame (meters).
        altitude : float
            Height above ground from ToF sensor (meters).
        marker_visible : bool
            Whether the ArUco marker was detected this tick.

        Returns
        -------
        (is_safe, reason) : tuple[bool, str]
            ``is_safe`` is True when the drone may continue flying.
            ``reason`` is one of the module-level constants (SAFE,
            OUTSIDE_RADIUS, etc.).
        """
        dt = 1.0 / self._hz
        self._elapsed_s += dt

        # ── Dropout timer ────────────────────────────────────
        if marker_visible:
            self._dropout_s = 0.0
        else:
            self._dropout_s += dt

        if self._dropout_s >= self._cfg.max_dropout_s:
            return False, DROPOUT_TIMEOUT

        # ── Episode timeout ──────────────────────────────────
        if self._elapsed_s >= self._cfg.max_duration_s:
            return False, EPISODE_TIMEOUT

        # ── Spatial checks ───────────────────────────────────
        d_horiz = math.hypot(world_x - self._mx, world_y - self._my)
        if d_horiz > self._cfg.geofence_radius:
            return False, OUTSIDE_RADIUS

        if altitude > self._cfg.max_altitude:
            return False, TOO_HIGH

        if altitude < self._cfg.min_altitude:
            return False, TOO_LOW

        return True, SAFE


# ============================================================
# Kill Switch
# ============================================================

class KillSwitch:
    """Background keyboard listener that triggers emergency land.

    Pressing the trigger key (default: 'q') sets the ``triggered``
    event.  The main loop checks this each tick and lands immediately
    if set.

    Uses raw terminal mode on Unix so the keypress is detected
    without requiring Enter.  On Windows falls back to msvcrt.

    Usage
    -----
    >>> ks = KillSwitch()
    >>> ks.start()
    >>> # ... in control loop ...
    >>> if ks.triggered.is_set():
    ...     tello.emergency()
    >>> ks.stop()
    """

    def __init__(self, trigger_key: str = "q"):
        self.triggered = threading.Event()
        self._key = trigger_key.lower()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start listening in a daemon thread."""
        self._thread = threading.Thread(
            target=self._listen, daemon=True, name="kill-switch")
        self._thread.start()

    def stop(self) -> None:
        """Signal the listener to exit (best-effort)."""
        self._stop_event.set()

    def _listen(self) -> None:
        """Block on keyboard input until trigger key or stop."""
        try:
            if sys.platform == "win32":
                self._listen_windows()
            else:
                self._listen_unix()
        except Exception:
            pass  # terminal not available (e.g. piped stdin) — degrade gracefully

    def _listen_unix(self) -> None:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop_event.is_set():
                ch = sys.stdin.read(1).lower()
                if ch == self._key:
                    print(f"\n[KILL SWITCH] '{self._key}' pressed — emergency land!")
                    self.triggered.set()
                    return
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _listen_windows(self) -> None:
        import msvcrt
        while not self._stop_event.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getch().decode("utf-8", errors="ignore").lower()
                if ch == self._key:
                    print(f"\n[KILL SWITCH] '{self._key}' pressed — emergency land!")
                    self.triggered.set()
                    return