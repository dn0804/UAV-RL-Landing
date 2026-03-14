"""
Unit tests for envs/drone_env.py — pure computation functions.

The step/reset integration loop requires a running Gazebo + ROS 2 stack
and cannot be tested here.  These tests cover the module-level pure
functions that are extractable and independently verifiable.

Run with:  python -m pytest test_drone_env.py -v
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from rl_uav_package.envs.drone_env import quat_to_euler, compute_derived
from rl_uav_package.config.constants import (
    MARKER_Z, PAD_WORLD_POS, MARKER_WORLD_POS,
    LANDING_PAD_FORWARD_OFFSET,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _quat(roll=0.0, pitch=0.0, yaw=0.0):
    """Create a mock quaternion namespace from Euler angles.
    Applies yaw, then pitch, then roll (ZYX convention)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

    return SimpleNamespace(
        w=cr * cp * cy + sr * sp * sy,
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
    )


def _state(**overrides):
    """Default mid-flight state dict."""
    s = dict(x=2.0, y=0.0, z=1.2, yaw=math.pi, roll=0.0, pitch=0.0)
    s.update(overrides)
    return s


# ── Quaternion to Euler ─────────────────────────────────────────────

class TestQuatToEuler:
    def test_identity(self):
        """No rotation → all zeros."""
        q = _quat()
        r, p, y = quat_to_euler(q)
        assert r == pytest.approx(0.0, abs=1e-6)
        assert p == pytest.approx(0.0, abs=1e-6)
        assert y == pytest.approx(0.0, abs=1e-6)

    def test_yaw_90(self):
        q = _quat(yaw=math.pi / 2)
        r, p, y = quat_to_euler(q)
        assert r == pytest.approx(0.0, abs=1e-4)
        assert p == pytest.approx(0.0, abs=1e-4)
        assert y == pytest.approx(math.pi / 2, abs=1e-4)

    def test_yaw_negative(self):
        q = _quat(yaw=-math.pi / 4)
        _, _, y = quat_to_euler(q)
        assert y == pytest.approx(-math.pi / 4, abs=1e-4)

    def test_yaw_pi(self):
        """Facing backward (180°)."""
        q = _quat(yaw=math.pi)
        _, _, y = quat_to_euler(q)
        assert abs(y) == pytest.approx(math.pi, abs=1e-4)

    def test_pitch_30(self):
        q = _quat(pitch=math.radians(30))
        _, p, _ = quat_to_euler(q)
        assert p == pytest.approx(math.radians(30), abs=1e-4)

    def test_roll_15(self):
        q = _quat(roll=math.radians(15))
        r, _, _ = quat_to_euler(q)
        assert r == pytest.approx(math.radians(15), abs=1e-4)

    def test_combined_small_angles(self):
        """Small combined rotation — linear regime, all axes should be close."""
        q = _quat(roll=0.1, pitch=0.05, yaw=0.2)
        r, p, y = quat_to_euler(q)
        assert r == pytest.approx(0.1, abs=0.01)
        assert p == pytest.approx(0.05, abs=0.01)
        assert y == pytest.approx(0.2, abs=0.01)

    def test_gimbal_lock_pitch_90(self):
        """At pitch = ±90°, gimbal lock is expected.  Just check pitch."""
        q = _quat(pitch=math.pi / 2)
        _, p, _ = quat_to_euler(q)
        assert p == pytest.approx(math.pi / 2, abs=1e-3)


# ── Derived quantities ──────────────────────────────────────────────

class TestComputeDerived:
    def test_d_pad_directly_ahead(self):
        """Drone at (2.0, 0, z) → pad at (0.30, 0, z).
        d_pad = |2.0 - 0.30| = 1.70."""
        d = compute_derived(_state(x=2.0, y=0.0))
        assert d["d_pad"] == pytest.approx(1.70, abs=0.001)

    def test_d_pad_on_pad(self):
        """Drone right on the pad center → d_pad ≈ 0."""
        d = compute_derived(_state(
            x=LANDING_PAD_FORWARD_OFFSET, y=0.0,
        ))
        assert d["d_pad"] == pytest.approx(0.0, abs=0.001)

    def test_d_pad_lateral_offset(self):
        """Pad at (0.30, 0).  Drone at (0.30, 0.50).
        d_pad = 0.50."""
        d = compute_derived(_state(x=LANDING_PAD_FORWARD_OFFSET, y=0.50))
        assert d["d_pad"] == pytest.approx(0.50, abs=0.001)

    def test_d_marker_at_origin(self):
        """Drone at (2.0, 0) → marker at (0, 0) → d_marker = 2.0."""
        d = compute_derived(_state(x=2.0, y=0.0))
        assert d["d_marker"] == pytest.approx(2.0, abs=0.001)

    def test_d_marker_with_lateral(self):
        """Drone at (3.0, 4.0) → d_marker = 5.0."""
        d = compute_derived(_state(x=3.0, y=4.0))
        assert d["d_marker"] == pytest.approx(5.0, abs=0.001)

    def test_h_above_marker(self):
        """Drone at z=1.5, MARKER_Z=0.90 → h = 0.60."""
        d = compute_derived(_state(z=1.5))
        assert d["h_above_marker"] == pytest.approx(1.5 - MARKER_Z, abs=0.001)

    def test_h_above_marker_below(self):
        """Drone below marker → negative h."""
        d = compute_derived(_state(z=MARKER_Z - 0.1))
        assert d["h_above_marker"] < 0

    def test_yaw_error_facing_marker(self):
        """Drone at (2.0, 0), yaw=π (facing -x, toward marker at origin).
        Bearing to marker = atan2(0-0, 0-2) = π.
        yaw_error = π - π = 0."""
        d = compute_derived(_state(x=2.0, y=0.0, yaw=math.pi))
        assert d["yaw_error"] == pytest.approx(0.0, abs=0.01)

    def test_yaw_error_perpendicular(self):
        """Drone at (2.0, 0), yaw=π/2 (facing +y, 90° off from marker).
        Bearing = π.  Error = π/2 - π = -π/2."""
        d = compute_derived(_state(x=2.0, y=0.0, yaw=math.pi / 2))
        assert d["yaw_error"] == pytest.approx(-math.pi / 2, abs=0.01)

    def test_yaw_error_wrapping(self):
        """Yaw error must be wrapped to [-π, π]."""
        d = compute_derived(_state(x=2.0, y=0.0, yaw=-0.1))
        # Bearing = π.  Raw diff = -0.1 - π ≈ -3.24.
        # Wrapped should be ≈ +3.04 → actually atan2(sin(-3.24), cos(-3.24))
        assert -math.pi <= d["yaw_error"] <= math.pi

    def test_yaw_error_from_side(self):
        """Drone at (2.0, 2.0), yaw facing marker (at origin).
        Bearing = atan2(-2, -2) = -3π/4.
        If yaw = bearing → error = 0."""
        bearing = math.atan2(0.0 - 2.0, 0.0 - 2.0)
        d = compute_derived(_state(x=2.0, y=2.0, yaw=bearing))
        assert d["yaw_error"] == pytest.approx(0.0, abs=0.01)


class TestComputeDerivedToF:
    def test_level_flight_tof_equals_z(self):
        """No tilt → z_tof = z."""
        d = compute_derived(_state(z=1.5, roll=0.0, pitch=0.0))
        assert d["z_tof"] == pytest.approx(1.5, abs=0.001)
        assert d["z_tof_raw"] == pytest.approx(1.5, abs=0.001)

    def test_pitched_tof_less_than_z(self):
        """Pitched → cosine correction reduces z_tof."""
        pitch = math.radians(20)
        d = compute_derived(_state(z=1.0, pitch=pitch, roll=0.0))
        expected = 1.0 * math.cos(pitch)
        assert d["z_tof"] == pytest.approx(expected, abs=0.001)
        # Raw should still be 1.0
        assert d["z_tof_raw"] == pytest.approx(1.0, abs=0.001)

    def test_rolled_tof(self):
        roll = math.radians(15)
        d = compute_derived(_state(z=0.8, roll=roll, pitch=0.0))
        expected = 0.8 * math.cos(roll)
        assert d["z_tof"] == pytest.approx(expected, abs=0.001)


class TestComputeDerivedKeys:
    def test_all_keys_present(self):
        d = compute_derived(_state())
        expected = {"d_pad", "d_marker", "h_above_marker",
                    "yaw_error", "z_tof_raw", "z_tof"}
        assert set(d.keys()) == expected

    def test_all_values_are_float(self):
        d = compute_derived(_state())
        for k, v in d.items():
            assert isinstance(v, float), f"{k} is {type(v)}"
