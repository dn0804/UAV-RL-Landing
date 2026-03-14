"""
Unit tests for vision/aruco_tracker.py

Tests cover the pure math: coordinate transforms, dropout timer state
machine, teleport rejection, and camera intrinsics.  The solvePnP
wrapper (detect_marker) requires actual images and is tested during
simulator integration, not here.

Run with:  python -m pytest test_aruco_tracker.py -v
"""

import math
import numpy as np
import pytest

from rl_uav_package.vision.aruco_tracker import (
    camera_to_body,
    body_to_camera,
    DropoutState,
    is_teleport,
    CameraIntrinsics,
)
from rl_uav_package.config.constants import DROPOUT_TIMER_CAP


# ── Coordinate transform ────────────────────────────────────────────

class TestCameraToBody:
    def test_forward_maps_correctly(self):
        """Camera +Z (forward out of lens) → body +X (forward).
        A marker 2m in front of the camera → body_x = 2."""
        body = camera_to_body([0.0, 0.0, 2.0])
        assert body[0] == pytest.approx(2.0)   # body X (forward)
        assert body[1] == pytest.approx(0.0)   # body Y (left)
        assert body[2] == pytest.approx(0.0)   # body Z (up)

    def test_right_maps_to_negative_left(self):
        """Camera +X (right) → body −Y (right = negative left).
        A marker 1m to the camera's right → body_y = −1."""
        body = camera_to_body([1.0, 0.0, 0.0])
        assert body[0] == pytest.approx(0.0)
        assert body[1] == pytest.approx(-1.0)
        assert body[2] == pytest.approx(0.0)

    def test_down_maps_to_negative_up(self):
        """Camera +Y (down) → body −Z (down = negative up).
        A marker 0.5m below the camera → body_z = −0.5."""
        body = camera_to_body([0.0, 0.5, 0.0])
        assert body[0] == pytest.approx(0.0)
        assert body[1] == pytest.approx(0.0)
        assert body[2] == pytest.approx(-0.5)

    def test_combined_vector(self):
        """Marker at camera [0.3, −0.2, 1.5] →
        body [1.5, −0.3, 0.2]."""
        body = camera_to_body([0.3, -0.2, 1.5])
        assert body[0] == pytest.approx(1.5)    # cam_z → body_x
        assert body[1] == pytest.approx(-0.3)   # -cam_x → body_y
        assert body[2] == pytest.approx(0.2)    # -cam_y → body_z

    def test_output_shape(self):
        body = camera_to_body([1.0, 2.0, 3.0])
        assert body.shape == (3,)

    def test_accepts_2d_input(self):
        """solvePnP sometimes returns shape (3, 1) — should be handled."""
        body = camera_to_body(np.array([[0.0], [0.0], [2.0]]))
        assert body[0] == pytest.approx(2.0)


class TestBodyToCamera:
    def test_inverse_of_camera_to_body(self):
        """body_to_camera(camera_to_body(v)) == v."""
        original = np.array([0.3, -0.2, 1.5])
        roundtrip = body_to_camera(camera_to_body(original))
        np.testing.assert_array_almost_equal(roundtrip, original)

    def test_round_trip_other_direction(self):
        """camera_to_body(body_to_camera(v)) == v."""
        original = np.array([1.5, -0.3, 0.2])
        roundtrip = camera_to_body(body_to_camera(original))
        np.testing.assert_array_almost_equal(roundtrip, original)

    def test_identity_at_zero(self):
        body = camera_to_body([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(body, [0.0, 0.0, 0.0])


# ── Dropout timer ───────────────────────────────────────────────────

class TestDropoutState:
    def test_initial_state(self):
        d = DropoutState()
        assert d.timer == 0
        assert d.is_visible

    def test_reset_sets_pose(self):
        d = DropoutState()
        d.on_miss()
        d.on_miss()
        d.reset(x=1.0, y=2.0, z=3.0, yaw=0.5)
        assert d.timer == 0
        assert d.is_visible
        x, y, z, yaw = d.get_pose()
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(2.0)
        assert z == pytest.approx(3.0)
        assert yaw == pytest.approx(0.5)

    def test_detection_resets_timer(self):
        d = DropoutState()
        d.on_miss()
        d.on_miss()
        assert d.timer == 2
        d.on_detection(x=1.0, y=0.0, z=0.5, yaw=0.0)
        assert d.timer == 0
        assert d.is_visible

    def test_detection_updates_pose(self):
        d = DropoutState()
        d.reset(x=0.0, y=0.0, z=0.0, yaw=0.0)
        d.on_detection(x=2.0, y=1.0, z=0.8, yaw=-0.3)
        x, y, z, yaw = d.get_pose()
        assert x == pytest.approx(2.0)
        assert yaw == pytest.approx(-0.3)

    def test_miss_increments_timer(self):
        d = DropoutState()
        d.on_miss()
        assert d.timer == 1
        assert not d.is_visible
        d.on_miss()
        assert d.timer == 2

    def test_miss_freezes_pose(self):
        d = DropoutState()
        d.on_detection(x=5.0, y=3.0, z=1.0, yaw=1.2)
        d.on_miss()
        d.on_miss()
        d.on_miss()
        x, y, z, yaw = d.get_pose()
        # Pose should still be the last detection
        assert x == pytest.approx(5.0)
        assert yaw == pytest.approx(1.2)

    def test_timer_caps_at_max(self):
        d = DropoutState()
        for _ in range(DROPOUT_TIMER_CAP + 50):
            d.on_miss()
        assert d.timer == DROPOUT_TIMER_CAP

    def test_timer_resets_immediately_on_reacquire(self):
        """When the marker is reacquired, timer goes to zero on that
        same step — no one-step lag."""
        d = DropoutState()
        for _ in range(10):
            d.on_miss()
        assert d.timer == 10
        d.on_detection(x=1.0, y=0.0, z=0.5, yaw=0.0)
        assert d.timer == 0


# ── Teleport rejection ─────────────────────────────────────────────

class TestTeleportDetection:
    def test_small_movement_accepted(self):
        prev = np.array([1.0, 0.0, 0.5])
        curr = np.array([1.02, 0.01, 0.49])
        assert not is_teleport(prev, curr)

    def test_large_jump_rejected(self):
        prev = np.array([1.0, 0.0, 0.5])
        curr = np.array([3.0, 2.0, 0.5])  # ~2.8 m jump
        assert is_teleport(prev, curr)

    def test_exactly_at_threshold(self):
        """Displacement == threshold should NOT be rejected (using >)."""
        prev = np.array([0.0, 0.0, 0.0])
        curr = np.array([0.5, 0.0, 0.0])  # exactly 0.5 m
        assert not is_teleport(prev, curr, max_plausible_displacement=0.5)

    def test_just_over_threshold(self):
        prev = np.array([0.0, 0.0, 0.0])
        curr = np.array([0.501, 0.0, 0.0])
        assert is_teleport(prev, curr, max_plausible_displacement=0.5)

    def test_custom_threshold(self):
        prev = np.array([0.0, 0.0, 0.0])
        curr = np.array([0.3, 0.0, 0.0])
        assert not is_teleport(prev, curr, max_plausible_displacement=0.5)
        assert is_teleport(prev, curr, max_plausible_displacement=0.2)

    def test_3d_distance(self):
        """Displacement is 3D Euclidean, not per-axis."""
        prev = np.array([0.0, 0.0, 0.0])
        # Each axis: 0.3 m.  L2 = sqrt(3 × 0.09) ≈ 0.52 m
        curr = np.array([0.3, 0.3, 0.3])
        assert is_teleport(prev, curr, max_plausible_displacement=0.5)


# ── Camera intrinsics ──────────────────────────────────────────────

class TestCameraIntrinsics:
    def test_default_values(self):
        c = CameraIntrinsics()
        assert c.fx > 0
        assert c.fy > 0
        assert c.cx > 0
        assert c.cy > 0

    def test_matrix_shape(self):
        c = CameraIntrinsics()
        assert c.matrix.shape == (3, 3)

    def test_matrix_structure(self):
        c = CameraIntrinsics(fx=500, fy=600, cx=320, cy=240)
        m = c.matrix
        assert m[0, 0] == pytest.approx(500)   # fx
        assert m[1, 1] == pytest.approx(600)   # fy
        assert m[0, 2] == pytest.approx(320)   # cx
        assert m[1, 2] == pytest.approx(240)   # cy
        assert m[2, 2] == pytest.approx(1.0)
        # Off-diagonal zeros
        assert m[0, 1] == pytest.approx(0.0)
        assert m[1, 0] == pytest.approx(0.0)
        assert m[2, 0] == pytest.approx(0.0)
        assert m[2, 1] == pytest.approx(0.0)

    def test_dist_coeffs_default_zeros(self):
        c = CameraIntrinsics()
        np.testing.assert_array_equal(c.dist_coeffs, np.zeros(5))

    def test_custom_distortion(self):
        dist = np.array([0.1, -0.2, 0.0, 0.0, 0.05])
        c = CameraIntrinsics(dist_coeffs=dist)
        np.testing.assert_array_almost_equal(c.dist_coeffs, dist)
