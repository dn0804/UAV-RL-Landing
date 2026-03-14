"""
Unit tests for envs/spawner.py

Run with:  python -m pytest test_spawner.py -v
"""

import math
import numpy as np
import pytest

from rl_uav_package.envs.spawner import Spawner, Z_SPAWN_MAX, MAX_REJECTION_ATTEMPTS
from rl_uav_package.config.constants import (
    MARKER_Z, CURRICULUM_STAGES,
    CAMERA_VFOV_RAD,
)


# ── Helpers ──────────────────────────────────────────────────────────

N_SAMPLES = 2000  # enough for distribution checks without being slow


def _sample_many(spawner, n=N_SAMPLES):
    """Collect n spawn dicts."""
    return [spawner.sample() for _ in range(n)]


# ── Basic output format ─────────────────────────────────────────────

class TestOutputFormat:
    def test_returns_dict_with_expected_keys(self):
        s = Spawner(stage=1)
        spawn = s.sample()
        assert set(spawn.keys()) == {"x", "y", "z", "yaw"}

    def test_all_values_are_float(self):
        s = Spawner(stage=1)
        spawn = s.sample()
        for key, val in spawn.items():
            assert isinstance(val, float), f"{key} is {type(val)}"


# ── Height constraint: z >= MARKER_Z ────────────────────────────────

class TestHeightConstraint:
    def test_all_spawns_above_marker(self):
        """Every spawn must have z >= MARKER_Z."""
        s = Spawner(stage=3, rng=np.random.default_rng(42))
        for spawn in _sample_many(s):
            assert spawn["z"] >= MARKER_Z, (
                f"Spawn z={spawn['z']:.3f} below MARKER_Z={MARKER_Z}"
            )

    def test_all_spawns_below_ceiling(self):
        s = Spawner(stage=3, rng=np.random.default_rng(42))
        for spawn in _sample_many(s):
            assert spawn["z"] <= Z_SPAWN_MAX, (
                f"Spawn z={spawn['z']:.3f} above Z_SPAWN_MAX={Z_SPAWN_MAX}"
            )


# ── Distance constraint per stage ───────────────────────────────────

class TestDistanceConstraint:
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_horizontal_distance_in_range(self, stage):
        cfg = CURRICULUM_STAGES[stage]
        d_min, d_max = cfg["d_min"], cfg["d_max"]
        s = Spawner(stage=stage, rng=np.random.default_rng(42))

        for spawn in _sample_many(s, n=1000):
            d = math.hypot(spawn["x"], spawn["y"])
            assert d >= d_min - 1e-6, (
                f"Stage {stage}: d={d:.3f} below d_min={d_min}"
            )
            assert d <= d_max + 1e-6, (
                f"Stage {stage}: d={d:.3f} above d_max={d_max}"
            )


# ── Approach angle constraint ───────────────────────────────────────

class TestAngleConstraint:
    @pytest.mark.parametrize("stage", [1, 2, 3])
    def test_approach_angle_within_bounds(self, stage):
        cfg = CURRICULUM_STAGES[stage]
        angle_max = cfg["angle_max"]
        s = Spawner(stage=stage, rng=np.random.default_rng(42))

        for spawn in _sample_many(s, n=1000):
            # Approach angle = atan2(y, x) from the wall normal
            theta = math.atan2(spawn["y"], spawn["x"])
            assert abs(theta) <= angle_max + 1e-6, (
                f"Stage {stage}: θ={math.degrees(theta):.1f}° "
                f"exceeds ±{math.degrees(angle_max):.1f}°"
            )


# ── Yaw: drone faces the marker ────────────────────────────────────

class TestYawOrientation:
    def test_yaw_points_toward_marker(self):
        """The drone's heading should point from (x, y) toward (0, 0)."""
        s = Spawner(stage=3, rng=np.random.default_rng(42))

        for spawn in _sample_many(s, n=500):
            expected_yaw = math.atan2(0.0 - spawn["y"], 0.0 - spawn["x"])
            # Wrap both to [-π, π] and compare
            diff = abs(math.atan2(
                math.sin(spawn["yaw"] - expected_yaw),
                math.cos(spawn["yaw"] - expected_yaw),
            ))
            assert diff < 1e-6, (
                f"Yaw mismatch: got {spawn['yaw']:.4f}, "
                f"expected {expected_yaw:.4f}"
            )

    def test_yaw_is_pi_when_directly_ahead(self):
        """θ = 0 → drone at (+d, 0) facing -x → yaw = π."""
        s = Spawner(stage=1, rng=np.random.default_rng(42))
        # Stage 1 has ±10° angle — sample many and find one near θ ≈ 0
        found = False
        for spawn in _sample_many(s, n=500):
            theta = abs(math.atan2(spawn["y"], spawn["x"]))
            if theta < math.radians(1):  # within 1° of dead-ahead
                assert abs(abs(spawn["yaw"]) - math.pi) < 0.02
                found = True
                break
        assert found, "No near-zero-angle spawn found in 500 samples"


# ── X is always positive (drone is in front of the wall) ───────────

class TestPositiveX:
    def test_x_always_positive(self):
        """The drone spawns in the room (x > 0), never behind the wall."""
        s = Spawner(stage=3, rng=np.random.default_rng(42))
        for spawn in _sample_many(s):
            assert spawn["x"] > 0, f"Spawn x={spawn['x']:.3f} is behind the wall"


# ── Camera FOV rejection ───────────────────────────────────────────

class TestFOVRejection:
    def test_high_close_spawn_rejected_or_valid(self):
        """If the drone is very high and very close, the marker could drop
        below the camera frame.  The spawner should either reject these
        or only produce ones that pass the FOV check."""
        s = Spawner(stage=1, rng=np.random.default_rng(42))
        for spawn in _sample_many(s, n=500):
            d = math.hypot(spawn["x"], spawn["y"])
            h = spawn["z"] - MARKER_Z
            if d > 0.01:
                vert_angle = math.atan2(h, d)
                # Must be within the camera's vertical FOV (with margin)
                assert abs(vert_angle) <= CAMERA_VFOV_RAD / 2.0, (
                    f"Vertical angle {math.degrees(vert_angle):.1f}° "
                    f"exceeds VFOV/2={math.degrees(CAMERA_VFOV_RAD/2):.1f}°"
                )


# ── Stage transitions ──────────────────────────────────────────────

class TestStageTransitions:
    def test_set_stage_changes_range(self):
        s = Spawner(stage=1)
        assert s.stage == 1
        s.set_stage(3)
        assert s.stage == 3

    def test_stage_3_produces_longer_distances(self):
        rng1 = np.random.default_rng(42)
        rng3 = np.random.default_rng(42)
        s1 = Spawner(stage=1, rng=rng1)
        s3 = Spawner(stage=3, rng=rng3)

        dists_1 = [math.hypot(sp["x"], sp["y"]) for sp in _sample_many(s1, 500)]
        dists_3 = [math.hypot(sp["x"], sp["y"]) for sp in _sample_many(s3, 500)]

        assert max(dists_3) > max(dists_1)

    def test_stage_3_produces_wider_angles(self):
        s = Spawner(stage=3, rng=np.random.default_rng(42))
        angles = [abs(math.atan2(sp["y"], sp["x"])) for sp in _sample_many(s, 500)]
        # Stage 3 allows ±60° — at least some samples should exceed 25° (Stage 2 max)
        assert max(angles) > math.radians(25)

    def test_invalid_stage_raises(self):
        with pytest.raises(ValueError):
            Spawner(stage=99)

    def test_set_invalid_stage_raises(self):
        s = Spawner(stage=1)
        with pytest.raises(ValueError):
            s.set_stage(0)


# ── Distribution quality ───────────────────────────────────────────

class TestDistribution:
    def test_distances_cover_range(self):
        """Samples should cover most of the [d_min, d_max] range."""
        cfg = CURRICULUM_STAGES[3]
        s = Spawner(stage=3, rng=np.random.default_rng(42))
        dists = [math.hypot(sp["x"], sp["y"]) for sp in _sample_many(s)]

        assert min(dists) < cfg["d_min"] + 0.2
        assert max(dists) > cfg["d_max"] - 0.2

    def test_angles_cover_range(self):
        """Samples should span both positive and negative angles."""
        s = Spawner(stage=3, rng=np.random.default_rng(42))
        angles = [math.atan2(sp["y"], sp["x"]) for sp in _sample_many(s)]

        assert min(angles) < -math.radians(10)
        assert max(angles) > math.radians(10)

    def test_heights_cover_range(self):
        """Samples should span from near MARKER_Z up toward Z_SPAWN_MAX."""
        s = Spawner(stage=3, rng=np.random.default_rng(42))
        heights = [sp["z"] for sp in _sample_many(s)]

        assert min(heights) < MARKER_Z + 0.15
        assert max(heights) > Z_SPAWN_MAX - 0.15


# ── Reproducibility ────────────────────────────────────────────────

class TestReproducibility:
    def test_same_seed_same_sequence(self):
        s1 = Spawner(stage=2, rng=np.random.default_rng(123))
        s2 = Spawner(stage=2, rng=np.random.default_rng(123))

        for _ in range(20):
            a = s1.sample()
            b = s2.sample()
            assert a["x"] == pytest.approx(b["x"])
            assert a["y"] == pytest.approx(b["y"])
            assert a["z"] == pytest.approx(b["z"])
            assert a["yaw"] == pytest.approx(b["yaw"])

    def test_different_seeds_differ(self):
        s1 = Spawner(stage=2, rng=np.random.default_rng(1))
        s2 = Spawner(stage=2, rng=np.random.default_rng(999))

        a = s1.sample()
        b = s2.sample()
        # Extremely unlikely to be identical
        assert a["x"] != pytest.approx(b["x"])
