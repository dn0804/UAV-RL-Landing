"""
ArUco marker tracking — core math and state management.

This module contains the coordinate transforms, dropout timer, and
solvePnP wrapper used by the vision pipeline.  It has no ROS 2
dependency and can be unit tested with synthetic inputs.

The ROS 2 node that subscribes to /camera/image_raw and calls these
functions will be added during simulator integration.

Coordinate conventions:
    OpenCV camera frame:  +X right, +Y down, +Z forward (out of lens)
    Drone body frame:     +X forward, +Y left, +Z up

The transform between them is a fixed axis swap with two negations:
    body_x =  camera_z    (camera forward  → body forward)
    body_y = -camera_x    (camera right    → body left, negated)
    body_z = -camera_y    (camera down     → body up, negated)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rl_uav_package.config.constants import DROPOUT_TIMER_CAP

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ── Coordinate transform ────────────────────────────────────────────

def camera_to_body(tvec: np.ndarray) -> np.ndarray:
    """Convert an OpenCV translation vector to drone body frame.

    Parameters
    ----------
    tvec : array-like, shape (3,)
        Translation vector from solvePnP [cam_x, cam_y, cam_z].
        Units are meters.

    Returns
    -------
    body : np.ndarray, shape (3,)
        [body_forward, body_left, body_up] in meters.
    """
    tvec = np.asarray(tvec, dtype=np.float64).ravel()
    return np.array([
        tvec[2],     # camera Z (forward) → body X (forward)
        -tvec[0],    # camera X (right)   → body Y (left), negated
        -tvec[1],    # camera Y (down)    → body Z (up), negated
    ], dtype=np.float64)


def body_to_camera(body: np.ndarray) -> np.ndarray:
    """Inverse of camera_to_body.  Useful for testing round-trips."""
    body = np.asarray(body, dtype=np.float64).ravel()
    return np.array([
        -body[1],    # body Y (left) → camera X (right), negated
        -body[2],    # body Z (up)   → camera Y (down), negated
        body[0],     # body X (forward) → camera Z (forward)
    ], dtype=np.float64)


# ── Dropout timer ───────────────────────────────────────────────────

@dataclass
class DropoutState:
    """Tracks marker visibility and manages stale-data signaling.

    When the marker is visible, ``timer`` is 0 and the position channels
    update live.  When the marker disappears, the position channels freeze
    at their last known values and ``timer`` ticks upward, capped at
    DROPOUT_TIMER_CAP.

    Attributes
    ----------
    timer : int
        Steps since last successful detection.  Zero means visible.
    last_x : float
        Frozen x position (body frame) from last detection.
    last_y : float
        Frozen y position (body frame).
    last_z : float
        Frozen z position (body frame).
    last_yaw : float
        Frozen yaw from last detection.
    last_px : float
        Frozen marker pixel x in frame, normalized [-1, 1].
    last_py : float
        Frozen marker pixel y in frame, normalized [-1, 1].
    """
    timer: int = 0
    last_x: float = 0.0
    last_y: float = 0.0
    last_z: float = 0.0
    last_yaw: float = 0.0
    last_px: float = 0.0
    last_py: float = 0.0

    def reset(self, x: float, y: float, z: float, yaw: float,
              px: float = 0.0, py: float = 0.0) -> None:
        """Initialize with a known pose at episode start."""
        self.timer = 0
        self.last_x = x
        self.last_y = y
        self.last_z = z
        self.last_yaw = yaw
        self.last_px = px
        self.last_py = py

    def on_detection(self, x: float, y: float, z: float, yaw: float,
                     px: float = 0.0, py: float = 0.0) -> None:
        """Marker was detected — update stored pose, reset timer."""
        self.timer = 0
        self.last_x = x
        self.last_y = y
        self.last_z = z
        self.last_yaw = yaw
        self.last_px = px
        self.last_py = py

    def on_miss(self) -> None:
        """Marker was NOT detected — increment timer, pose stays frozen."""
        self.timer = min(self.timer + 1, DROPOUT_TIMER_CAP)

    @property
    def is_visible(self) -> bool:
        return self.timer == 0

    def get_pose(self) -> tuple[float, float, float, float, float, float]:
        """Return the current (possibly frozen) pose and pixel coords.

        Returns
        -------
        x, y, z, yaw, px, py : float
            Position, heading, and normalized pixel coords.
            Live if visible, frozen if in dropout.
        """
        return (self.last_x, self.last_y, self.last_z,
                self.last_yaw, self.last_px, self.last_py)


# ── Teleport detection (anti-flip guard) ────────────────────────────

def is_teleport(
    prev_pos: np.ndarray,
    curr_pos: np.ndarray,
    max_plausible_displacement: float = 0.5,
) -> bool:
    """Reject solvePnP solutions that imply impossible motion.

    The ArUco pose flip ambiguity can produce a detection where the marker
    appears to jump to a physically impossible position.  This filter
    rejects any detection where the implied displacement since the last
    frame exceeds the maximum plausible distance.

    Parameters
    ----------
    prev_pos : array-like, shape (3,)
        Previous position in body frame [x, y, z].
    curr_pos : array-like, shape (3,)
        Candidate new position from solvePnP.
    max_plausible_displacement : float
        Maximum distance (m) the drone can travel in one frame.
        At 10 Hz and a generous 2 m/s, this is 0.2 m.  Default of 0.5 m
        provides a safety margin for acceleration spikes.

    Returns
    -------
    teleported : bool
        True if the displacement exceeds the threshold (reject this detection).
    """
    prev = np.asarray(prev_pos, dtype=np.float64).ravel()
    curr = np.asarray(curr_pos, dtype=np.float64).ravel()
    return float(np.linalg.norm(curr - prev)) > max_plausible_displacement


# ── solvePnP wrapper ───────────────────────────────────────────────

@dataclass
class CameraIntrinsics:
    """Camera parameters loaded from camera_config.yaml.

    fx, fy : float
        Focal length in pixels.
    cx, cy : float
        Principal point (image center) in pixels.
    dist_coeffs : np.ndarray, shape (5,) or (4,)
        Distortion coefficients.  For the sim camera, these are all zero.

    Defaults match the Gazebo sim camera: 960×720 at 82° (1.43117 rad) HFOV.
    fx = width / (2 × tan(hfov / 2)) ≈ 552.28.
    """
    fx: float = 552.28
    fy: float = 552.28
    cx: float = 480.0
    cy: float = 360.0
    dist_coeffs: np.ndarray = field(
        default_factory=lambda: np.zeros(5, dtype=np.float64)
    )

    @property
    def matrix(self) -> np.ndarray:
        """3×3 camera matrix."""
        return np.array([
            [self.fx, 0.0,     self.cx],
            [0.0,     self.fy, self.cy],
            [0.0,     0.0,     1.0],
        ], dtype=np.float64)


def detect_marker(
    image: np.ndarray,
    marker_size: float,
    intrinsics: CameraIntrinsics,
    marker_id: Optional[int] = None,
    dictionary_name: str = "DICT_5X5_250",
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Detect an ArUco marker and estimate its 3D pose.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or BGR image from the camera.
    marker_size : float
        Physical side length of the marker in meters.
    intrinsics : CameraIntrinsics
        Camera calibration parameters.
    marker_id : int, optional
        If given, only accept detections matching this ID.
    dictionary_name : str
        ArUco dictionary name (e.g. "DICT_4X4_50").

    Returns
    -------
    (tvec, rvec, pixel_center) : tuple
        tvec : np.ndarray, shape (3,) — translation in OpenCV camera frame.
        rvec : np.ndarray, shape (3,) — rotation (Rodrigues) in camera frame.
        pixel_center : np.ndarray, shape (2,) — marker center in pixels (x, y).
        Returns None if no valid detection was found.
    """
    if not _CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is required for marker detection")

    # Get the ArUco dictionary and detector
    dict_id = getattr(cv2.aruco, dictionary_name)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    # Detect markers
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None or len(ids) == 0:
        return None

    # Filter by ID if specified
    if marker_id is not None:
        matches = [i for i, mid in enumerate(ids.ravel()) if mid == marker_id]
        if not matches:
            return None
        idx = matches[0]
    else:
        idx = 0

    # Pixel center of the detected marker (mean of 4 corners)
    pixel_center = corners[idx].reshape(4, 2).mean(axis=0)

    # Define the 3D marker corners (centered at origin, in marker frame)
    half = marker_size / 2.0
    obj_points = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float64)

    # solvePnP with IPPE_SQUARE — purpose-built for square markers,
    # ranks both candidate solutions by reprojection error.
    success, rvec, tvec = cv2.solvePnP(
        obj_points,
        corners[idx].reshape(4, 2),
        intrinsics.matrix,
        intrinsics.dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not success:
        return None

    return tvec.ravel(), rvec.ravel(), pixel_center