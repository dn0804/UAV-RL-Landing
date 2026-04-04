"""
Sim-to-real inference runner for DJI Tello.

Loads a trained PPO policy and runs it on real hardware at ~10 Hz.
Reuses the sim's vision pipeline, observation builder, and EMA filter.

Usage:
    python -m rl_uav_package.deploy.tello_runner \
        --model models/ppo_landing/finished.zip \
        --start-distance 2.0

Prerequisites:
    pip install djitellopy

Safety:
    - Press 'q' at any time to trigger emergency land.
    - Geofence auto-lands if the drone leaves the safe volume.
    - RC commands are clamped to --max-rc (default 40 out of 100).
"""

import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
from stable_baselines3 import PPO

from rl_uav_package.config.constants import (
    ACTION_DIM,
    MARKER_WORLD_POS,
    MARKER_CENTER_Z,
    MARKER_SIZE,
    MARKER_ID,
    MARKER_DICTIONARY,
    PAD_WORLD_POS,
    PAD_ELEVATION,
    DESK_X_MIN, DESK_X_MAX,
    DESK_Y_MIN, DESK_Y_MAX,
    CONTROL_RATE_HZ,
)
from rl_uav_package.envs.observations import ObservationBuilder
from rl_uav_package.envs.drone_env import compute_derived
from rl_uav_package.filters.ema import EMAFilter
from rl_uav_package.vision.aruco_tracker import (
    DropoutState, CameraIntrinsics, detect_marker, camera_to_body,
)
from rl_uav_package.deploy.safety import (
    SafetyConfig, GeofenceMonitor, KillSwitch, clamp_rc,
    SAFE,
)


# ============================================================
# Coordinate Frame Helpers
# ============================================================

def load_calibration(path: str) -> tuple[CameraIntrinsics, np.ndarray | None]:
    """Load camera intrinsics from a calibration .npz file.

    Expected keys: fx, fy, cx, cy, and optionally dist_coeffs.
    If no file exists, returns None and prints a warning.

    Parameters
    ----------
    path : str
        Path to ``tello_calibration.npz``.

    Returns
    -------
    (intrinsics, dist_coeffs)
        CameraIntrinsics and distortion coefficients (or None).
    """
    if not os.path.exists(path):
        print(f"[WARN] Calibration file not found: {path}")
        print("       Using Tello defaults. Run calibrate_camera.py first.")
        # Tello 720p defaults (approximate)
        return CameraIntrinsics(fx=920.0, fy=920.0, cx=480.0, cy=360.0), None

    data = np.load(path)
    intrinsics = CameraIntrinsics(
        fx=float(data["fx"]), fy=float(data["fy"]),
        cx=float(data["cx"]), cy=float(data["cy"]),
    )
    dist = data.get("dist_coeffs", None)
    print(f"[INFO] Loaded calibration: fx={intrinsics.fx:.1f}, "
          f"fy={intrinsics.fy:.1f}, cx={intrinsics.cx:.1f}, "
          f"cy={intrinsics.cy:.1f}")
    return intrinsics, dist


def reconstruct_world_pose(body_pos: np.ndarray, world_yaw: float
                           ) -> tuple[float, float, float]:
    """Compute drone world position from vision-derived body offset.

    Given the marker's position in the drone's body frame (from ArUco
    solvePnP) and the drone's world-frame heading, reconstruct where
    the drone is in world coordinates.

    Parameters
    ----------
    body_pos : np.ndarray
        Marker position in drone body frame [forward, left, up].
    world_yaw : float
        Drone heading in world frame (radians).

    Returns
    -------
    (world_x, world_y, world_z)
    """
    cos_y = math.cos(world_yaw)
    sin_y = math.sin(world_yaw)

    # Rotate body offset to world frame
    dx_world = body_pos[0] * cos_y - body_pos[1] * sin_y
    dy_world = body_pos[0] * sin_y + body_pos[1] * cos_y

    # Drone world pos = marker world pos - rotated offset
    wx = MARKER_WORLD_POS[0] - dx_world
    wy = MARKER_WORLD_POS[1] - dy_world
    wz = MARKER_CENTER_Z - body_pos[2]

    return wx, wy, wz


def compute_world_yaw(tello_yaw_deg: float, yaw_offset: float) -> float:
    """Convert Tello IMU yaw (degrees from takeoff) to world yaw.

    Parameters
    ----------
    tello_yaw_deg : float
        Yaw reading from the Tello SDK (degrees, CW-positive).
    yaw_offset : float
        Offset applied at initialization so that the Tello's takeoff
        heading maps to the correct world-frame angle (radians).

    Returns
    -------
    float
        World-frame yaw in radians (CCW-positive, standard math).
    """
    # Tello yaw: CW-positive degrees → convert to CCW-positive radians
    tello_rad = -math.radians(tello_yaw_deg)
    return tello_rad + yaw_offset


# ============================================================
# Action Scaling
# ============================================================

# Sign conventions: sim body frame → Tello RC channels.
# These may need flipping during the plumbing test.
# Sim:  action[0]=forward, action[1]=left, action[2]=up, action[3]=yaw-CCW
# Tello: fb=forward+, lr=right+, ud=up+, yaw=CW+
_SIGN_FB = 1      # sim forward → Tello forward
_SIGN_LR = -1     # sim left-positive → Tello right-positive
_SIGN_UD = 1      # sim up → Tello up
_SIGN_YAW = -1    # sim CCW → Tello CW


def scale_action(action: np.ndarray, max_rc: int) -> tuple[int, int, int, int]:
    """Convert policy output [-1, 1] to Tello RC commands.

    Parameters
    ----------
    action : np.ndarray
        4-element array with values in [-1, 1].
    max_rc : int
        Maximum RC channel magnitude (e.g. 40).

    Returns
    -------
    (lr, fb, ud, yaw) : tuple[int, int, int, int]
        Tello RC channel values (before safety clamping).
    """
    fb = int(_SIGN_FB * action[0] * max_rc)
    lr = int(_SIGN_LR * action[1] * max_rc)
    ud = int(_SIGN_UD * action[2] * max_rc)
    yaw = int(_SIGN_YAW * action[3] * max_rc)
    return lr, fb, ud, yaw


# ============================================================
# Tello State Extraction
# ============================================================

def extract_tello_state(tello, yaw_offset: float) -> dict:
    """Read Tello SDK sensors and package into a state dict.

    The dict matches the format expected by ``compute_derived()``
    from drone_env.py, so all downstream code works unchanged.

    Parameters
    ----------
    tello : djitellopy.Tello
        Connected Tello instance.
    yaw_offset : float
        World-yaw offset from initialization (radians).

    Returns
    -------
    dict
        Keys: x, y, z, vx, vy, vz, roll, pitch, yaw, tof_m.
        Positions are not populated here (set to 0) — world position
        comes from vision.  Velocities are in m/s, angles in radians.
    """
    # Tello SDK returns cm/s → m/s, degrees → radians
    vx = tello.get_speed_x() / 100.0
    vy = tello.get_speed_y() / 100.0
    vz = tello.get_speed_z() / 100.0

    roll = math.radians(tello.get_roll())
    pitch = math.radians(tello.get_pitch())
    world_yaw = compute_world_yaw(tello.get_yaw(), yaw_offset)

    # ToF height in meters
    tof_cm = tello.get_height()
    tof_m = tof_cm / 100.0

    return {
        "x": 0.0, "y": 0.0, "z": 0.0,  # filled in by vision
        "vx": vx, "vy": vy, "vz": vz,
        "roll": roll, "pitch": pitch, "yaw": world_yaw,
        "tof_m": tof_m,
    }


# ============================================================
# Main Runner
# ============================================================

def run(args):
    """Connect to Tello, load policy, and run the inference loop."""

    try:
        from djitellopy import Tello
    except ImportError:
        print("[ERROR] djitellopy not installed. Run: pip install djitellopy")
        return 1

    # ── Load model ───────────────────────────────────────────
    print(f"[INFO] Loading policy from {args.model}")
    model = PPO.load(args.model)
    print(f"[INFO] Policy loaded "
          f"({sum(p.numel() for p in model.policy.parameters()):,} params)")

    # ── Load calibration ─────────────────────────────────────
    intrinsics, dist_coeffs = load_calibration(args.calibration)

    # ── Initialize reusable sim components ───────────────────
    obs_builder = ObservationBuilder()
    ema = EMAFilter(n_channels=3)
    dropout = DropoutState()

    # ── Safety setup ─────────────────────────────────────────
    safety_cfg = SafetyConfig(
        max_rc=args.max_rc,
        geofence_radius=args.geofence,
        max_altitude=args.max_alt,
        max_duration_s=args.max_duration,
        max_dropout_s=args.max_dropout,
    )
    geofence = GeofenceMonitor(
        config=safety_cfg,
        marker_world_xy=(MARKER_WORLD_POS[0], MARKER_WORLD_POS[1]),
        control_rate_hz=CONTROL_RATE_HZ,
    )

    kill_switch = KillSwitch()
    kill_switch.start()
    print("[INFO] Kill switch active — press 'q' to emergency land")

    # ── Connect to Tello ─────────────────────────────────────
    print("[INFO] Connecting to Tello...")
    tello = Tello()
    tello.connect()
    battery = tello.get_battery()
    print(f"[INFO] Connected. Battery: {battery}%")

    if battery < 15:
        print("[WARN] Battery below 15% — aborting for safety")
        kill_switch.stop()
        return 1

    tello.streamon()
    frame_reader = tello.get_frame_read()
    time.sleep(1.0)  # let the stream stabilize

    # ── Takeoff + establish yaw reference ────────────────────
    print(f"[INFO] Starting distance: {args.start_distance:.1f}m from marker")
    print("[INFO] Ensure the Tello is facing the marker, then press Enter to take off.")
    input(">>> Press Enter to take off...")

    tello.takeoff()
    time.sleep(2.0)  # wait for stable hover

    # Record Tello yaw at takeoff for world-frame alignment.
    # Convention: the drone is facing the marker at takeoff.
    # World yaw = atan2(marker_y - drone_y, marker_x - drone_x)
    # We approximate: drone is at (marker_x + start_distance, marker_y)
    initial_world_yaw = math.atan2(
        MARKER_WORLD_POS[1] - 0.0,  # drone assumed on marker's y-axis
        MARKER_WORLD_POS[0] - (MARKER_WORLD_POS[0] + args.start_distance),
    )  # ≈ π (facing -X toward marker)
    tello_yaw_at_takeoff = math.radians(tello.get_yaw())
    yaw_offset = initial_world_yaw - (-tello_yaw_at_takeoff)
    # Note: Tello yaw is CW-positive, we convert to CCW in compute_world_yaw

    print(f"[INFO] Yaw offset: {math.degrees(yaw_offset):.1f}°")
    print("[INFO] Running policy. Press 'q' to abort at any time.\n")

    # ── Initialize state trackers ────────────────────────────
    obs_builder.reset()
    ema.reset(np.zeros(3))
    dropout.reset(x=args.start_distance, y=0.0, z=0.0, yaw=0.0, px=0.0, py=0.0)
    geofence.reset()

    prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
    world_x, world_y, world_z = (
        MARKER_WORLD_POS[0] + args.start_distance,
        MARKER_WORLD_POS[1],
        MARKER_CENTER_Z,
    )

    tick_period = 1.0 / CONTROL_RATE_HZ
    step_count = 0
    landed = False

    # ── Control loop ─────────────────────────────────────────
    try:
        while True:
            tick_start = time.monotonic()
            step_count += 1

            # ── 1. Check kill switch ─────────────────────────
            if kill_switch.triggered.is_set():
                print("[SAFETY] Kill switch — landing immediately")
                tello.land()
                landed = True
                break

            # ── 2. Read Tello state ──────────────────────────
            state = extract_tello_state(tello, yaw_offset)

            # ── 3. Grab camera frame ─────────────────────────
            frame = frame_reader.frame
            if frame is None:
                dropout.on_miss()
                vis_x, vis_y, vis_z, vis_px, vis_py = dropout.get_pose()[:3] + dropout.get_pose()[4:6]
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Undistort if calibration is available
                if dist_coeffs is not None:
                    camera_matrix = np.array([
                        [intrinsics.fx, 0, intrinsics.cx],
                        [0, intrinsics.fy, intrinsics.cy],
                        [0, 0, 1],
                    ], dtype=np.float64)
                    gray = cv2.undistort(gray, camera_matrix, dist_coeffs)

                # ── 4. ArUco detection (same pipeline as sim) ─
                result = detect_marker(
                    image=gray, marker_size=MARKER_SIZE,
                    intrinsics=intrinsics, marker_id=MARKER_ID,
                    dictionary_name=MARKER_DICTIONARY,
                )

                if result is None:
                    dropout.on_miss()
                    pose = dropout.get_pose()
                    vis_x, vis_y, vis_z = pose[0], pose[1], pose[2]
                    vis_px, vis_py = pose[4], pose[5]
                else:
                    tvec, rvec, pixel_center = result
                    body_pos = camera_to_body(tvec)

                    vis_px = (pixel_center[0] - intrinsics.cx) / intrinsics.cx
                    vis_py = (pixel_center[1] - intrinsics.cy) / intrinsics.cy

                    dropout.on_detection(
                        x=float(body_pos[0]), y=float(body_pos[1]),
                        z=float(body_pos[2]),
                        yaw=0.0, px=float(vis_px), py=float(vis_py),
                    )
                    vis_x = float(body_pos[0])
                    vis_y = float(body_pos[1])
                    vis_z = float(body_pos[2])

                    # Update world position estimate from vision
                    world_x, world_y, world_z = reconstruct_world_pose(
                        body_pos, state["yaw"])

            # Fill in world position for compute_derived
            state["x"] = world_x
            state["y"] = world_y
            state["z"] = world_z

            # ── 5. Derived quantities ────────────────────────
            derived = compute_derived(state)

            # ── 6. EMA velocity filter ───────────────────────
            vel_raw = np.array([state["vx"], state["vy"], state["vz"]])
            vel_filtered = ema.update(vel_raw)

            # ── 7. Build observation ─────────────────────────
            obs = obs_builder.build(
                x=vis_x, y=vis_y, z=vis_z,
                yaw=derived["yaw_error"],
                vx=vel_filtered[0], vy=vel_filtered[1], vz=vel_filtered[2],
                z_tof_raw=state["tof_m"],
                roll=state["roll"], pitch=state["pitch"],
                dropout_timer=dropout.timer,
                marker_px=vis_px, marker_py=vis_py,
            )

            if np.isnan(obs).any():
                obs = np.nan_to_num(obs, nan=0.0)

            # ── 8. Policy inference ──────────────────────────
            action, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action, dtype=np.float32)
            obs_builder.push_action(action)

            # ── 9. Geofence check ────────────────────────────
            marker_visible = (dropout.timer == 0)
            is_safe, reason = geofence.check(
                world_x, world_y, state["tof_m"], marker_visible)

            if not is_safe:
                print(f"[SAFETY] Geofence violation: {reason} — landing")
                tello.land()
                landed = True
                break

            # ── 10. Scale + clamp + send ─────────────────────
            lr, fb, ud, yaw_cmd = scale_action(action, safety_cfg.max_rc)
            lr, fb, ud, yaw_cmd = clamp_rc(
                lr, fb, ud, yaw_cmd, safety_cfg.max_rc)

            tello.send_rc_control(lr, fb, ud, yaw_cmd)

            # ── 11. Landing detection ────────────────────────
            if state["tof_m"] < 0.15 and derived["d_pad"] < 0.30:
                print(f"\n[LANDED] d_pad={derived['d_pad']:.3f}m, "
                      f"tof={state['tof_m']:.3f}m")
                tello.send_rc_control(0, 0, 0, 0)
                tello.land()
                landed = True
                break

            # ── 12. Logging ──────────────────────────────────
            if step_count % 10 == 0:
                status = "MARKER" if marker_visible else "DROPOUT"
                print(
                    f"  [{step_count:4d}] {status:7s}  "
                    f"d_pad={derived['d_pad']:.2f}  "
                    f"tof={state['tof_m']:.2f}  "
                    f"v_xy={math.hypot(state['vx'], state['vy']):.2f}  "
                    f"rc=({lr:+4d},{fb:+4d},{ud:+4d},{yaw_cmd:+4d})"
                )

            # ── 13. Rate limiting ────────────────────────────
            prev_action = action.copy()
            elapsed = time.monotonic() - tick_start
            sleep_time = tick_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C — landing...")
        tello.land()
        landed = True

    finally:
        # ── Cleanup ──────────────────────────────────────────
        if not landed:
            print("[INFO] Ensuring drone lands...")
            try:
                tello.land()
            except Exception:
                pass

        kill_switch.stop()
        tello.streamoff()
        tello.end()
        print("[INFO] Done.")

    return 0


# ============================================================
# CLI
# ============================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run a trained PPO landing policy on a real DJI Tello.")

    # ── Required ─────────────────────────────────────────────
    p.add_argument("--model", type=str, required=True,
                   help="Path to trained model .zip (e.g. models/ppo_landing/finished.zip)")

    # ── Starting position ────────────────────────────────────
    p.add_argument("--start-distance", type=float, default=2.0,
                   help="Distance from marker at takeoff (meters, default: 2.0). "
                        "The drone should be placed this far from the marker, facing it.")

    # ── Camera calibration ───────────────────────────────────
    p.add_argument("--calibration", type=str,
                   default="rl_uav_package/deploy/tello_calibration.npz",
                   help="Path to camera calibration .npz file")

    # ── Safety parameters ────────────────────────────────────
    p.add_argument("--max-rc", type=int, default=40,
                   help="Max RC channel magnitude (0-100, default: 40)")
    p.add_argument("--geofence", type=float, default=3.0,
                   help="Geofence radius in meters (default: 3.0)")
    p.add_argument("--max-alt", type=float, default=2.0,
                   help="Maximum altitude in meters (default: 2.0)")
    p.add_argument("--max-duration", type=float, default=60.0,
                   help="Max episode duration in seconds (default: 60)")
    p.add_argument("--max-dropout", type=float, default=5.0,
                   help="Max seconds without marker before auto-land (default: 5.0)")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())