"""
Gymnasium environment wrapper for Tello RL landing.

This is the central orchestrator.  It owns the ROS 2 node lifecycle, the
Gymnasium step/reset loop, and Gazebo teleportation.  All computation is
delegated to specialist modules:

    rewards.py        -> per-step shaping reward
    termination.py    -> episode-ending conditions
    observations.py   -> 31-dim normalized vector assembly
    spawner.py        -> rejection-sampled spawn positions
    ema.py            -> velocity smoothing
    aruco_tracker.py  -> ArUco detection, coordinate transforms, dropout state

Position observations come from the vision pipeline (ArUco solvePnP →
body-frame transform).  Rewards and termination use world-frame odom
ground truth (no sim-to-real concern — rewards aren't computed on hardware).
"""

import math
import subprocess
import threading
import time
import os

import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Empty, Bool

from rl_uav_package.config.constants import (
    OBS_DIM, ACTION_DIM, ACTION_LOW, ACTION_HIGH,
    MAX_STEPS, CONTROL_RATE_HZ,
    MARKER_Z, PAD_WORLD_POS, MARKER_WORLD_POS,
    LANDING_PAD_FORWARD_OFFSET,
    GZ_WORLD_NAME, GZ_DRONE_MODEL_NAME,
    PAD_ELEVATION, CURRICULUM_STAGES,
    SUCCESS_VZ_MAX, SUCCESS_VXY_MAX,
    SUCCESS_D_XY_MAX,
    Z_MAX,
    MARKER_SIZE, MARKER_ID, MARKER_DICTIONARY,
    SIM_CAMERA_FX, SIM_CAMERA_FY, SIM_CAMERA_CX, SIM_CAMERA_CY,
)
from rl_uav_package.envs.rewards import compute_reward
from rl_uav_package.envs.termination import check_termination
from rl_uav_package.envs.observations import ObservationBuilder
from rl_uav_package.envs.spawner import Spawner
from rl_uav_package.filters.ema import EMAFilter
from rl_uav_package.vision.aruco_tracker import (
    DropoutState, CameraIntrinsics, detect_marker, camera_to_body, is_teleport,
)


class DroneEnv(gym.Env):
    """Gymnasium environment for autonomous UAV precision landing.

    Parameters
    ----------
    initial_stage : int
        Curriculum stage to start in (0, 1, 2, or 3).
    seed : int, optional
        Random seed for reproducibility.
    """

    metadata = {"render_modes": []}

    def __init__(self, initial_stage: int = 0, seed: int = 0):
        super().__init__()

        # ── Spaces ───────────────────────────────────────────────
        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH,
            shape=(ACTION_DIM,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(OBS_DIM,), dtype=np.float32,
        )

        # ── Sub-modules ──────────────────────────────────────────
        self._rng = np.random.default_rng(seed)
        self._obs_builder = ObservationBuilder()
        self._ema = EMAFilter(n_channels=3)
        self._spawner = Spawner(stage=initial_stage, rng=self._rng)
        self._dropout = DropoutState()

        # ── Episode state ────────────────────────────────────────
        self._step_count = 0
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._d_pad_prev = 0.0
        self._z_prev = 0.0

        # ── ROS 2 ────────────────────────────────────────────────
        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node("rl_drone_env")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._odom_sub = self._node.create_subscription(
            Odometry, "/odom", self._odom_cb, qos,
        )
        self._cmd_pub = self._node.create_publisher(Twist, "/cmd_vel", 10)
        self._takeoff_pub = self._node.create_publisher(Empty, "/takeoff", 10)
        self._land_pub = self._node.create_publisher(Empty, "/land", 10)
        self._enable_pub = self._node.create_publisher(Bool, "/enable", 10)

        # Odom synchronization
        self._latest_odom = None
        self._odom_event = threading.Event()

        # Camera subscription (30 Hz from Gazebo)
        self._latest_image = None
        self._image_event = threading.Event()
        self._cam_sub = self._node.create_subscription(
            RosImage, "/camera/image_raw", self._image_cb, qos,
        )

        # ── Vision pipeline ───────────────────────────────────────
        self._intrinsics = CameraIntrinsics(
            fx=SIM_CAMERA_FX, fy=SIM_CAMERA_FY,
            cx=SIM_CAMERA_CX, cy=SIM_CAMERA_CY,
        )
        self._marker_prev_pos = None  # for teleport/flip rejection

        # Spin ROS 2 in background
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        # Start persistent teleport helper (avoids subprocess-per-teleport)
        # Dynamically search upwards to find the workspace root containing 'scripts'
        current_dir = os.path.dirname(os.path.abspath(__file__))
        helper_path = None
        
        while current_dir != '/':
            potential_path = os.path.join(current_dir, 'scripts', 'gz_teleport_helper')
            if os.path.exists(potential_path):
                helper_path = potential_path
                break
            current_dir = os.path.dirname(current_dir)
            
        if helper_path is None:
            raise FileNotFoundError("Could not find 'scripts/gz_teleport_helper' in any parent directory.")

        self._teleport_proc = subprocess.Popen(
            [helper_path, GZ_WORLD_NAME],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,  # line buffered
        )
        # Wait for READY
        ready = self._teleport_proc.stdout.readline().strip()
        self._node.get_logger().info(f"Teleport helper: {ready}")

        # Warmup: teleport the drone and let the physics engine + velocity
        # controller stabilize.  With MulticopterMotorModel, the first few
        # teleports after Gazebo starts need extra settling time for the
        # rotors to spool up and the PID to converge.
        self._node.get_logger().info("Running warmup teleports...")

        # Arm the multicopter physics plugin
        enable_msg = Bool()
        enable_msg.data = True
        self._enable_pub.publish(enable_msg)

        for i in range(3):
            self._teleport(2.0, 0.0, 1.5, math.pi)
            # Publish zero-velocity commands so the controller actively
            # holds position during warmup (not just coasting/falling)
            for _ in range(20):
                self._publish_action(np.zeros(ACTION_DIM))
                self._odom_event.clear()
                self._odom_event.wait(timeout=0.05)
            # Brief pause between warmup cycles
            time.sleep(0.3)

        self._node.get_logger().info("DroneEnv initialized.")

    # ── Public accessors (for curriculum manager) ──────────────

    @property
    def spawner(self) -> Spawner:
        """Expose the spawner so the curriculum manager can update its stage."""
        return self._spawner

    # ── ROS 2 internals ──────────────────────────────────────────

    def _spin(self):
        rclpy.spin(self._node)

    def _odom_cb(self, msg: Odometry):
        self._latest_odom = msg
        self._odom_event.set()

    def _image_cb(self, msg: RosImage):
        self._latest_image = msg
        self._image_event.set()

    # ── Vision pipeline ──────────────────────────────────────────

    def _process_vision(self) -> tuple[float, float, float, float, float]:
        """Run ArUco detection on the latest camera frame.

        Updates DropoutState internally.  Returns body-frame (x, y, z)
        relative to the marker and normalized pixel coordinates (px, py)
        — live if detected, frozen if in dropout.
        """
        img_msg = self._latest_image
        if img_msg is None:
            self._dropout.on_miss()
            x, y, z, _, px, py = self._dropout.get_pose()
            return x, y, z, px, py

        # Convert ROS Image (RGB8 from Gazebo) → grayscale for ArUco
        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3
        )
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detect marker
        result = detect_marker(
            image=gray,
            marker_size=MARKER_SIZE,
            intrinsics=self._intrinsics,
            marker_id=MARKER_ID,
            dictionary_name=MARKER_DICTIONARY,
        )

        if result is None:
            self._dropout.on_miss()
            x, y, z, _, px, py = self._dropout.get_pose()
            return x, y, z, px, py

        tvec, rvec, pixel_center = result
        body_pos = camera_to_body(tvec)

        # Anti-flip: reject detections implying impossible motion
        if self._marker_prev_pos is not None:
            if is_teleport(self._marker_prev_pos, body_pos):
                self._dropout.on_miss()
                x, y, z, _, px, py = self._dropout.get_pose()
                return x, y, z, px, py

        # Normalize pixel coordinates: (0,0) = frame center, ±1 = frame edge
        px = (pixel_center[0] - SIM_CAMERA_CX) / SIM_CAMERA_CX
        py = (pixel_center[1] - SIM_CAMERA_CY) / SIM_CAMERA_CY

        # Valid detection — update state
        self._marker_prev_pos = body_pos.copy()
        self._dropout.on_detection(
            x=float(body_pos[0]),
            y=float(body_pos[1]),
            z=float(body_pos[2]),
            yaw=0.0,  # yaw comes from odom, not vision
            px=float(px),
            py=float(py),
        )

        return float(body_pos[0]), float(body_pos[1]), float(body_pos[2]), float(px), float(py)

    def _detect_first_frame(self) -> tuple[np.ndarray, float, float] | None:
        """Wait for a camera frame and attempt marker detection.

        Used during reset() for first-frame marker validation (§9).
        Tries up to 5 frames to account for rendering lag.

        Returns
        -------
        (body_pos, px, py) or None
            body_pos : np.ndarray shape (3,) — body-frame [x, y, z].
            px, py : float — normalized pixel coords [-1, 1].
        """
        for _ in range(5):
            self._image_event.clear()
            self._image_event.wait(timeout=0.2)

            img_msg = self._latest_image
            if img_msg is None:
                continue

            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width, 3
            )
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            result = detect_marker(
                image=gray,
                marker_size=MARKER_SIZE,
                intrinsics=self._intrinsics,
                marker_id=MARKER_ID,
                dictionary_name=MARKER_DICTIONARY,
            )
            if result is not None:
                tvec, _, pixel_center = result
                px = (pixel_center[0] - SIM_CAMERA_CX) / SIM_CAMERA_CX
                py = (pixel_center[1] - SIM_CAMERA_CY) / SIM_CAMERA_CY
                return camera_to_body(tvec), float(px), float(py)

        return None

    # ── Gymnasium API ────────────────────────────────────────────

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).copy()
        self._step_count += 1

        # 1. Record for jerk computation, push into obs history
        prev_action = self._prev_action.copy()
        self._obs_builder.push_action(action)

        # 2. Publish and wait for physics
        self._odom_event.clear()
        self._publish_action(action)
        self._odom_event.wait(timeout=0.5)

        # 3. Extract state from odom
        state = self._extract_state()

        # 4. Filter velocity
        vel_filtered = self._ema.update(
            np.array([state["vx"], state["vy"], state["vz"]])
        )

        # 5. Derived quantities
        derived = self._compute_derived(state)

        # Get success thresholds for this curriculum stage
        vz_thresh, vxy_thresh, d_xy_thresh = self._get_success_thresholds()

        # 6. Termination (uses raw velocity for accurate touchdown checks)
        terminated, truncated, outcome, terminal_reward, terminal_breakdown = check_termination(
            x=state["x"], y=state["y"], z=state["z"],
            roll=state["roll"], pitch=state["pitch"],
            vx=state["vx"], vy=state["vy"], vz=state["vz"],
            z_tof=derived["z_tof"],
            d_pad=derived["d_pad"],
            step=self._step_count,
            success_vz_max=vz_thresh,
            success_vxy_max=vxy_thresh,
            success_d_xy_max=d_xy_thresh,
        )

        # Debug: log full state on step-1 crashes to diagnose stale odom
        if self._step_count == 1 and terminated:
            self._node.get_logger().warn(
                f"  STEP-1 CRASH [{outcome}]: "
                f"pos=({state['x']:.2f}, {state['y']:.2f}, {state['z']:.2f}) "
                f"rpy=({math.degrees(state['roll']):.1f}°, "
                f"{math.degrees(state['pitch']):.1f}°, "
                f"{math.degrees(state['yaw']):.1f}°) "
                f"vel=({state['vx']:.2f}, {state['vy']:.2f}, {state['vz']:.2f}) "
                f"z_tof={derived['z_tof']:.3f}"
            )

        # 7. Vision pipeline — get body-frame position + pixel coords
        vis_x, vis_y, vis_z, vis_px, vis_py = self._process_vision()

        # 8. Shaping reward (uses delta trackers + pixel centering)
        shaping_reward, breakdown = compute_reward(
            d_pad=derived["d_pad"],
            d_pad_prev=self._d_pad_prev,
            z=state["z"],
            z_prev=self._z_prev,
            yaw_error=derived["yaw_error"],
            d_marker=derived["d_marker"],
            marker_px=vis_px,
            marker_py=vis_py,
            vx=state["vx"],
            vy=state["vy"],
            vz=state["vz"],
            action=action,
            prev_action=prev_action,
        )

        reward = shaping_reward + terminal_reward

        # 9. Build observation (vision position + pixel coords, odom yaw/velocity/attitude)
        obs = self._obs_builder.build(
            x=vis_x, y=vis_y, z=vis_z,
            yaw=derived["yaw_error"],
            vx=vel_filtered[0], vy=vel_filtered[1], vz=vel_filtered[2],
            z_tof_raw=derived["z_tof_raw"],
            roll=state["roll"], pitch=state["pitch"],
            dropout_timer=self._dropout.timer,
            marker_px=vis_px,
            marker_py=vis_py,
        )

        # Guard: NaN in observations or reward crashes training.
        # Sources: Gazebo physics glitches, solvePnP edge cases, odom
        # corruption.  Replace with safe values rather than crashing.
        if np.isnan(obs).any():
            self._node.get_logger().warn(
                f"NaN in observation at step {self._step_count}, replacing with zeros"
            )
            obs = np.nan_to_num(obs, nan=0.0)
        if math.isnan(reward):
            self._node.get_logger().warn(
                f"NaN reward at step {self._step_count}, replacing with 0"
            )
            reward = 0.0

        # 10. Update trackers for next step
        self._d_pad_prev = derived["d_pad"]
        self._z_prev = state["z"]
        self._prev_action = action.copy()

        # 11. Info dict for logging callback
        info = {
            "outcome": outcome,
            "terminal_reward": terminal_reward,
            **breakdown,
            **terminal_breakdown,
        }
        if terminated or truncated:
            info["episode_outcome"] = outcome
            info["final_d_pad"] = derived["d_pad"]
            info["final_vz"] = state["vz"]
            info["final_vxy"] = math.hypot(state["vx"], state["vy"])

        return obs, reward, terminated, truncated, info

    # Stabilization thresholds — the reset loop waits until ALL of these
    # are satisfied before starting the episode.
    _SETTLE_VZ_MAX = 0.15       # m/s  vertical velocity
    _SETTLE_VXY_MAX = 0.10      # m/s  horizontal velocity
    _SETTLE_ROLL_MAX = math.radians(5)   # rad
    _SETTLE_PITCH_MAX = math.radians(5)  # rad
    _SETTLE_MIN_CLEARANCE = 0.05  # m — reduced from 0.10 for new marker
                                  #     geometry where spawns near marker_z
                                  #     (0.878) are only ~0.128m above desk
    _SETTLE_TIMEOUT = 3.0       # seconds — hard limit to avoid hangs
    _SETTLE_CHECK_HZ = 30       # how often to poll odom during settle

    # Maximum respawn attempts for first-frame marker validation (§9).
    _MAX_RESPAWN_ATTEMPTS = 10

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

        # ── Spawn + stabilize + first-frame marker validation ─────
        # The marker must be detected in the first frame after each
        # reset.  If detection fails, the spawn is discarded and a new
        # one is sampled.  This prevents invalid episodes (agent starts
        # with no visual lock) from polluting the rollout buffer.
        body_pos = None
        initial_px = 0.0
        initial_py = 0.0
        state = None

        for attempt in range(self._MAX_RESPAWN_ATTEMPTS):
            # 1. Sample spawn position
            spawn = self._spawner.sample()

            # 2. Stop the drone, then teleport directly to spawn_z
            self._publish_action(np.zeros(ACTION_DIM))
            self._teleport(spawn["x"], spawn["y"], spawn["z"], spawn["yaw"])

            # 3. Arm Gazebo velocity controller
            self._takeoff_pub.publish(Empty())
            enable_msg = Bool()
            enable_msg.data = True
            self._enable_pub.publish(enable_msg)

            # 4. Wait for the drone to stabilize after teleport
            state = self._wait_for_stable_hover()

            # 5. First-frame marker validation
            detection = self._detect_first_frame()
            if detection is not None:
                body_pos, initial_px, initial_py = detection
                break

            self._node.get_logger().warn(
                f"First-frame marker validation failed "
                f"(attempt {attempt + 1}/{self._MAX_RESPAWN_ATTEMPTS}), "
                f"resampling spawn"
            )

        if body_pos is None:
            # Emergency fallback — should be extremely rare with correct
            # spawn validation.  Use an approximate body-frame position
            # derived from odom so the episode can proceed.
            self._node.get_logger().error(
                "Marker not detected after all respawn attempts; "
                "using odom-derived fallback"
            )
            body_pos = np.array([
                state["x"],
                state["y"],
                state["z"] - MARKER_Z,
            ])
            initial_px = 0.0
            initial_py = 0.0

        # ── Reset sub-modules with vision pose ────────────────────
        self._marker_prev_pos = body_pos.copy()
        vel_initial = np.array([state["vx"], state["vy"], state["vz"]])
        self._ema.reset(vel_initial)
        self._obs_builder.reset()
        self._dropout.reset(
            x=float(body_pos[0]),
            y=float(body_pos[1]),
            z=float(body_pos[2]),
            yaw=0.0,
            px=initial_px,
            py=initial_py,
        )

        # ── Initialize delta trackers (odom-based, for rewards) ───
        derived = self._compute_derived(state)
        self._d_pad_prev = derived["d_pad"]
        self._z_prev = state["z"]

        # ── Build initial observation with vision position ────────
        vel_filtered = self._ema.value
        obs = self._obs_builder.build(
            x=float(body_pos[0]),
            y=float(body_pos[1]),
            z=float(body_pos[2]),
            yaw=derived["yaw_error"],
            vx=vel_filtered[0], vy=vel_filtered[1], vz=vel_filtered[2],
            z_tof_raw=derived["z_tof_raw"],
            roll=state["roll"], pitch=state["pitch"],
            dropout_timer=0,
            marker_px=initial_px,
            marker_py=initial_py,
        )

        if np.isnan(obs).any():
            self._node.get_logger().warn("NaN in initial observation, replacing with zeros")
            obs = np.nan_to_num(obs, nan=0.0)

        return obs, {}

    def close(self):
        self._node.get_logger().info("Shutting down DroneEnv.")
        if self._teleport_proc and self._teleport_proc.poll() is None:
            self._teleport_proc.terminate()
            self._teleport_proc.wait(timeout=2.0)
        self._node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self._spin_thread.join(timeout=2.0)

    # ── State extraction ─────────────────────────────────────────

    def _extract_state(self) -> dict:
        """Pull position, velocity, and attitude from the latest odom msg.

        Returns a dict with keys: x, y, z, vx, vy, vz, roll, pitch, yaw.
        Returns zeros if no odom has been received yet.
        """
        odom = self._latest_odom
        if odom is None:
            return {k: 0.0 for k in
                    ["x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw"]}

        pos = odom.pose.pose.position
        vel = odom.twist.twist.linear
        q = odom.pose.pose.orientation

        roll, pitch, yaw = self._quat_to_euler(q)

        return {
            "x": pos.x, "y": pos.y, "z": pos.z,
            "vx": vel.x, "vy": vel.y, "vz": vel.z,
            "roll": roll, "pitch": pitch, "yaw": yaw,
        }

    def _compute_derived(self, state: dict) -> dict:
        """Delegate to module-level pure function."""
        return compute_derived(state)

    def _get_success_thresholds(self) -> tuple[float, float, float]:
        """Get per-stage success thresholds for the current curriculum stage."""
        stage_cfg = CURRICULUM_STAGES.get(self._spawner.stage, {})
        vz_max = stage_cfg.get("success_vz_max", SUCCESS_VZ_MAX)
        vxy_max = stage_cfg.get("success_vxy_max", SUCCESS_VXY_MAX)
        d_xy_max = stage_cfg.get("success_d_xy_max", SUCCESS_D_XY_MAX)
        return vz_max, vxy_max, d_xy_max

    # ── Post-teleport stabilization ─────────────────────────────

    def _wait_for_stable_hover(self) -> dict:
        """Block until the drone is hovering stably with clearance.

        After teleport the MulticopterVelocityControl PID needs time to
        spool the motors and arrest the gravity-induced drop.  This method
        polls odom at ~30 Hz and returns once vertical velocity, horizontal
        velocity, roll, pitch, AND surface clearance are all within
        thresholds — or after a hard timeout.

        The clearance check is critical: a drone sitting on a surface also
        has zero velocity and level attitude, so without it the settle loop
        would exit immediately after the drone lands on the desk/ground
        during the post-teleport fall.

        Also publishes zero-velocity commands each iteration so the
        velocity controller actively holds position rather than coasting.

        Returns the final settled state dict.
        """
        deadline = time.monotonic() + self._SETTLE_TIMEOUT
        poll_interval = 1.0 / self._SETTLE_CHECK_HZ
        zero_action = np.zeros(ACTION_DIM)

        settled_state = None
        while time.monotonic() < deadline:
            # Command the controller to hold position
            self._publish_action(zero_action)

            # Wait for fresh odom
            self._odom_event.clear()
            self._odom_event.wait(timeout=poll_interval)

            state = self._extract_state()
            vz = abs(state["vz"])
            vxy = math.hypot(state["vx"], state["vy"])
            roll_ok = abs(state["roll"]) < self._SETTLE_ROLL_MAX
            pitch_ok = abs(state["pitch"]) < self._SETTLE_PITCH_MAX

            # Compute clearance above nearest surface (same logic as
            # compute_derived, inlined to avoid import cycle overhead).
            x, y, z = state["x"], state["y"], state["z"]
            if 0.0 <= x <= 0.60 and -0.50 <= y <= 0.50:
                surface_below = PAD_ELEVATION
            else:
                surface_below = 0.0
            clearance = z - surface_below

            has_clearance = clearance > self._SETTLE_MIN_CLEARANCE

            if (vz < self._SETTLE_VZ_MAX
                    and vxy < self._SETTLE_VXY_MAX
                    and roll_ok and pitch_ok
                    and has_clearance):
                settled_state = state
                break

        if settled_state is None:
            # Timed out — use whatever state we have, log a warning
            settled_state = self._extract_state()
            x, y, z = settled_state["x"], settled_state["y"], settled_state["z"]
            if 0.0 <= x <= 0.60 and -0.50 <= y <= 0.50:
                sfc = PAD_ELEVATION
            else:
                sfc = 0.0
            clr = z - sfc
            self._node.get_logger().warn(
                f"Settle timeout: vz={abs(settled_state['vz']):.3f} "
                f"vxy={math.hypot(settled_state['vx'], settled_state['vy']):.3f} "
                f"roll={math.degrees(settled_state['roll']):.1f}° "
                f"pitch={math.degrees(settled_state['pitch']):.1f}° "
                f"clearance={clr:.3f}m"
            )

        # One final odom flush to make sure we have the latest reading
        for _ in range(2):
            self._odom_event.clear()
            self._odom_event.wait(timeout=0.05)
        settled_state = self._extract_state()

        return settled_state

    # ── Gazebo interface ─────────────────────────────────────────

    def _teleport(self, x: float, y: float, z: float, yaw: float):
        """Move the drone via the persistent teleport helper.

        Flushes stale odom after the teleport so subsequent reads reflect
        the new position.  The caller (reset or warmup) is responsible for
        the full stabilization wait — this method only ensures the odom
        pipeline has cleared the pre-teleport readings.
        """
        w = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)

        cmd = f"{GZ_DRONE_MODEL_NAME} {x} {y} {z} {w} 0.0 0.0 {qz}\n"
        self._teleport_proc.stdin.write(cmd)
        self._teleport_proc.stdin.flush()
        # Read response (OK or FAIL — either way teleport worked)
        self._teleport_proc.stdout.readline()

        # Flush stale odom — 3 cycles at 50 ms gives the odom publisher
        # (30 Hz) at least one full cycle to emit a post-teleport reading.
        for _ in range(3):
            self._odom_event.clear()
            self._odom_event.wait(timeout=0.05)

    def _publish_action(self, action: np.ndarray):
        enable_msg = Bool()
        enable_msg.data = True
        self._enable_pub.publish(enable_msg)

        # MulticopterVelocityControl expects world-frame velocity.
        # Action semantics are body-frame (matching real Tello SDK),
        # so rotate linear x/y into world frame before publishing.
        state = self._extract_state()
        yaw = state["yaw"]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        vx_body = float(action[0])
        vy_body = float(action[1])

        msg = Twist()
        msg.linear.x = vx_body * cos_yaw - vy_body * sin_yaw  # world frame
        msg.linear.y = vx_body * sin_yaw + vy_body * cos_yaw  # world frame
        msg.linear.z = float(action[2])   # z is frame-invariant
        msg.angular.z = float(action[3])  # yaw rate is frame-invariant
        self._cmd_pub.publish(msg)
        
    # ── Math helpers (delegate to module-level functions) ────────

    @staticmethod
    def _quat_to_euler(q) -> tuple[float, float, float]:
        return quat_to_euler(q)


# =====================================================================
# Module-level pure functions (testable without ROS 2 or Gazebo)
# =====================================================================

def quat_to_euler(q) -> tuple[float, float, float]:
    """Convert a quaternion (with .x, .y, .z, .w attrs) to (roll, pitch, yaw)."""
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)

    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def compute_derived(state: dict) -> dict:
    """Compute quantities needed by reward and termination.

    Pure function — uses only the state dict and module-level constants.
    Extracted so it can be unit tested without instantiating DroneEnv.

    Parameters
    ----------
    state : dict
        Must contain keys: x, y, z, yaw, roll, pitch.

    Returns
    -------
    dict with keys: d_pad, d_marker, h_above_marker, yaw_error,
                    z_tof_raw, z_tof.
    """
    x, y, z = state["x"], state["y"], state["z"]

    d_pad = math.hypot(x - PAD_WORLD_POS[0], y - PAD_WORLD_POS[1])
    d_marker = math.hypot(x - MARKER_WORLD_POS[0], y - MARKER_WORLD_POS[1])
    h_above_marker = z - MARKER_Z

    bearing = math.atan2(
        MARKER_WORLD_POS[1] - y,
        MARKER_WORLD_POS[0] - x,
    )
    yaw_error = math.atan2(
        math.sin(state["yaw"] - bearing),
        math.cos(state["yaw"] - bearing),
    )

    # Determine surface height below the drone.
    # Desk occupies x=[0, 0.60], y=[-0.50, 0.50], top at PAD_ELEVATION.
    # Everywhere else, the surface is the ground (z=0).
    if 0.0 <= x <= 0.60 and -0.50 <= y <= 0.50:
        surface_below = PAD_ELEVATION
    else:
        surface_below = 0.0

    z_tof_raw = max(z - surface_below, 0.0)
    z_tof = z_tof_raw * math.cos(state["pitch"]) * math.cos(state["roll"])

    return {
        "d_pad": d_pad,
        "d_marker": d_marker,
        "h_above_marker": h_above_marker,
        "yaw_error": yaw_error,
        "z_tof_raw": z_tof_raw,
        "z_tof": z_tof,
    }