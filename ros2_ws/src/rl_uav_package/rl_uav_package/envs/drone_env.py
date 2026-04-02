"""
Gymnasium environment wrapper for Tello RL landing.

Central orchestrator with checkpoint pipeline:
    1. Approach marker
    2. Hover over pad (accumulate dwell) → checkpoint bonus
    3. Post-checkpoint: z-alignment targets pad, descent gradient
    4. Phase C latch: heavy descent + XY hold
    5. Landing: success terminal reward
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
    MARKER_CENTER_Z, PAD_WORLD_POS, MARKER_WORLD_POS,
    PAD_ELEVATION, CURRICULUM_STAGES,
    DESK_X_MIN, DESK_X_MAX, DESK_Y_MIN, DESK_Y_MAX,
    DESCENT_GATE_D_PAD, DESCENT_COMMIT_Z_MARGIN,
    HOVER_Z_MIN_CLEARANCE, HOVER_DWELL_STEPS,
    R_HOVER_CHECKPOINT,
    DROPOUT_GRACE_STEPS,
    SUCCESS_VZ_MAX, SUCCESS_VXY_MAX, SUCCESS_D_XY_MAX,
    Z_CEILING,
    MARKER_SIZE, MARKER_ID, MARKER_DICTIONARY,
    SIM_CAMERA_FX, SIM_CAMERA_FY, SIM_CAMERA_CX, SIM_CAMERA_CY,
    GZ_WORLD_NAME, GZ_DRONE_MODEL_NAME,
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
    metadata = {"render_modes": []}

    def __init__(self, initial_stage: int = 0, seed: int = 0):
        super().__init__()

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH,
            shape=(ACTION_DIM,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(OBS_DIM,), dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._obs_builder = ObservationBuilder()
        self._ema = EMAFilter(n_channels=3)
        self._spawner = Spawner(stage=initial_stage, rng=self._rng)
        self._dropout = DropoutState()

        # Episode state
        self._step_count = 0
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._d_pad_prev = 0.0
        self._z_prev = 0.0

        # Checkpoint pipeline state
        self._hover_checkpoint_reached = False
        self._hover_dwell_count = 0
        self._descent_committed = False
        self._dropout_freeze_logged = False  # log once per dropout event

        # ROS 2
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
            Odometry, "/odom", self._odom_cb, qos)
        self._cmd_pub = self._node.create_publisher(Twist, "/cmd_vel", 10)
        self._takeoff_pub = self._node.create_publisher(Empty, "/takeoff", 10)
        self._land_pub = self._node.create_publisher(Empty, "/land", 10)
        self._enable_pub = self._node.create_publisher(Bool, "/enable", 10)

        self._latest_odom = None
        self._odom_event = threading.Event()

        self._latest_image = None
        self._image_event = threading.Event()
        self._cam_sub = self._node.create_subscription(
            RosImage, "/camera/image_raw", self._image_cb, qos)

        self._intrinsics = CameraIntrinsics(
            fx=SIM_CAMERA_FX, fy=SIM_CAMERA_FY,
            cx=SIM_CAMERA_CX, cy=SIM_CAMERA_CY,
        )
        self._marker_prev_pos = None

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        # Teleport helper
        current_dir = os.path.dirname(os.path.abspath(__file__))
        helper_path = None
        while current_dir != '/':
            potential_path = os.path.join(current_dir, 'scripts', 'gz_teleport_helper')
            if os.path.exists(potential_path):
                helper_path = potential_path
                break
            current_dir = os.path.dirname(current_dir)
        if helper_path is None:
            raise FileNotFoundError(
                "Could not find 'scripts/gz_teleport_helper' in any parent directory.")

        self._teleport_proc = subprocess.Popen(
            [helper_path, GZ_WORLD_NAME],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1)
        ready = self._teleport_proc.stdout.readline().strip()
        self._node.get_logger().info(f"Teleport helper: {ready}")

        # Warmup
        self._node.get_logger().info("Running warmup teleports...")
        enable_msg = Bool()
        enable_msg.data = True
        self._enable_pub.publish(enable_msg)
        for _ in range(3):
            self._teleport(2.0, 0.0, 1.5, math.pi)
            for _ in range(20):
                self._publish_action(np.zeros(ACTION_DIM))
                self._odom_event.clear()
                self._odom_event.wait(timeout=0.05)
            time.sleep(0.3)

        self._node.get_logger().info("DroneEnv initialized.")

    @property
    def spawner(self) -> Spawner:
        return self._spawner

    def _spin(self):
        rclpy.spin(self._node)

    def _odom_cb(self, msg):
        self._latest_odom = msg
        self._odom_event.set()

    def _image_cb(self, msg):
        self._latest_image = msg
        self._image_event.set()

    # ── Vision pipeline ──────────────────────────────────────

    def _process_vision(self):
        img_msg = self._latest_image
        if img_msg is None:
            self._dropout.on_miss()
            x, y, z, _, px, py = self._dropout.get_pose()
            return x, y, z, px, py

        img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
            img_msg.height, img_msg.width, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        result = detect_marker(
            image=gray, marker_size=MARKER_SIZE,
            intrinsics=self._intrinsics, marker_id=MARKER_ID,
            dictionary_name=MARKER_DICTIONARY)

        if result is None:
            self._dropout.on_miss()
            x, y, z, _, px, py = self._dropout.get_pose()
            return x, y, z, px, py

        tvec, rvec, pixel_center = result
        body_pos = camera_to_body(tvec)

        if self._marker_prev_pos is not None:
            if is_teleport(self._marker_prev_pos, body_pos):
                self._dropout.on_miss()
                x, y, z, _, px, py = self._dropout.get_pose()
                return x, y, z, px, py

        px = (pixel_center[0] - SIM_CAMERA_CX) / SIM_CAMERA_CX
        py = (pixel_center[1] - SIM_CAMERA_CY) / SIM_CAMERA_CY

        self._marker_prev_pos = body_pos.copy()
        self._dropout.on_detection(
            x=float(body_pos[0]), y=float(body_pos[1]), z=float(body_pos[2]),
            yaw=0.0, px=float(px), py=float(py))

        # Reset dropout freeze log flag when marker is reacquired
        self._dropout_freeze_logged = False

        return float(body_pos[0]), float(body_pos[1]), float(body_pos[2]), float(px), float(py)

    def _detect_first_frame(self):
        for _ in range(5):
            self._image_event.clear()
            self._image_event.wait(timeout=0.2)
            img_msg = self._latest_image
            if img_msg is None:
                continue
            img = np.frombuffer(img_msg.data, dtype=np.uint8).reshape(
                img_msg.height, img_msg.width, 3)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            result = detect_marker(
                image=gray, marker_size=MARKER_SIZE,
                intrinsics=self._intrinsics, marker_id=MARKER_ID,
                dictionary_name=MARKER_DICTIONARY)
            if result is not None:
                tvec, _, pixel_center = result
                px = (pixel_center[0] - SIM_CAMERA_CX) / SIM_CAMERA_CX
                py = (pixel_center[1] - SIM_CAMERA_CY) / SIM_CAMERA_CY
                return camera_to_body(tvec), float(px), float(py)
        return None

    # ── Stage config ─────────────────────────────────────────

    def _get_stage_config(self):
        return CURRICULUM_STAGES.get(self._spawner.stage, CURRICULUM_STAGES[0])

    def _get_success_thresholds(self):
        cfg = self._get_stage_config()
        return (cfg.get("success_vz_max", SUCCESS_VZ_MAX),
                cfg.get("success_vxy_max", SUCCESS_VXY_MAX),
                cfg.get("success_d_xy_max", SUCCESS_D_XY_MAX))

    def _get_hover_thresholds(self):
        cfg = self._get_stage_config()
        return (cfg.get("hover_d_pad_max", 0.25),
                cfg.get("hover_vxy_max", 0.4),
                cfg.get("hover_dwell_steps", HOVER_DWELL_STEPS))

    def _get_max_steps(self):
        return self._get_stage_config().get("max_steps", MAX_STEPS)

    # ── Gymnasium API ────────────────────────────────────────

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).copy()
        self._step_count += 1

        prev_action = self._prev_action.copy()
        self._obs_builder.push_action(action)

        self._odom_event.clear()
        self._publish_action(action)
        self._odom_event.wait(timeout=0.5)

        state = self._extract_state()
        vel_filtered = self._ema.update(
            np.array([state["vx"], state["vy"], state["vz"]]))
        derived = self._compute_derived(state)

        # ── Checkpoint pipeline ──────────────────────────────
        checkpoint_bonus = 0.0

        # Hover dwell tracking
        hover_d_pad_max, hover_vxy_max, hover_dwell_target = self._get_hover_thresholds()
        v_xy = math.hypot(state["vx"], state["vy"])
        hover_ok = (
            derived["d_pad"] < hover_d_pad_max
            and state["z"] > PAD_ELEVATION + HOVER_Z_MIN_CLEARANCE
            and v_xy < hover_vxy_max
            and self._dropout.timer == 0
        )
        if hover_ok:
            self._hover_dwell_count += 1
        else:
            self._hover_dwell_count = 0

        # Checkpoint: one-time bonus
        if (not self._hover_checkpoint_reached
                and self._hover_dwell_count >= hover_dwell_target):
            self._hover_checkpoint_reached = True
            checkpoint_bonus = R_HOVER_CHECKPOINT

        # Descent latch: only after checkpoint
        descent_z_threshold = MARKER_CENTER_Z - DESCENT_COMMIT_Z_MARGIN
        if (self._hover_checkpoint_reached
                and not self._descent_committed
                and derived["d_pad"] < DESCENT_GATE_D_PAD
                and state["z"] < descent_z_threshold):
            self._descent_committed = True

        # ── Dropout freeze logging (once per dropout event) ──
        dropout_freeze_active = (
            self._dropout.timer >= DROPOUT_GRACE_STEPS
            and not self._hover_checkpoint_reached
            and not self._descent_committed
        )
        # if dropout_freeze_active and not self._dropout_freeze_logged:
        #     self._dropout_freeze_logged = True
        #     self._node.get_logger().warn(
        #         f"[DROPOUT FREEZE] Marker lost at step {self._step_count}, "
        #         f"dropout_timer={self._dropout.timer}, "
        #         f"pos=({state['x']:.2f}, {state['y']:.2f}, {state['z']:.2f})")

        # ── Termination ──────────────────────────────────────
        vz_thresh, vxy_thresh, d_xy_thresh = self._get_success_thresholds()
        max_steps = self._get_max_steps()

        terminated, truncated, outcome, terminal_reward, terminal_breakdown = check_termination(
            x=state["x"], y=state["y"], z=state["z"],
            roll=state["roll"], pitch=state["pitch"],
            vx=state["vx"], vy=state["vy"], vz=state["vz"],
            z_tof=derived["z_tof"], d_pad=derived["d_pad"],
            step=self._step_count,
            success_vz_max=vz_thresh, success_vxy_max=vxy_thresh,
            success_d_xy_max=d_xy_thresh, max_steps=max_steps,
            hover_checkpoint_reached=self._hover_checkpoint_reached)

        if self._step_count == 1 and terminated:
            self._node.get_logger().warn(
                f"  STEP-1 CRASH [{outcome}]: "
                f"pos=({state['x']:.2f}, {state['y']:.2f}, {state['z']:.2f}) "
                f"vel=({state['vx']:.2f}, {state['vy']:.2f}, {state['vz']:.2f})")

        # ── Vision ───────────────────────────────────────────
        vis_x, vis_y, vis_z, vis_px, vis_py = self._process_vision()

        # ── Reward ───────────────────────────────────────────
        shaping_reward, breakdown = compute_reward(
            d_pad=derived["d_pad"], d_pad_prev=self._d_pad_prev,
            z=state["z"], z_prev=self._z_prev,
            yaw_error=derived["yaw_error"], d_marker=derived["d_marker"],
            marker_px=vis_px, marker_py=vis_py,
            vx=state["vx"], vy=state["vy"], vz=state["vz"],
            action=action, prev_action=prev_action,
            marker_center_z=MARKER_CENTER_Z,
            descent_committed=self._descent_committed,
            hover_checkpoint_reached=self._hover_checkpoint_reached,
            dropout_timer=self._dropout.timer,
        )

        reward = shaping_reward + checkpoint_bonus + terminal_reward

        # ── Observation ──────────────────────────────────────
        obs = self._obs_builder.build(
            x=vis_x, y=vis_y, z=vis_z,
            yaw=derived["yaw_error"],
            vx=vel_filtered[0], vy=vel_filtered[1], vz=vel_filtered[2],
            z_tof_raw=derived["z_tof_raw"],
            roll=state["roll"], pitch=state["pitch"],
            dropout_timer=self._dropout.timer,
            marker_px=vis_px, marker_py=vis_py)

        if np.isnan(obs).any():
            obs = np.nan_to_num(obs, nan=0.0)
        if math.isnan(reward):
            reward = 0.0

        # ── Update trackers ──────────────────────────────────
        self._d_pad_prev = derived["d_pad"]
        self._z_prev = state["z"]
        self._prev_action = action.copy()

        # ── Info ─────────────────────────────────────────────
        info = {
            "outcome": outcome,
            "terminal_reward": terminal_reward,
            "checkpoint_bonus": checkpoint_bonus,
            "hover_checkpoint_reached": self._hover_checkpoint_reached,
            "descent_committed": self._descent_committed,
            "hover_dwell_count": self._hover_dwell_count,
            "phase": breakdown.get("diag/phase", 0),
            **breakdown,
            **terminal_breakdown,
        }
        if terminated or truncated:
            info["episode_outcome"] = outcome
            info["final_d_pad"] = derived["d_pad"]
            info["final_vz"] = state["vz"]
            info["final_vxy"] = math.hypot(state["vx"], state["vy"])

        return obs, reward, terminated, truncated, info

    # ── Reset ────────────────────────────────────────────────

    _SETTLE_VZ_MAX = 0.15
    _SETTLE_VXY_MAX = 0.10
    _SETTLE_ROLL_MAX = math.radians(5)
    _SETTLE_PITCH_MAX = math.radians(5)
    _SETTLE_MIN_CLEARANCE = 0.05
    _SETTLE_TIMEOUT = 3.0
    _SETTLE_CHECK_HZ = 30
    _MAX_RESPAWN_ATTEMPTS = 10

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._hover_checkpoint_reached = False
        self._hover_dwell_count = 0
        self._descent_committed = False
        self._dropout_freeze_logged = False

        body_pos = None
        initial_px = 0.0
        initial_py = 0.0
        state = None

        for attempt in range(self._MAX_RESPAWN_ATTEMPTS):
            spawn = self._spawner.sample()
            self._publish_action(np.zeros(ACTION_DIM))
            self._teleport(spawn["x"], spawn["y"], spawn["z"], spawn["yaw"])
            self._takeoff_pub.publish(Empty())
            enable_msg = Bool()
            enable_msg.data = True
            self._enable_pub.publish(enable_msg)
            state = self._wait_for_stable_hover()
            detection = self._detect_first_frame()
            if detection is not None:
                body_pos, initial_px, initial_py = detection
                break
            self._node.get_logger().warn(
                f"First-frame validation failed (attempt {attempt + 1})")

        if body_pos is None:
            self._node.get_logger().error("Marker not detected; using odom fallback")
            body_pos = np.array([
                state["x"] - MARKER_WORLD_POS[0],
                state["y"] - MARKER_WORLD_POS[1],
                state["z"] - MARKER_CENTER_Z])
            initial_px = 0.0
            initial_py = 0.0

        self._marker_prev_pos = body_pos.copy()
        self._ema.reset(np.array([state["vx"], state["vy"], state["vz"]]))
        self._obs_builder.reset()
        self._dropout.reset(
            x=float(body_pos[0]), y=float(body_pos[1]), z=float(body_pos[2]),
            yaw=0.0, px=initial_px, py=initial_py)

        derived = self._compute_derived(state)
        self._d_pad_prev = derived["d_pad"]
        self._z_prev = state["z"]

        vel_filtered = self._ema.value
        obs = self._obs_builder.build(
            x=float(body_pos[0]), y=float(body_pos[1]), z=float(body_pos[2]),
            yaw=derived["yaw_error"],
            vx=vel_filtered[0], vy=vel_filtered[1], vz=vel_filtered[2],
            z_tof_raw=derived["z_tof_raw"],
            roll=state["roll"], pitch=state["pitch"],
            dropout_timer=0, marker_px=initial_px, marker_py=initial_py)

        if np.isnan(obs).any():
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

    # ── State extraction ─────────────────────────────────────

    def _extract_state(self):
        odom = self._latest_odom
        if odom is None:
            return {k: 0.0 for k in
                    ["x", "y", "z", "vx", "vy", "vz", "roll", "pitch", "yaw"]}
        pos = odom.pose.pose.position
        vel = odom.twist.twist.linear
        q = odom.pose.pose.orientation
        roll, pitch, yaw = quat_to_euler(q)
        return {"x": pos.x, "y": pos.y, "z": pos.z,
                "vx": vel.x, "vy": vel.y, "vz": vel.z,
                "roll": roll, "pitch": pitch, "yaw": yaw}

    def _compute_derived(self, state):
        return compute_derived(state)

    def _wait_for_stable_hover(self):
        deadline = time.monotonic() + self._SETTLE_TIMEOUT
        poll_interval = 1.0 / self._SETTLE_CHECK_HZ
        zero_action = np.zeros(ACTION_DIM)
        settled_state = None

        while time.monotonic() < deadline:
            self._publish_action(zero_action)
            self._odom_event.clear()
            self._odom_event.wait(timeout=poll_interval)
            state = self._extract_state()
            x, y, z = state["x"], state["y"], state["z"]
            if (DESK_X_MIN <= x <= DESK_X_MAX and DESK_Y_MIN <= y <= DESK_Y_MAX):
                surface = PAD_ELEVATION
            else:
                surface = 0.0
            clearance = z - surface

            if (abs(state["vz"]) < self._SETTLE_VZ_MAX
                    and math.hypot(state["vx"], state["vy"]) < self._SETTLE_VXY_MAX
                    and abs(state["roll"]) < self._SETTLE_ROLL_MAX
                    and abs(state["pitch"]) < self._SETTLE_PITCH_MAX
                    and clearance > self._SETTLE_MIN_CLEARANCE):
                settled_state = state
                break

        if settled_state is None:
            settled_state = self._extract_state()

        for _ in range(2):
            self._odom_event.clear()
            self._odom_event.wait(timeout=0.05)
        return self._extract_state()

    # ── Gazebo interface ─────────────────────────────────────

    def _teleport(self, x, y, z, yaw):
        w = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        cmd = f"{GZ_DRONE_MODEL_NAME} {x} {y} {z} {w} 0.0 0.0 {qz}\n"
        self._teleport_proc.stdin.write(cmd)
        self._teleport_proc.stdin.flush()
        self._teleport_proc.stdout.readline()
        for _ in range(3):
            self._odom_event.clear()
            self._odom_event.wait(timeout=0.05)

    def _publish_action(self, action):
        enable_msg = Bool()
        enable_msg.data = True
        self._enable_pub.publish(enable_msg)
        state = self._extract_state()
        yaw = state["yaw"]
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        vx_body, vy_body = float(action[0]), float(action[1])
        msg = Twist()
        msg.linear.x = vx_body * cos_yaw - vy_body * sin_yaw
        msg.linear.y = vx_body * sin_yaw + vy_body * cos_yaw
        msg.linear.z = float(action[2])
        msg.angular.z = float(action[3])
        self._cmd_pub.publish(msg)


# =====================================================================
# Module-level pure functions
# =====================================================================

def quat_to_euler(q):
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def compute_derived(state):
    x, y, z = state["x"], state["y"], state["z"]
    d_pad = math.hypot(x - PAD_WORLD_POS[0], y - PAD_WORLD_POS[1])
    d_marker = math.hypot(x - MARKER_WORLD_POS[0], y - MARKER_WORLD_POS[1])
    bearing = math.atan2(MARKER_WORLD_POS[1] - y, MARKER_WORLD_POS[0] - x)
    yaw_error = math.atan2(
        math.sin(state["yaw"] - bearing),
        math.cos(state["yaw"] - bearing))

    if (DESK_X_MIN <= x <= DESK_X_MAX and DESK_Y_MIN <= y <= DESK_Y_MAX):
        surface_below = PAD_ELEVATION
    else:
        surface_below = 0.0

    z_tof_raw = max(z - surface_below, 0.0)
    z_tof = z_tof_raw * math.cos(state["pitch"]) * math.cos(state["roll"])

    return {"d_pad": d_pad, "d_marker": d_marker,
            "h_above_marker": z - MARKER_CENTER_Z,
            "yaw_error": yaw_error,
            "z_tof_raw": z_tof_raw, "z_tof": z_tof}