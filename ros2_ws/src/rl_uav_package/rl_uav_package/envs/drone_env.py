"""
Gymnasium environment wrapper for Tello RL landing.

This is the central orchestrator.  It owns the ROS 2 node lifecycle, the
Gymnasium step/reset loop, and Gazebo teleportation.  All computation is
delegated to specialist modules:

    rewards.py        → per-step shaping reward
    termination.py    → episode-ending conditions
    observations.py   → 29-dim normalized vector assembly
    spawner.py        → rejection-sampled spawn positions
    ema.py            → velocity smoothing
    aruco_tracker.py  → dropout state (vision integration later)

For the baseline (no domain randomization, no live vision), position comes
from odom ground truth and z_tof is approximated as drone altitude above
the ground plane.
"""

import math
import subprocess
import threading
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

from rl_uav_package.config.constants import (
    OBS_DIM, ACTION_DIM, ACTION_LOW, ACTION_HIGH,
    MAX_STEPS, CONTROL_RATE_HZ,
    MARKER_Z, PAD_WORLD_POS, MARKER_WORLD_POS,
    LANDING_PAD_FORWARD_OFFSET,
    GZ_WORLD_NAME, GZ_DRONE_MODEL_NAME,
)
from rl_uav_package.envs.rewards import compute_reward
from rl_uav_package.envs.termination import check_termination
from rl_uav_package.envs.observations import ObservationBuilder
from rl_uav_package.envs.spawner import Spawner
from rl_uav_package.filters.ema import EMAFilter
from rl_uav_package.vision.aruco_tracker import DropoutState


class DroneEnv(gym.Env):
    """Gymnasium environment for autonomous UAV precision landing.

    Parameters
    ----------
    initial_stage : int
        Curriculum stage to start in (1, 2, or 3).
    seed : int, optional
        Random seed for reproducibility.
    """

    metadata = {"render_modes": []}

    def __init__(self, initial_stage: int = 1, seed: int = 0):
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

        # Odom synchronization
        self._latest_odom = None
        self._odom_event = threading.Event()

        # Spin ROS 2 in background
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

        # Warmup: teleport the drone and let the physics engine + velocity
        # controller stabilize.  The first few teleports after Gazebo starts
        # can produce transient bad states (residual velocity, stale odom).
        # Running a few warmup cycles here burns through those before
        # training begins.
        self._node.get_logger().info("Running warmup teleports...")
        for _ in range(3):
            self._teleport(2.0, 0.0, 1.5, math.pi)
            time.sleep(0.5)
            self._odom_event.clear()
            self._odom_event.wait(timeout=1.0)

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

        # 6. Termination (uses raw velocity for accurate touchdown checks)
        terminated, truncated, outcome, terminal_reward = check_termination(
            x=state["x"], y=state["y"], z=state["z"],
            roll=state["roll"], pitch=state["pitch"],
            vx=state["vx"], vy=state["vy"], vz=state["vz"],
            z_tof=derived["z_tof"],
            d_pad=derived["d_pad"],
            yaw_error=derived["yaw_error"],
            step=self._step_count,
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

        # 7. Shaping reward (uses delta trackers)
        shaping_reward, breakdown = compute_reward(
            d_pad=derived["d_pad"],
            d_pad_prev=self._d_pad_prev,
            z=state["z"],
            z_prev=self._z_prev,
            h_above_marker=derived["h_above_marker"],
            yaw_error=derived["yaw_error"],
            d_marker=derived["d_marker"],
            action=action,
            prev_action=prev_action,
        )

        reward = shaping_reward + terminal_reward

        # 8. Build observation (uses filtered velocity)
        obs = self._obs_builder.build(
            x=state["x"], y=state["y"], z=state["z"],
            yaw=state["yaw"],
            vx=vel_filtered[0], vy=vel_filtered[1], vz=vel_filtered[2],
            z_tof_raw=derived["z_tof_raw"],
            roll=state["roll"], pitch=state["pitch"],
            dropout_timer=self._dropout.timer,
        )

        # 9. Update trackers for next step
        self._d_pad_prev = derived["d_pad"]
        self._z_prev = state["z"]
        self._prev_action = action.copy()

        # 10. Info dict for logging callback
        info = {
            "outcome": outcome,
            "terminal_reward": terminal_reward,
            **breakdown,
        }
        if terminated or truncated:
            info["episode_outcome"] = outcome
            info["final_d_pad"] = derived["d_pad"]
            info["final_vz"] = state["vz"]
            info["final_vxy"] = math.hypot(state["vx"], state["vy"])

        return obs, reward, terminated, truncated, info

    # Altitude buffer added to teleport target.  After teleport, the
    # X4 model falls briefly under gravity before the velocity controller
    # engages.  The buffer ensures the drone settles near the spawn height
    # rather than below PAD_ELEVATION.
    _SPAWN_Z_BUFFER = 0.0  # m — tuning knob if needed

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

        # 1. Sample spawn position
        spawn = self._spawner.sample()

        self._node.get_logger().info(
            f"RESET: x={spawn['x']:.2f} y={spawn['y']:.2f} "
            f"z={spawn['z']:.2f} yaw={math.degrees(spawn['yaw']):.1f}°"
        )

        # 2. Stop the drone, then teleport (with built-in retry + verification)
        self._publish_action(np.zeros(ACTION_DIM))
        time.sleep(0.1)
        teleport_z = spawn["z"] + self._SPAWN_Z_BUFFER
        self._teleport(spawn["x"], spawn["y"], teleport_z, spawn["yaw"])

        # 3. Publish takeoff (no-op in sim, primes the real Tello later)
        self._takeoff_pub.publish(Empty())

        # 4. Extract settled state
        state = self._extract_state()
        self._node.get_logger().info(
            f"  Settled: pos=({state['x']:.2f}, {state['y']:.2f}, {state['z']:.2f}) "
            f"rpy=({math.degrees(state['roll']):.1f}°, "
            f"{math.degrees(state['pitch']):.1f}°, "
            f"{math.degrees(state['yaw']):.1f}°)"
        )

        # 5. Reset sub-modules
        vel_initial = np.array([state["vx"], state["vy"], state["vz"]])
        self._ema.reset(vel_initial)
        self._obs_builder.reset()
        self._dropout.reset(
            x=state["x"], y=state["y"], z=state["z"], yaw=state["yaw"],
        )

        # 6. Initialize delta trackers with actual settled values.
        derived = self._compute_derived(state)
        self._d_pad_prev = derived["d_pad"]
        self._z_prev = state["z"]

        # 7. Build initial observation
        vel_filtered = self._ema.value
        obs = self._obs_builder.build(
            x=state["x"], y=state["y"], z=state["z"],
            yaw=state["yaw"],
            vx=vel_filtered[0], vy=vel_filtered[1], vz=vel_filtered[2],
            z_tof_raw=derived["z_tof_raw"],
            roll=state["roll"], pitch=state["pitch"],
            dropout_timer=0,
        )

        return obs, {}

    def close(self):
        self._node.get_logger().info("Shutting down DroneEnv.")
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

    # ── Gazebo interface ─────────────────────────────────────────

    def _teleport(self, x: float, y: float, z: float, yaw: float):
        """Move the drone to a new pose via Gazebo service, with retry.

        Verifies the teleport worked by checking odom after each attempt.
        Retries up to 5 times if the drone didn't reach the target.
        """
        w = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)

        req = (
            f'name: "{GZ_DRONE_MODEL_NAME}", '
            f'position: {{x: {x}, y: {y}, z: {z}}}, '
            f'orientation: {{w: {w}, x: 0.0, y: 0.0, z: {qz}}}'
        )

        cmd = [
            "gz", "service",
            "-s", f"/world/{GZ_WORLD_NAME}/set_pose",
            "--reqtype", "gz.msgs.Pose",
            "--reptype", "gz.msgs.Boolean",
            "--timeout", "2000",
            "--req", req,
        ]

        for attempt in range(5):
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self._node.get_logger().warn(
                    f"  Teleport service call failed (attempt {attempt + 1}): "
                    f"{result.stderr.strip()}"
                )

            # Wait for physics to process the teleport
            time.sleep(0.2)

            # Flush stale odom and read fresh position
            for _ in range(3):
                self._odom_event.clear()
                self._odom_event.wait(timeout=0.2)

            state = self._extract_state()
            pos_error = math.sqrt(
                (state["x"] - x) ** 2 +
                (state["y"] - y) ** 2 +
                (state["z"] - z) ** 2
            )

            if pos_error < 0.3:
                return  # success

            self._node.get_logger().warn(
                f"  Teleport verify failed (attempt {attempt + 1}): "
                f"target=({x:.2f}, {y:.2f}, {z:.2f}), "
                f"actual=({state['x']:.2f}, {state['y']:.2f}, {state['z']:.2f}), "
                f"error={pos_error:.2f}m"
            )

        self._node.get_logger().error(
            f"  Teleport failed after 5 attempts! Continuing with current pose."
        )

    def _publish_action(self, action: np.ndarray):
        """Send velocity command to the drone."""
        msg = Twist()
        msg.linear.x = float(action[0])   # forward
        msg.linear.y = float(action[1])   # left
        msg.linear.z = float(action[2])   # up
        msg.angular.z = float(action[3])  # yaw rate
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

    z_tof_raw = z
    z_tof = z_tof_raw * math.cos(state["pitch"]) * math.cos(state["roll"])

    return {
        "d_pad": d_pad,
        "d_marker": d_marker,
        "h_above_marker": h_above_marker,
        "yaw_error": yaw_error,
        "z_tof_raw": z_tof_raw,
        "z_tof": z_tof,
    }