"""
Gymnasium environment wrapper for Tello RL landing.

This is the central orchestrator.  It owns the ROS 2 node lifecycle, the
Gymnasium step/reset loop, and Gazebo teleportation.  All computation is
delegated to specialist modules:

    rewards.py        -> per-step shaping reward
    termination.py    -> episode-ending conditions
    observations.py   -> 29-dim normalized vector assembly
    spawner.py        -> rejection-sampled spawn positions
    ema.py            -> velocity smoothing
    aruco_tracker.py  -> dropout state (vision integration later)

For the baseline (no domain randomization, no live vision), position comes
from odom ground truth and z_tof is approximated as drone altitude above
the ground plane.
"""

import math
import subprocess
import threading
import time
import os

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, Bool

from rl_uav_package.config.constants import (
    OBS_DIM, ACTION_DIM, ACTION_LOW, ACTION_HIGH,
    MAX_STEPS, CONTROL_RATE_HZ,
    MARKER_Z, PAD_WORLD_POS, MARKER_WORLD_POS,
    LANDING_PAD_FORWARD_OFFSET,
    GZ_WORLD_NAME, GZ_DRONE_MODEL_NAME,
    PAD_ELEVATION, CURRICULUM_STAGES,
    SUCCESS_VZ_MAX, SUCCESS_VXY_MAX,
    SUCCESS_D_XY_MAX, SUCCESS_YAW_ERROR_MAX,
    Z_MAX,
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
        vz_thresh, vxy_thresh, d_xy_thresh, yaw_thresh = self._get_success_thresholds()

        # 6. Termination (uses raw velocity for accurate touchdown checks)
        terminated, truncated, outcome, terminal_reward, terminal_breakdown = check_termination(
            x=state["x"], y=state["y"], z=state["z"],
            roll=state["roll"], pitch=state["pitch"],
            vx=state["vx"], vy=state["vy"], vz=state["vz"],
            z_tof=derived["z_tof"],
            d_pad=derived["d_pad"],
            yaw_error=derived["yaw_error"],
            step=self._step_count,
            success_vz_max=vz_thresh,
            success_vxy_max=vxy_thresh,
            success_d_xy_max=d_xy_thresh,
            success_yaw_error_max=yaw_thresh,
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
            **terminal_breakdown,
        }
        if terminated or truncated:
            info["episode_outcome"] = outcome
            info["final_d_pad"] = derived["d_pad"]
            info["final_vz"] = state["vz"]
            info["final_vxy"] = math.hypot(state["vx"], state["vy"])
            info["final_yaw_error"] = abs(derived["yaw_error"])

        return obs, reward, terminated, truncated, info

    # Altitude buffer added to teleport target.  After teleport, the
    # drone freefalls until the MulticopterVelocityControl PID spools
    # the motors and generates hover thrust.  With corrected drag
    # (8e-06) the controller arrests the fall quickly.  0.35 m is enough
    # for spool-up while keeping the drone close to target altitude so
    # it doesn't have excessive height to descend through.
    _SPAWN_Z_BUFFER = 0.35  # m — cushion for motor spool-up

    # Stabilization thresholds — the reset loop waits until ALL of these
    # are satisfied before starting the episode.
    _SETTLE_VZ_MAX = 0.15       # m/s  vertical velocity
    _SETTLE_VXY_MAX = 0.10      # m/s  horizontal velocity
    _SETTLE_ROLL_MAX = math.radians(5)   # rad
    _SETTLE_PITCH_MAX = math.radians(5)  # rad
    _SETTLE_MIN_CLEARANCE = 0.10  # m — z_tof must exceed this
    _SETTLE_TIMEOUT = 3.0       # seconds — hard limit to avoid hangs
    _SETTLE_CHECK_HZ = 30       # how often to poll odom during settle

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._prev_action = np.zeros(ACTION_DIM, dtype=np.float32)

        # 1. Sample spawn position
        spawn = self._spawner.sample()

        """self._node.get_logger().info(
            f"RESET: x={spawn['x']:.2f} y={spawn['y']:.2f} "
            f"z={spawn['z']:.2f} yaw={math.degrees(spawn['yaw']):.1f}°"
        )"""

        # 2. Stop the drone, then teleport (with built-in retry + verification)
        self._publish_action(np.zeros(ACTION_DIM))
        teleport_z = min(spawn["z"] + self._SPAWN_Z_BUFFER, Z_MAX - 0.10)
        self._teleport(spawn["x"], spawn["y"], teleport_z, spawn["yaw"])

        # 3. Publish takeoff (no-op in sim, primes the real Tello later)
        self._takeoff_pub.publish(Empty())

        # Arm Gazebo velocity controller
        enable_msg = Bool()
        enable_msg.data = True
        self._enable_pub.publish(enable_msg)

        # 4. Wait for the drone to stabilize after teleport.
        #    With MulticopterMotorModel, the drone falls under gravity
        #    until the PID controller spools the motors.  We must wait
        #    for attitude and velocity to settle before starting the
        #    episode, otherwise the agent sees a falling/tumbling drone
        #    and learns that the pad is a death zone.
        state = self._wait_for_stable_hover()
        """self._node.get_logger().info(
            f"  Settled: pos=({state['x']:.2f}, {state['y']:.2f}, {state['z']:.2f}) "
            f"rpy=({math.degrees(state['roll']):.1f}°, "
            f"{math.degrees(state['pitch']):.1f}°, "
            f"{math.degrees(state['yaw']):.1f}°)"
        )"""

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

    def _get_success_thresholds(self) -> tuple[float, float, float, float]:
        """Get per-stage success thresholds for the current curriculum stage."""
        stage_cfg = CURRICULUM_STAGES.get(self._spawner.stage, {})
        vz_max = stage_cfg.get("success_vz_max", SUCCESS_VZ_MAX)
        vxy_max = stage_cfg.get("success_vxy_max", SUCCESS_VXY_MAX)
        d_xy_max = stage_cfg.get("success_d_xy_max", SUCCESS_D_XY_MAX)
        yaw_max = stage_cfg.get("success_yaw_error_max", SUCCESS_YAW_ERROR_MAX)
        return vz_max, vxy_max, d_xy_max, yaw_max

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