import gymnasium as gym
from gymnasium import spaces
import numpy as np
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math
import time

# PX4 Messages
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleOdometry
from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import VehicleCommand

class DroneEnv(gym.Env):
    """
    Gymnasium environment wrapper for PX4 SITL via ROS 2.
    """
    def __init__(self):
        super(DroneEnv, self).__init__()
        
        # 1. Initialize ROS 2 in the main thread
        if not rclpy.ok():
            rclpy.init()
            
        self.node = rclpy.create_node('rl_drone_env_node')
        print("[DEBUG] ROS 2 Node 'rl_drone_env_node' initialized.")
        
        # 2. Define Spaces (Phase 1: Ground Truth)
        # Action: [vx, vy, vz, yaw_rate] in BODY FRAME (m/s and rad/s)
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(4,), dtype=np.float32)
        
        # Observation: [x, y, z, vx, vy, vz, roll, pitch, yaw, pad_rel_fwd, pad_rel_right, pad_rel_down]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        # 3. ROS 2 Publishers & Subscribers
        # PX4 heavily relies on BEST_EFFORT QoS. Applying this to publishers as well.
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.odom_sub = self.node.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self._odom_callback, qos_profile)
            
        self.trajectory_pub = self.node.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
            
        self.offboard_mode_pub = self.node.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
            
        self.vehicle_command_pub = self.node.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # 4. Synchronization & Tracking Variables
        self.state_event = threading.Event()
        self.latest_odom = None
        self.current_action = np.zeros(4, dtype=np.float32)
        self.previous_action = np.zeros(4, dtype=np.float32)
        
        self.max_steps = 300  # 30 seconds at 10Hz
        self.current_step = 0
        self.odom_receive_count = 0
        
        # 5. Start ROS 2 Spin in a Background Thread
        self.executor_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.executor_thread.start()

        # 6. PX4 Offboard Heartbeat Timer
        self.heartbeat_timer = self.node.create_timer(0.1, self._publish_heartbeat)
        print("[DEBUG] Environment setup complete. Waiting for first reset().")

    def _spin_ros(self):
        """Continuously processes ROS 2 callbacks in the background."""
        rclpy.spin(self.node)

    def _odom_callback(self, msg):
        """Triggered asynchronously whenever PX4 publishes new odometry."""
        self.latest_odom = msg
        self.odom_receive_count += 1
        self.state_event.set()

    def _publish_heartbeat(self):
        """Published at 10Hz to keep PX4 in Offboard mode."""
        offboard_msg = OffboardControlMode()
        offboard_msg.position = False
        offboard_msg.velocity = True
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        self.offboard_mode_pub.publish(offboard_msg)
        
        self._publish_action(self.current_action)

    def _send_vehicle_command(self, command, param1=0.0, param2=0.0):
        """Sends MAVLink commands directly to PX4."""
        print(f"[DEBUG] Sending VehicleCommand: {command} (p1:{param1}, p2:{param2})")
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_pub.publish(msg)

    def _get_euler_from_quaternion(self, q):
        """Converts FRD quaternion [w, x, y, z] to roll, pitch, yaw in radians."""
        sinr_cosp = 2.0 * (q[0] * q[1] + q[2] * q[3])
        cosr_cosp = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q[0] * q[2] - q[3] * q[1])
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def _publish_action(self, action):
        """Translates the RL agent's body-frame action to PX4's NED frame message."""
        if self.latest_odom is None:
            return
            
        vx_body, vy_body, vz_body, yaw_rate = action
        _, _, yaw = self._get_euler_from_quaternion(self.latest_odom.q)
        
        v_north = vx_body * math.cos(yaw) - vy_body * math.sin(yaw)
        v_east  = vx_body * math.sin(yaw) + vy_body * math.cos(yaw)
        
        msg = TrajectorySetpoint()
        msg.velocity[0] = float(v_north)
        msg.velocity[1] = float(v_east)
        msg.velocity[2] = float(vz_body)
        msg.yawspeed = float(yaw_rate)
        msg.timestamp = int(self.node.get_clock().now().nanoseconds / 1000)
        
        self.trajectory_pub.publish(msg)

    def step(self, action):
        """Synchronous step function expected by Stable-Baselines3."""
        self.previous_action = np.copy(self.current_action)
        self.current_action = action
        self.current_step += 1
        
        self.state_event.clear()
        self._publish_action(action)
        
        msg_received = self.state_event.wait(timeout=0.5) 
        if not msg_received:
            print("[DEBUG] WARNING: Timeout waiting for odometry data in step()!")
        
        obs = self._get_obs()
        reward, terminated = self._compute_reward_and_termination()
        
        truncated = False
        if self.current_step >= self.max_steps and not terminated:
            truncated = True
            
        # Periodically log flight status to prove it is or isn't moving
        if self.current_step % 50 == 0:
            z_alt = self.latest_odom.position[2] if self.latest_odom else 0.0
            print(f"[DEBUG] Step {self.current_step:03d} | Z-Alt (NED): {z_alt:.3f} | Odom Msgs Received: {self.odom_receive_count}")
            
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("\n[DEBUG] --- RESET CALLED ---")
        self.current_step = 0
        
        # 1. Command an aggressive upward velocity (-2.0 m/s Z)
        self.current_action = np.array([0.0, 0.0, -2.0, 0.0], dtype=np.float32)
        self.previous_action = np.copy(self.current_action)
        
        # 2. Warmup Delay
        print(f"[DEBUG] Warming up DDS connection for 1.5s. Odoms received so far: {self.odom_receive_count}")
        time.sleep(1.5) 
        
        # 3. Arm and Offboard
        print("[DEBUG] Firing ARM command (400)...")
        self._send_vehicle_command(400, param1=1.0)
        time.sleep(0.5) 
        
        print("[DEBUG] Firing OFFBOARD command (176)...")
        self._send_vehicle_command(176, param1=1.0, param2=6.0)
        
        # --- 4. NEW: AUTOMATED TAKEOFF PHASE ---
        print("[DEBUG] Executing automated takeoff to start altitude...")
        # The background heartbeat is currently publishing our [0, 0, -2.0, 0] action.
        # We just wait here for 3 seconds while PX4 physically flies the drone upward.
        for _ in range(30): 
            time.sleep(0.1)
            
        print("[DEBUG] Takeoff complete. Handing control to RL Agent.")
        
        # 5. Clear event flag and wait for next physics tick
        self.state_event.clear()
        self.state_event.wait(timeout=1.0)
        
        info = {}
        return self._get_obs(), info

    def _get_obs(self):
        """Extracts numerical state vector and transforms target to Body Frame."""
        if self.latest_odom is None:
            return np.zeros(12, dtype=np.float32)

        x, y, z = self.latest_odom.position
        vx, vy, vz = self.latest_odom.velocity
        roll, pitch, yaw = self._get_euler_from_quaternion(self.latest_odom.q)

        target_vec_n = 0.0 - x
        target_vec_e = 0.0 - y
        target_vec_d = 0.0 - z

        pad_rel_fwd = target_vec_n * math.cos(yaw) + target_vec_e * math.sin(yaw)
        pad_rel_right = -target_vec_n * math.sin(yaw) + target_vec_e * math.cos(yaw)
        pad_rel_down = target_vec_d

        return np.array([
            x, y, z, 
            vx, vy, vz, 
            roll, pitch, yaw, 
            pad_rel_fwd, pad_rel_right, pad_rel_down
        ], dtype=np.float32)

    def _compute_reward_and_termination(self):
        """Calculates dense reward, penalties, and checks episode boundaries."""
        if self.latest_odom is None:
            return 0.0, False

        x, y, z = self.latest_odom.position
        vx, vy, vz = self.latest_odom.velocity
        roll, pitch, yaw = self._get_euler_from_quaternion(self.latest_odom.q)

        d_lat = math.hypot(x, y)
        d_alt = abs(z) 
        
        w1 = 1.0  
        w2 = 0.5  
        w_jerk = 0.05
        w_yaw = 0.2

        reward = - (w1 * d_lat + w2 * d_alt)

        jerk = np.sum((self.current_action - self.previous_action)**2)
        reward -= (w_jerk * jerk)

        desired_yaw = math.atan2(0.0 - y, 0.0 - x)
        yaw_error = math.atan2(math.sin(desired_yaw - yaw), math.cos(desired_yaw - yaw))
        reward -= (w_yaw * abs(yaw_error))

        terminated = False
        
        if abs(roll) > math.radians(45) or abs(pitch) > math.radians(45):
            terminated = True
            reward -= 100.0
            print(f"[DEBUG] Episode Terminated: CRASH (Flipped) - Roll: {math.degrees(roll):.1f}°, Pitch: {math.degrees(pitch):.1f}°")
            
        elif d_lat > 5.0:
            terminated = True
            reward -= 50.0
            print(f"[DEBUG] Episode Terminated: OUT OF BOUNDS - Lateral Dist: {d_lat:.2f}m")
            
        elif z >= 0.0:
            terminated = True
            if d_lat < 0.2 and abs(vz) < 0.5:
                reward += 100.0
                print(f"[DEBUG] Episode Terminated: SUCCESSFUL LANDING!")
            else:
                reward -= 100.0
                print(f"[DEBUG] Episode Terminated: CRASH (Ground Impact) - Vz: {vz:.2f}m/s, Dist: {d_lat:.2f}m")

        return float(reward), terminated

    def close(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self.executor_thread.join()