import subprocess
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math
import time

# Standard ROS 2 Messages (Replaces PX4)
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty

class DroneEnv(gym.Env):
    """
    Gymnasium environment wrapper for Tello ROS 2 (Sim and Real).
    """
    def __init__(self):
        super(DroneEnv, self).__init__()
        
        # 1. Initialize ROS 2
        if not rclpy.ok():
            rclpy.init()
            
        self.node = rclpy.create_node('rl_tello_env_node')
        
        # 2. Define Spaces
        # Action: [vx (fwd), vy (left), vz (up), yaw_rate]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Observation: [x, y, z, vx, vy, vz, roll, pitch, yaw, pad_rel_fwd, pad_rel_left, pad_rel_up]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        # 3. ROS 2 Publishers & Subscribers
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Standard Tello topics
        self.odom_sub = self.node.create_subscription(
            Odometry, '/odom', self._odom_callback, qos_profile)
            
        self.cmd_vel_pub = self.node.create_publisher(
            Twist, '/cmd_vel', 10)
            
        self.takeoff_pub = self.node.create_publisher(
            Empty, '/takeoff', 10)
            
        self.land_pub = self.node.create_publisher(
            Empty, '/land', 10)

        # 4. Synchronization Variables
        self.state_event = threading.Event()
        self.latest_odom = None
        self.current_action = np.zeros(4, dtype=np.float32)
        self.previous_action = np.zeros(4, dtype=np.float32)
        
        self.max_steps = 300  # 30 seconds at 10Hz
        self.current_step = 0
        
        # 5. Start ROS 2 Spin Thread
        self.executor_thread = threading.Thread(target=self._spin_ros, daemon=True)
        self.executor_thread.start()
        
        self.node.get_logger().info("Tello Environment initialized.")

    def _spin_ros(self):
        rclpy.spin(self.node)

    def _odom_callback(self, msg):
        self.latest_odom = msg
        self.state_event.set()

    def _get_euler_from_quaternion(self, q):
        """Converts standard ROS 2 quaternion (x, y, z, w) to roll, pitch, yaw."""
        # Note: ROS 2 places 'w' at the end, unlike PX4.
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def _publish_action(self, action):
        """Publish Body-Frame velocities directly to /cmd_vel."""
        msg = Twist()
        msg.linear.x = float(action[0])  # Forward
        msg.linear.y = float(action[1])  # Left
        msg.linear.z = float(action[2])  # Up
        msg.angular.z = float(action[3]) # Yaw
        
        self.cmd_vel_pub.publish(msg)

    def step(self, action):
        self.previous_action = np.copy(self.current_action)
        self.current_action = action
        self.current_step += 1
        
        self.state_event.clear()
        self._publish_action(action)
        
        # Wait for physics update
        self.state_event.wait(timeout=0.5) 
        
        obs = self._get_obs()
        reward, terminated = self._compute_reward_and_termination()
        
        truncated = False
        if self.current_step >= self.max_steps and not terminated:
            truncated = True
            
        return obs, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_action = np.zeros(4, dtype=np.float32)
        self.previous_action = np.zeros(4, dtype=np.float32)
        
        self.node.get_logger().info("--- RESETTING EPISODE (TELEPORTING) ---")
        
        # 1. Kill all current velocities
        self._publish_action([0.0, 0.0, 0.0, 0.0])
        
        # 2. TELEPORT: Use Gazebo's command-line service to instantly move the drone
        # We spawn it 1.0 meters in the air so the RL agent starts in a hover state.
        subprocess.run([
            'gz', 'service', '-s', '/world/tello_sim/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '2000',
            '--req', 'name: "tello", position: {x: 0.0, y: 0.0, z: 1.0}, orientation: {w: 1.0, x: 0.0, y: 0.0, z: 0.0}'
        ], capture_output=True)
        
        # 3. Publish Standard ROS 2 Takeoff Command (For the REAL Tello later)
        self.takeoff_pub.publish(Empty())
        
        # 4. Wait a brief moment for the physics engine to register the teleport
        time.sleep(0.5) 
        
        self.state_event.clear()
        self.state_event.wait(timeout=1.0)
        
        return self._get_obs(), {}

    def _get_obs(self):
        """Extracts numerical state and calculates target relative to drone nose."""
        if self.latest_odom is None:
            return np.zeros(12, dtype=np.float32)

        # Extract position and velocity from nav_msgs/Odometry
        x = self.latest_odom.pose.pose.position.x
        y = self.latest_odom.pose.pose.position.y
        z = self.latest_odom.pose.pose.position.z
        
        vx = self.latest_odom.twist.twist.linear.x
        vy = self.latest_odom.twist.twist.linear.y
        vz = self.latest_odom.twist.twist.linear.z
        
        roll, pitch, yaw = self._get_euler_from_quaternion(self.latest_odom.pose.pose.orientation)

        # Target is global [0, 0, 0]. Vector pointing FROM drone TO target:
        target_vec_x = 0.0 - x
        target_vec_y = 0.0 - y
        target_vec_z = 0.0 - z

        # Rotate world vector into Body Frame (Forward, Left, Up)
        pad_rel_fwd = target_vec_x * math.cos(yaw) + target_vec_y * math.sin(yaw)
        pad_rel_left = -target_vec_x * math.sin(yaw) + target_vec_y * math.cos(yaw)
        pad_rel_up = target_vec_z

        return np.array([
            x, y, z, 
            vx, vy, vz, 
            roll, pitch, yaw, 
            pad_rel_fwd, pad_rel_left, pad_rel_up
        ], dtype=np.float32)

    def _compute_reward_and_termination(self):
        """Calculates dense reward based on distance, smoothness, and altitude."""
        if self.latest_odom is None:
            return 0.0, False

        x = self.latest_odom.pose.pose.position.x
        y = self.latest_odom.pose.pose.position.y
        z = self.latest_odom.pose.pose.position.z
        roll, pitch, yaw = self._get_euler_from_quaternion(self.latest_odom.pose.pose.orientation)

        d_lat = math.hypot(x, y)
        d_alt = abs(z)  # Z is already positive (Up)
        
        w1 = 1.0  
        w2 = 0.5  
        w_jerk = 0.05
        w_yaw = 0.2

        # Reward = negative distance
        reward = - (w1 * d_lat + w2 * d_alt)
        
        # Jerk Penalty
        jerk = np.sum((self.current_action - self.previous_action)**2)
        reward -= (w_jerk * jerk)

        # Yaw Alignment Penalty
        desired_yaw = math.atan2(0.0 - y, 0.0 - x)
        yaw_error = math.atan2(math.sin(desired_yaw - yaw), math.cos(desired_yaw - yaw))
        reward -= (w_yaw * abs(yaw_error))

        terminated = False
        
        # Crash Condition 1: Flipped
        if abs(roll) > math.radians(45) or abs(pitch) > math.radians(45):
            terminated = True
            reward -= 100.0
            print(f"[DEBUG] CRASH: Flipped over! Roll: {math.degrees(roll):.1f}, Pitch: {math.degrees(pitch):.1f}")
            
        # Crash Condition 2: Out of Bounds
        elif d_lat > 5.0:
            terminated = True
            reward -= 50.0
            print(f"[DEBUG] CRASH: Strayed out of bounds. Distance: {d_lat:.2f}m")
            
        # Ground Condition: Z drops to roughly floor level
        elif z <= 0.01:
            terminated = True
            # Success: Inside 20cm radius
            if d_lat < 0.2:
                reward += 100.0
                print("[DEBUG] SUCCESS: Landed on the pad!")
            else:
                reward -= 100.0
                print(f"[DEBUG] CRASH: Hit the ground away from pad. Z={z:.3f}, Dist={d_lat:.2f}m")

        return float(reward), terminated

    def close(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        self.executor_thread.join()