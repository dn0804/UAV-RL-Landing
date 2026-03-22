"""
Central configuration for the UAV RL Landing project.

Every tunable number lives here. Other modules import from this file
and never hardcode physical constants, reward weights, or scale factors.
Section references (§) point to the project plan for rationale.
"""

import math
import numpy as np

# ============================================================
# Physical Setup  (§3)
# ============================================================

# ArUco marker center is at the coordinate origin on the wall (x=0 plane).
# The landing pad is a circle on a surface 0.15 m below the marker.
MARKER_HEIGHT_ABOVE_PAD = 0.15          # m  (§3.1)
LANDING_PAD_RADIUS = 0.10               # m  (§3.2)
LANDING_PAD_FORWARD_OFFSET = 0.30       # m in front of marker  (§3.2)

# Default pad elevation (desk height). Adjustable per experiment.
PAD_ELEVATION = 0.75                    # m above ground  (§3.2)
MARKER_Z = PAD_ELEVATION + MARKER_HEIGHT_ABOVE_PAD  # absolute marker height

# World-frame reference positions.
# Coordinate system: wall at x=0, marker at (0, 0, MARKER_Z), +x into room.
MARKER_WORLD_POS = (0.0, 0.0, MARKER_Z)            # on the wall
PAD_WORLD_POS = (LANDING_PAD_FORWARD_OFFSET, 0.0, PAD_ELEVATION)  # in front of marker

# ============================================================
# Observation Space  (§6)
# ============================================================

# Channel layout of the 29-dim observation vector:
#   [0]  x          forward distance to marker (vision)
#   [1]  y          lateral offset (vision)
#   [2]  z          vertical offset (vision)
#   [3]  yaw        heading relative to marker (vision)
#   [4]  vx         forward velocity (odom, EMA-filtered)
#   [5]  vy         lateral velocity (odom, EMA-filtered)
#   [6]  vz         vertical velocity (odom, EMA-filtered)
#   [7]  z_tof      cosine-corrected ToF range (§6.2, §6.5)
#   [8]  dropout_t  steps since last marker detection (§6.4)
#   [9..28]         prev_actions: 5 steps × 4 axes (§6.3)

OBS_DIM = 29
ACTION_DIM = 4
ACTION_HISTORY_STEPS = 5  # number of past actions in obs  (§6.3)

# Per-channel normalization scale factors  (§6.6)
# Order must match the channel layout above.
OBS_SCALE = np.array([
    4.0,    # x        (§6.6: forward range up to ~4.4 m)
    3.5,    # y        (§6.6: lateral extent ±3.5 m)
    2.5,    # z        (§6.6: vertical extent)
    math.pi,# yaw      (§6.6: ±π rad)
    2.0,    # vx       (§6.6: indoor speed cap)
    2.0,    # vy
    2.0,    # vz
    2.5,    # z_tof    (§6.6: matches z scale)
    30.0,   # dropout  (§6.4: cap at 30 steps)
] + [1.0] * (ACTION_HISTORY_STEPS * ACTION_DIM),  # prev actions already in [-1, 1]
    dtype=np.float32,
)

OBS_CLIP = 1.5  # symmetric clip after normalization  (§6.6)

# Dropout timer  (§6.4)
DROPOUT_TIMER_CAP = 30  # steps; beyond this, data is dangerously stale

# ============================================================
# Action Space  (§7)
# ============================================================

# Agent outputs continuous [-1, 1] on 4 axes.
# For deployment: multiply by 100, round, clamp to [-100, 100].
ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# ============================================================
# Reward Function  (§8)
# ============================================================

# Horizontal centering  (§8.3)
W_HORIZONTAL = 10.0

# Gated descent  (§8.4)
W_DESCENT = 20.0
DESCENT_GATE_H_MID = 0.6    # m above marker where sigmoid is 50%
DESCENT_GATE_TAU = 0.1      # sigmoid steepness
DESCENT_GATE_SIGMA = 0.20   # m; Gaussian centering width = pad radius

# Marker tracking / yaw-to-bearing  (§8.5)
W_YAW = 0.3
YAW_PENALTY_D_MIN = 0.3     # m; penalty floor distance (caps at close range)

# Jerk penalty  (§8.6)
W_JERK = 0.1

# Time penalty  (§8.6)
W_TIME = 0.2

# Terminal rewards  (§8.9)
R_SUCCESS = 100.0
R_CRASH = -100.0
R_TIMEOUT = 0.0

# ============================================================
# Episode Termination  (§8.9)
# ============================================================

# Operational volume
X_MIN = 0.05    # m; wall collision boundary
X_MAX = 5.0     # m; Stage 3 max + 1 m buffer
Y_MIN = -4.0    # m
Y_MAX = 4.0     # m
Z_MAX = 2.5     # m; no useful trajectory above this

# Surface contact
TOF_CONTACT_THRESHOLD = 0.10  # m; any ToF below this = surface contact

# Success criteria (all must be met simultaneously)
SUCCESS_D_XY_MAX = 0.10      # m; horizontal distance to pad center
SUCCESS_VZ_MAX = 1.5         # m/s; vertical speed at touchdown
SUCCESS_VXY_MAX = 1.0        # m/s; horizontal speed at touchdown
SUCCESS_YAW_ERROR_MAX = math.radians(15)  # rad; heading alignment

# Crash: attitude limits
CRASH_ROLL_MAX = math.radians(45)
CRASH_PITCH_MAX = math.radians(45)

# Episode length
MAX_STEPS = 300              # 30 seconds at 10 Hz  (§8.6)

# ============================================================
# EMA Velocity Filter  (§11.5)
# ============================================================

EMA_ALPHA = 0.4  # smoothing parameter; τ ≈ 0.25 s at 10 Hz

# ============================================================
# ToF Sensor  (§6.5)
# ============================================================

# Cosine correction applied before obs: z_corrected = z_raw * cos(pitch) * cos(roll)
# No constants needed — just roll and pitch from IMU at runtime.

# ============================================================
# Curriculum Stages  (§9)
# ============================================================

CURRICULUM_STAGES = {
    0: {
        "name": "relaxed_landing",
        "d_min": 0.3,
        "d_max": 0.8,
        "angle_max": math.radians(10),
        "furniture_count": (0, 0),
        "vision_dropout_rate": 0.0,
        "odom_noise_tier": 1,
        "success_vz_max": 1.5,
        "success_vxy_max": 1.0,
    },
    1: {
        "name": "moderate_landing",
        "d_min": 0.3,
        "d_max": 0.8,
        "angle_max": math.radians(10),
        "furniture_count": (0, 0),
        "vision_dropout_rate": 0.0,
        "odom_noise_tier": 1,
        "success_vz_max": 0.5,
        "success_vxy_max": 0.5,
    },
    2: {
        "name": "precision_landing",
        "d_min": 0.3,
        "d_max": 0.8,
        "angle_max": math.radians(10),
        "furniture_count": (0, 0),
        "vision_dropout_rate": 0.0,
        "odom_noise_tier": 1,
        "success_vz_max": 0.3,
        "success_vxy_max": 0.2,
    },
    3: {
        "name": "medium_range",
        "d_min": 0.5,
        "d_max": 1.5,
        "angle_max": math.radians(25),
        "furniture_count": (1, 3),
        "vision_dropout_rate": 0.01,
        "odom_noise_tier": 2,
        "success_vz_max": 0.3,
        "success_vxy_max": 0.2,
    },
    4: {
        "name": "full_cone",
        "d_min": 0.3,
        "d_max": 4.0,
        "angle_max": math.radians(60),
        "furniture_count": (1, 5),
        "vision_dropout_rate": 0.02,
        "odom_noise_tier": 3,
        "success_vz_max": 0.3,
        "success_vxy_max": 0.2,
    },
}

# Stage transition  (§9, blended transitions)
CURRICULUM_PROMOTION_THRESHOLD = 0.70   # success rate to trigger transition
CURRICULUM_WINDOW_SIZE = 300            # rolling episode window for success rate
CURRICULUM_BLEND_EPISODES = 200         # episodes over which to blend distributions
CURRICULUM_BLEND_STEPS = 5              # number of ratio steps (80/20→60/40→...)

# ============================================================
# PPO Hyperparameters  (§10.2)
# ============================================================

PPO_CONFIG = {
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_epochs": 10,
    "batch_size": 64,
    "n_steps": 2048,             # rollout buffer per iteration
    "ent_coef": 0.01,
    "vf_coef": 0.5,              # SB3 default
    "max_grad_norm": 0.5,        # SB3 default
}

# Network architecture  (§10.1)
# Separate policy and value networks, 2×128 Tanh
NET_ARCH = dict(pi=[128, 128], vf=[128, 128])
ACTIVATION_FN = "Tanh"  # string key; resolved in train_ppo.py

# ============================================================
# Camera  (§5)
# ============================================================

# Tello camera horizontal FOV (used for spawn validation)
CAMERA_HFOV_RAD = math.radians(66)  # ~±33° from center
CAMERA_VFOV_RAD = math.radians(49)  # approximate vertical FOV at 960×720

# Sim camera resolution (from SDF: 640×480; plan target: 960×720)
SIM_CAMERA_WIDTH = 640
SIM_CAMERA_HEIGHT = 480

# ============================================================
# Gazebo / Sim Interface
# ============================================================

GZ_WORLD_NAME = "tello_sim"
GZ_DRONE_MODEL_NAME = "tello"
CONTROL_RATE_HZ = 10  # agent decision rate  (§7.2)

# ============================================================
# Domain Randomization — future  (§11)
# Placeholders so DR modules can import without guessing.
# ============================================================

# Wind  (§11.3)
WIND_DRIFT_RANGE = 0.03       # N per axis, uniform ±
WIND_IMPULSE_STD = 0.02       # N per axis per step

# Odometry noise  (§11.4)
ODOM_GAUSSIAN_STD = 0.1       # m/s per axis
ODOM_BIAS_WALK_STD = 0.005    # m/s per step increment
ODOM_FREEZE_PROB = 0.005      # per step (if not already frozen)
ODOM_FREEZE_DURATION = (1, 5) # steps (min, max)

# Battery sag  (§11.7)
BATTERY_EFFICACY_RANGE = (0.70, 1.10)  # thrust multiplier per episode

# Vision noise  (§11.2)
VISION_POS_NOISE_BASE = 0.01    # m, additive
VISION_POS_NOISE_SCALE = 0.015  # fraction of range (1.5%)
VISION_YAW_NOISE_STD = 0.03     # rad (~1.7°)
VISION_DROPOUT_DURATION = (1, 3)  # consecutive frames

# Observation delay queue  (§11.8)
OBS_DELAY_MIN = 2   # steps
OBS_DELAY_MAX = 4   # steps

# ToF noise  (§11.6)
TOF_NOISE_STD_NORMAL = 0.015    # m
TOF_NOISE_STD_CLOSE = 0.025     # m, below 0.05 m altitude

# Furniture  (§11.9)
FURNITURE_HEIGHT_RANGE = (0.30, None)  # max computed as PAD_ELEVATION - 0.10
FURNITURE_WIDTH_RANGE = (0.5, 1.2)     # m
FURNITURE_DEPTH_RANGE = (0.4, 0.8)     # m
FURNITURE_Y_RANGE = (-2.0, 2.0)        # m from centerline