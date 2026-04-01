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
# The landing pad is on the desk surface directly in front of the marker.
#
# Placement is optimized so the entire marker stays within the Tello's
# 49° vertical FOV throughout the final descent to touchdown:
#   marker_z = PAD_ELEVATION - HALF_MARKER + pad_offset × tan(24.5°)
# This gives a ~25.6 cm continuous descent window from z ≈ 1.006 down
# to z = PAD_ELEVATION with the marker fully visible.
LANDING_PAD_RADIUS = 0.10               # m  (§3.2)
LANDING_PAD_FORWARD_OFFSET = 0.50       # m in front of marker  (was 0.30)

# Default pad elevation (desk height). Adjustable per experiment.
PAD_ELEVATION = 0.75                    # m above ground  (§3.2)

# Marker height derived from the exact FOV constraint:
#   marker_z = PAD_ELEVATION - 0.10 + pad_offset × tan(VFOV/2)
# This places the marker's bottom corners exactly at the lower VFOV
# edge when the drone is at pad height, maximizing the descent window.
MARKER_HEIGHT_ABOVE_PAD = -0.10 + LANDING_PAD_FORWARD_OFFSET * math.tan(math.radians(24.5))
MARKER_Z = PAD_ELEVATION + MARKER_HEIGHT_ABOVE_PAD

# World-frame reference positions.
# Coordinate system: wall at x=0, marker at (0, 0, MARKER_Z), +x into room.
MARKER_WORLD_POS = (0.0, 0.0, MARKER_Z)            # on the wall
PAD_WORLD_POS = (LANDING_PAD_FORWARD_OFFSET, 0.0, PAD_ELEVATION)  # in front of marker

# Marker half-width (used by spawner FOV validation)
MARKER_HALF_WIDTH = 0.10                # m; half of 0.20m marker

# ============================================================
# Observation Space  (§6)
# ============================================================

# Channel layout of the 33-dim observation vector:
#   [0]  x          forward distance to marker (vision, body frame)
#   [1]  y          lateral offset (vision, body frame)
#   [2]  z          vertical offset (vision, body frame)
#   [3]  yaw        heading relative to marker (odom-derived)
#   [4]  vx         forward velocity (odom, EMA-filtered)
#   [5]  vy         lateral velocity (odom, EMA-filtered)
#   [6]  vz         vertical velocity (odom, EMA-filtered)
#   [7]  z_tof      cosine-corrected ToF range (§6.2, §6.5)
#   [8]  dropout_t  steps since last marker detection (§6.4)
#   [9]  roll       body roll angle (IMU)
#   [10] pitch      body pitch angle (IMU)
#   [11] marker_px  marker pixel x in frame, normalized [-1, 1]
#   [12] marker_py  marker pixel y in frame, normalized [-1, 1]
#   [13..32]        prev_actions: 5 steps × 4 axes (§6.3)

OBS_DIM = 33
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
    math.pi,# roll
    math.pi,# pitch
    1.0,    # marker_px (already normalized to [-1, 1])
    1.0,    # marker_py (already normalized to [-1, 1])
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
#
# All weights scaled to keep total returns in roughly [−35, +30].
# This prevents value function gradient overflow (NaN) that occurred
# with the original [−150, +125] range.  The 5x reduction cuts value
# loss by 25x (MSE) and gradient magnitude by 5x.
# ============================================================

# Horizontal centering  (§8.3)
W_HORIZONTAL = 2.0

# Descent reward  (§8.4)
# Plain delta-z reward: positive when descending, negative when ascending.
W_DESCENT = 4.0

# Marker tracking / yaw-to-bearing  (§8.5)
W_YAW = 0.006
YAW_PENALTY_D_MIN = 0.3     # m; penalty floor distance (caps at close range)

# Pixel centering penalty
W_CENTERING = 0.10
CENTERING_DEADZONE = 0.15   # normalized; 15% of half-frame in each axis

# Jerk penalty  (§8.6)
W_JERK = 0.004

# Velocity penalty (proximity-scaled)
# Penalizes speed proportional to closeness to the pad.  Horizontal
# speed is weighted more heavily — lateral overshoot at close range is
# hard to recover from.  Vertical gets a lighter touch so the descent
# reward (W_DESCENT = 4.0) still dominates at range.
#
# proximity = D_REF / max(d_pad, D_REF)  →  1.0 inside D_REF, decays beyond
# r_vel_xy = -W_VEL_XY * v_xy * proximity
# r_vel_z  = -W_VEL_Z  * |vz| * proximity
W_VEL_XY = 0.07
W_VEL_Z = 0.04
VEL_PENALTY_D_REF = 0.5    # m; penalty at full strength inside this radius

W_VEL_XY_UNI = 0.10
MAX_VEL_XY = 0.5

# Time penalty  (§8.6)
# −0.01/step × 300 steps = −3.0 max.  Forward progress at 0.5 m/s
# yields +0.1/step from horizontal centering alone, easily overcoming
# the time penalty.  Timeout (−3 shaping + −18 terminal = −21) is now
# better than a wall crash (−24.5), fixing the old kamikaze incentive.
W_TIME = 0.01

# Terminal rewards  (§8.9)
R_SUCCESS = 20.0
R_CRASH = -25.0
R_TIMEOUT = -12.0

# ============================================================
# Episode Termination  (§8.9)
# ============================================================

# Operational volume
X_MIN = 0.08    # m; wall collision boundary — reduced from 0.05 for
                #     attitude dynamics: the drone must pitch backward to
                #     decelerate, which takes time and distance.  Gazebo
                #     collision geometry prevents actual wall penetration.
X_MAX = 5.0     # m; Stage 3 max + 1 m buffer
Y_MIN = -4.0    # m
Y_MAX = 4.0     # m
Z_MAX = 2.5     # m; no useful trajectory above this

# Surface contact
TOF_CONTACT_THRESHOLD = 0.03  # m; any ToF below this = surface contact

# Success criteria (all must be met simultaneously)
SUCCESS_D_XY_MAX = 0.10      # m; horizontal distance to pad center
SUCCESS_VZ_MAX = 1.5         # m/s; vertical speed at touchdown
SUCCESS_VXY_MAX = 1.0        # m/s; horizontal speed at touchdown

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
    # ── Close-range velocity tightening (each ~25% stricter) ──
    0: {
        "name": "relaxed_landing",
        "d_min": 0.3,
        "d_max": 0.8,
        "angle_max": math.radians(10),
        "furniture_count": (0, 0),
        "vision_dropout_rate": 0.0,
        "odom_noise_tier": 1,
        "success_vz_max": 1.2,
        "success_vxy_max": 1.0,
        "success_d_xy_max": 0.25,
    },
    1: {
        "name": "easy_landing",
        "d_min": 0.3,
        "d_max": 0.8,
        "angle_max": math.radians(10),
        "furniture_count": (0, 0),
        "vision_dropout_rate": 0.0,
        "odom_noise_tier": 1,
        "success_vz_max": 0.9,
        "success_vxy_max": 0.75,
        "success_d_xy_max": 0.22,
    },
    2: {
        "name": "moderate_landing",
        "d_min": 0.3,
        "d_max": 0.8,
        "angle_max": math.radians(10),
        "furniture_count": (0, 0),
        "vision_dropout_rate": 0.0,
        "odom_noise_tier": 1,
        "success_vz_max": 0.65,
        "success_vxy_max": 0.55,
        "success_d_xy_max": 0.19,
    },
    3: {
        "name": "firm_landing",
        "d_min": 0.3,
        "d_max": 0.8,
        "angle_max": math.radians(10),
        "furniture_count": (0, 0),
        "vision_dropout_rate": 0.0,
        "odom_noise_tier": 1,
        "success_vz_max": 0.45,
        "success_vxy_max": 0.40,
        "success_d_xy_max": 0.16,
    },
    4: {
        "name": "precision_landing",
        "d_min": 0.3,
        "d_max": 0.8,
        "angle_max": math.radians(10),
        "furniture_count": (0, 0),
        "vision_dropout_rate": 0.0,
        "odom_noise_tier": 1,
        "success_vz_max": 0.33,
        "success_vxy_max": 0.25,
        "success_d_xy_max": 0.13,
    },
    # ── Spatial expansion (velocity holds at precision level) ──
    5: {
        "name": "medium_range",
        "d_min": 0.5,
        "d_max": 1.5,
        "angle_max": math.radians(25),
        "furniture_count": (0, 0),
        "vision_dropout_rate": 0.0,
        "odom_noise_tier": 1,
        "success_vz_max": 0.33,
        "success_vxy_max": 0.25,
        "success_d_xy_max": 0.13,
    },
    6: {
        "name": "full_cone_moderate",
        "d_min": 0.3,
        "d_max": 4.0,
        "angle_max": math.radians(60),
        "furniture_count": (0, 0),
        "vision_dropout_rate": 0.0,
        "odom_noise_tier": 1,
        "success_vz_max": 0.50,
        "success_vxy_max": 0.40,
        "success_d_xy_max": 0.15,
    },
    7: {
        "name": "full_cone",
        "d_min": 0.3,
        "d_max": 4.0,
        "angle_max": math.radians(60),
        "furniture_count": (0, 0),
        "vision_dropout_rate": 0.0,
        "odom_noise_tier": 1,
        "success_vz_max": 0.33,
        "success_vxy_max": 0.25,
        "success_d_xy_max": 0.10,
    },
}

# Stage transition  (§9, blended transitions)
CURRICULUM_PROMOTION_THRESHOLD = 0.75   # success rate to trigger transition
CURRICULUM_WINDOW_SIZE = 300            # rolling episode window for success rate
CURRICULUM_BLEND_EPISODES = 1000        # episodes over which to blend distributions
CURRICULUM_BLEND_STEPS = 5              # number of ratio steps (80/20→60/40→...)

# ============================================================
# PPO Hyperparameters  (§10.2)
# ============================================================

PPO_CONFIG = {
    "learning_rate": 0.00015,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "n_epochs": 10,
    "batch_size": 64,
    "n_steps": 2048,             # rollout buffer per iteration
    "ent_coef": 0.0001,           # reduced for attitude dynamics — motor
                                  # physics already provide ample exploration
                                  # noise; too much entropy prevents std
                                  # from decreasing
    "vf_coef": 0.5,              # SB3 default
    "max_grad_norm": 0.5,        # SB3 default
}

# Network architecture  (§10.1)
# Separate policy and value networks, 2×128 Tanh
NET_ARCH = dict(pi=[128, 128], vf=[128, 128])
ACTIVATION_FN = "Tanh"  # string key; resolved in train_ppo.py

# Initial action std — controls exploration noise at training start.
# With real attitude dynamics, high std (default 1.0) produces chaotic
# trajectories that make returns unpredictable, preventing the value
# function from learning.  0.6 (log_std_init=-0.5) gives enough
# exploration to discover success while keeping trajectories smooth
# enough for the value function to extract signal.
LOG_STD_INIT = -0.5  # initial std ≈ 0.6; use in policy_kwargs
LOG_STD_MIN = -1.5   # minimum std ≈ 0.30; prevents entropy collapse
                      # that makes every SGD update exceed the KL threshold

# ============================================================
# Camera  (§5)
# ============================================================

# Tello camera horizontal FOV (used for spawn validation)
CAMERA_HFOV_RAD = math.radians(66)  # ~±33° from center
CAMERA_VFOV_RAD = math.radians(49)  # approximate vertical FOV at 960×720

# Sim camera resolution (from model.sdf)
SIM_CAMERA_WIDTH = 960
SIM_CAMERA_HEIGHT = 720
SIM_CAMERA_HFOV_RAD = 1.43117  # 82° horizontal FOV from SDF

# Derived camera intrinsics for solvePnP (computed from SDF parameters)
SIM_CAMERA_FX = SIM_CAMERA_WIDTH / (2.0 * math.tan(SIM_CAMERA_HFOV_RAD / 2.0))
SIM_CAMERA_FY = SIM_CAMERA_FX  # square pixels
SIM_CAMERA_CX = SIM_CAMERA_WIDTH / 2.0
SIM_CAMERA_CY = SIM_CAMERA_HEIGHT / 2.0

# ArUco marker (§3.1)
MARKER_SIZE = 0.20   # m; physical side length (20 cm)
MARKER_ID = 0        # ArUco dictionary ID to track
MARKER_DICTIONARY = "DICT_4X4_50"  # must match the generated aruco_marker.png

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