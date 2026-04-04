"""
Central configuration for the UAV RL Landing project.

Positions reconciled against tello_world.sdf (source of truth):
    Desk:        center (0, 0, 0.375), rotated 90° → 1.00m(X) × 0.60m(Y) × 0.75m(Z)
    Landing pad: (0.05, 0, 0.75), cylinder r=0.15
    Marker pole: (-0.40, 0, 0.925), 35cm tall
    ArUco marker:(-0.40, 0, 1.10), 20×20cm, faces +X

Training pipeline (every stage, every episode):
    Approach → hover checkpoint (+bonus) → descend → land (+success)
"""

import math
import numpy as np

# ============================================================
# Physical Setup
# ============================================================

DESK_X_MIN = -0.50
DESK_X_MAX = 0.50
DESK_Y_MIN = -0.30
DESK_Y_MAX = 0.30
PAD_ELEVATION = 0.75

LANDING_PAD_RADIUS = 0.15
PAD_WORLD_POS = (0.05, 0.0, PAD_ELEVATION)

MARKER_CENTER_Z = 1.10
MARKER_WORLD_POS = (-0.40, 0.0, MARKER_CENTER_Z)
MARKER_Z = MARKER_CENTER_Z
MARKER_HALF_WIDTH = 0.10
MARKER_SIZE = 0.20
MARKER_ID = 0
MARKER_DICTIONARY = "DICT_4X4_50"
LANDING_PAD_FORWARD_OFFSET = PAD_WORLD_POS[0]

# ============================================================
# Observation Space
# ============================================================

OBS_DIM = 33
ACTION_DIM = 4
ACTION_HISTORY_STEPS = 5

OBS_SCALE = np.array([
    5.5, 5.0, 2.0, math.pi,
    2.0, 2.0, 2.0,
    2.0, 30.0,
    math.pi, math.pi,
    1.0, 1.0,
] + [1.0] * (ACTION_HISTORY_STEPS * ACTION_DIM),
    dtype=np.float32,
)

OBS_CLIP = 1.5
DROPOUT_TIMER_CAP = 30

# ── Hover checkpoint dropout tolerance ───────────────────────
# Number of consecutive dropout frames tolerated during hover
# dwell accumulation.  0 = original strict behavior.  A small
# value (1-2) absorbs ArUco flicker without crediting dwell
# against a fully stale position estimate.
HOVER_DROPOUT_TOLERANCE = 2

# ============================================================
# Action Space
# ============================================================

ACTION_LOW = -1.0
ACTION_HIGH = 1.0

# ============================================================
# Reward Function
# ============================================================

# ── Approach + centering ─────────────────────────────────────

W_HORIZONTAL = 2.0
W_Z_ALIGN = 3.0

W_YAW = 0.006
YAW_PENALTY_D_MIN = 0.3

W_CENTERING = 0.10
CENTERING_DEADZONE = 0.15

W_VEL_XY = 0.07
W_VEL_Z = 0.04
VEL_PENALTY_D_REF = 0.5
W_VEL_XY_UNI = 0.10
MAX_VEL_XY = 0.5

W_JERK = 0.004
W_TIME = 0.01

# ── Pre-checkpoint hover bonus ───────────────────────────────

W_HOVER_BONUS = 0.05
HOVER_Z_TOLERANCE = 0.30

# ── Hover checkpoint ─────────────────────────────────────────

R_HOVER_CHECKPOINT = 15.0

# ── Post-checkpoint descent ──────────────────────────────────

W_DESCENT_COMMITTED = 6.0
W_XY_HOLD = 0.15

DESCENT_GATE_D_PAD = 0.25
DESCENT_COMMIT_Z_MARGIN = 0.10

# ── Dropout freeze (pre-checkpoint only) ─────────────────────
# When the marker is lost before checkpoint, penalize XYZ action
# commands to teach the drone to stop and search with yaw only.
# Grace period of 3 steps (0.3s) to ignore detection flicker.
W_DROPOUT_FREEZE = 0.1       # per unit of |action_xyz| per step
DROPOUT_GRACE_STEPS = 15    # steps before freeze penalty activates

# ── Terminal rewards ─────────────────────────────────────────

R_SUCCESS = 15.0
R_SUCCESS_NO_CHECKPOINT = 5.0
R_CRASH = -25.0
R_TIMEOUT = -22.0

# ============================================================
# Episode Termination — Conical Operational Volume
#
# 3D cone with tip 0.5m behind marker, opening in +X direction.
# Half-angle 60° (120° full), length 6m.
# Flat floor at z=0.20, flat ceiling at Z_CEILING.
# ============================================================

# Cone geometry
CONE_TIP_X = MARKER_WORLD_POS[0] - 1   # -0.90
CONE_TIP_Y = MARKER_WORLD_POS[1]          #  0.00
CONE_TIP_Z = MARKER_CENTER_Z              #  1.10
CONE_HALF_ANGLE_RAD = math.radians(70)    # 120° full opening
CONE_COS_HALF_ANGLE = math.cos(CONE_HALF_ANGLE_RAD)  # 0.5
CONE_LENGTH = 7.0                          # m from tip along +X axis

# Flat floor and ceiling
Z_FLOOR = 0.20       # m — drone must stay above this
Z_CEILING = 1.95     # m — highest spawn point, drone must stay below

# Surface contact
TOF_CONTACT_THRESHOLD = 0.03

# Landing success criteria
SUCCESS_D_XY_MAX = 0.10
SUCCESS_VZ_MAX = 1.5
SUCCESS_VXY_MAX = 1.0

# Crash: attitude limits
CRASH_ROLL_MAX = math.radians(45)
CRASH_PITCH_MAX = math.radians(45)

MAX_STEPS = 150

# ============================================================
# Hover Checkpoint Conditions
# ============================================================

HOVER_DWELL_STEPS = 10
HOVER_Z_MIN_CLEARANCE = 0.10

# ============================================================
# EMA Velocity Filter
# ============================================================

EMA_ALPHA = 0.4

# ============================================================
# Curriculum Stages
# ============================================================

CURRICULUM_STAGES = {
    0: {
        "name": "close_easy",
        "d_min": 0.5, "d_max": 1.5,
        "angle_max": math.radians(15),
        "hover_d_pad_max": 0.30, "hover_vxy_max": 0.5, "hover_dwell_steps": 5,
        "success_vz_max": 1.2, "success_vxy_max": 0.8, "success_d_xy_max": 0.25,
        "max_steps": 1000,
        "promotion_threshold": 0.90,
    },
    1: {
        "name": "close_moderate",
        "d_min": 0.5, "d_max": 1.5,
        "angle_max": math.radians(15),
        "hover_d_pad_max": 0.25, "hover_vxy_max": 0.4, "hover_dwell_steps": 8,
        "success_vz_max": 0.9, "success_vxy_max": 0.6, "success_d_xy_max": 0.22,
        "max_steps": 750,
        "promotion_threshold": 0.90,
    },
    2: {
        "name": "close_firm",
        "d_min": 0.5, "d_max": 1.5,
        "angle_max": math.radians(20),
        "hover_d_pad_max": 0.20, "hover_vxy_max": 0.35, "hover_dwell_steps": 10,
        "success_vz_max": 0.7, "success_vxy_max": 0.45, "success_d_xy_max": 0.20,
        "max_steps": 500,
        "promotion_threshold": 0.90,
    },
    3: {
        "name": "close_precise",
        "d_min": 0.5, "d_max": 1.5,
        "angle_max": math.radians(25),
        "hover_d_pad_max": 0.18, "hover_vxy_max": 0.30, "hover_dwell_steps": 10,
        "success_vz_max": 0.5, "success_vxy_max": 0.35, "success_d_xy_max": 0.18,
        "max_steps": 300,
        "promotion_threshold": 0.85,
    },
    4: {
        "name": "medium_moderate",
        "d_min": 0.5, "d_max": 2.5,
        "angle_max": math.radians(30),
        "hover_d_pad_max": 0.20, "hover_vxy_max": 0.35, "hover_dwell_steps": 10,
        "success_vz_max": 0.6, "success_vxy_max": 0.40, "success_d_xy_max": 0.18,
        "max_steps": 400,
        "promotion_threshold": 0.85,
    },
    5: {
        "name": "medium_precise",
        "d_min": 0.5, "d_max": 2.5,
        "angle_max": math.radians(40),
        "hover_d_pad_max": 0.18, "hover_vxy_max": 0.30, "hover_dwell_steps": 10,
        "success_vz_max": 0.45, "success_vxy_max": 0.30, "success_d_xy_max": 0.15,
        "max_steps": 400,
        "promotion_threshold": 0.85,
    },
    6: {
        "name": "far_moderate",
        "d_min": 0.5, "d_max": 4.0,
        "angle_max": math.radians(45),
        "hover_d_pad_max": 0.20, "hover_vxy_max": 0.35, "hover_dwell_steps": 10,
        "success_vz_max": 0.5, "success_vxy_max": 0.35, "success_d_xy_max": 0.18,
        "max_steps": 400,
        "promotion_threshold": 0.80,
    },
    7: {
        "name": "far_wide",
        "d_min": 0.5, "d_max": 4.0,
        "angle_max": math.radians(60),
        "hover_d_pad_max": 0.20, "hover_vxy_max": 0.35, "hover_dwell_steps": 10,
        "success_vz_max": 0.5, "success_vxy_max": 0.35, "success_d_xy_max": 0.18,
        "max_steps": 400,
        "promotion_threshold": 0.80,
    },
    8: {
        "name": "far_precise",
        "d_min": 0.5, "d_max": 4.0,
        "angle_max": math.radians(60),
        "hover_d_pad_max": 0.15, "hover_vxy_max": 0.25, "hover_dwell_steps": 10,
        "success_vz_max": 0.3, "success_vxy_max": 0.20, "success_d_xy_max": 0.18,
        "max_steps": 500,
        "promotion_threshold": 0.80,  # final stage — no promotion, but kept for consistency
    },
}

# Default fallback if a stage lacks "promotion_threshold" key
CURRICULUM_PROMOTION_THRESHOLD = 0.80
CURRICULUM_WINDOW_SIZE = 300
CURRICULUM_BLEND_EPISODES = 300
CURRICULUM_BLEND_STEPS = 5

# ============================================================
# PPO Hyperparameters
# ============================================================

PPO_CONFIG = {
    "learning_rate": 0.00015, "gamma": 0.99, "gae_lambda": 0.95,
    "clip_range": 0.2, "n_epochs": 10, "batch_size": 64,
    "n_steps": 2048, "ent_coef": 0.0001, "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}

NET_ARCH = dict(pi=[128, 128], vf=[128, 128])
ACTIVATION_FN = "Tanh"
LOG_STD_INIT = -0.5
LOG_STD_MIN = -1.5

# ============================================================
# Camera
# ============================================================

CAMERA_HFOV_RAD = math.radians(66)
CAMERA_VFOV_RAD = math.radians(49)
SIM_CAMERA_WIDTH =  480 #1280
SIM_CAMERA_HEIGHT = 360 #720
SIM_CAMERA_HFOV_RAD = 1.43117
SIM_CAMERA_FX = SIM_CAMERA_WIDTH / (2.0 * math.tan(SIM_CAMERA_HFOV_RAD / 2.0))
SIM_CAMERA_FY = SIM_CAMERA_FX
SIM_CAMERA_CX = SIM_CAMERA_WIDTH / 2.0
SIM_CAMERA_CY = SIM_CAMERA_HEIGHT / 2.0

# ============================================================
# Gazebo / Sim
# ============================================================

GZ_WORLD_NAME = "tello_sim"
GZ_DRONE_MODEL_NAME = "tello"
CONTROL_RATE_HZ = 10

# ============================================================
# Domain Randomization — future
# ============================================================

WIND_DRIFT_RANGE = 0.03
WIND_IMPULSE_STD = 0.02
ODOM_GAUSSIAN_STD = 0.1
ODOM_BIAS_WALK_STD = 0.005
ODOM_FREEZE_PROB = 0.005
ODOM_FREEZE_DURATION = (1, 5)
BATTERY_EFFICACY_RANGE = (0.70, 1.10)
VISION_POS_NOISE_BASE = 0.01
VISION_POS_NOISE_SCALE = 0.015
VISION_YAW_NOISE_STD = 0.03
VISION_DROPOUT_DURATION = (1, 3)
OBS_DELAY_MIN = 2
OBS_DELAY_MAX = 4
TOF_NOISE_STD_NORMAL = 0.015
TOF_NOISE_STD_CLOSE = 0.025
FURNITURE_HEIGHT_RANGE = (0.30, None)
FURNITURE_WIDTH_RANGE = (0.5, 1.2)
FURNITURE_DEPTH_RANGE = (0.4, 0.8)
FURNITURE_Y_RANGE = (-2.0, 2.0)