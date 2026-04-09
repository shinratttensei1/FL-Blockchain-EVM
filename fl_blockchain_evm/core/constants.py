"""Shared constants for the FL-Blockchain-EVM project (MHEALTH).

Activity class names and metadata for Human Activity Recognition
used across the model, data loading, training, and evaluation modules.

Dataset: MHEALTH (Mobile Health) — Banos et al., 2014
- 10 subjects, 12 physical activity classes
- 23 sensor channels (chest acc, ECG x2, left-ankle acc/gyro/mag,
  right-arm acc/gyro/mag)
- Labels 1-12 (0 = null class, excluded)
"""

from typing import List

# ── Activity class names (MHEALTH) ───────────────────────────

ACTIVITY_NAMES: List[str] = [
    "STANDING",
    "SITTING",
    "LYING",
    "WALKING",
    "CLIMBING_STAIRS",
    "WAIST_BENDS",
    "ARM_ELEVATION",
    "KNEES_BENDING",
    "CYCLING",
    "JOGGING",
    "RUNNING",
    "JUMP_FRONT_BACK",
]

SC_NAMES: List[str] = ACTIVITY_NAMES  # alias used throughout codebase
NUM_CLASSES = 12

# ── Sensor channel count ──────────────────────────────────────
# 23 sensor columns per row (columns 1-23); column 24 is the label
NUM_CHANNELS = 23

# ── Window / stride for sliding-window segmentation ──────────
WINDOW_SIZE = 256   # samples at 50 Hz ≈ 5.12 s
WINDOW_STEP = 128   # 50 % overlap
