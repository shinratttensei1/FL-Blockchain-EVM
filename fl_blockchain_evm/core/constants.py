"""Shared constants for the FL-Blockchain-EVM project (UCI-HAR).

Activity class names and metadata for Human Activity Recognition
used across the model, data loading, training, and evaluation modules.

Dataset: UCI Human Activity Recognition Using Smartphones
(Anguita et al., 2013) — 6 activity classes, 30 subjects.
"""

from typing import List

# ── Activity class names (UCI-HAR) ────────────────────────────

ACTIVITY_NAMES: List[str] = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]

SC_NAMES: List[str] = ACTIVITY_NAMES  # alias used throughout codebase
NUM_CLASSES = 6
