"""UCI-HAR data loading, caching, balancing, and augmentation.

Handles the full data pipeline from raw UCI Human Activity Recognition
inertial signals to PyTorch DataLoaders with class-balanced sampling.

Dataset: UCI Human Activity Recognition Using Smartphones (Anguita et al., 2013)
- 30 subjects (21 train, 9 test), 6 activities
- 9 inertial signal channels, 128 timesteps per window
- Expected at: data/UCI HAR Dataset/

FL partitioning: training subjects are split across clients by subject ID,
producing a realistic non-IID federated scenario.
"""

import os
import urllib.request
import zipfile
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from fl_blockchain_evm.core.constants import NUM_CLASSES, SC_NAMES


# ── Subject splits (per original dataset partition) ───────────

_TRAIN_SUBJECTS = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17,
                   19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
_TEST_SUBJECTS  = [2, 4, 9, 10, 12, 13, 18, 20, 24]

_SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

DATA_DIR = "data/UCI HAR Dataset"

_CACHE: dict = {}


# ── Dataset download ─────────────────────────────────────────

_UCI_HAR_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00240/UCI%20HAR%20Dataset.zip"
)
_SENTINEL = os.path.join(DATA_DIR, "train", "y_train.txt")


def _ensure_dataset():
    """Download and unzip UCI-HAR if not already present."""
    if os.path.exists(_SENTINEL):
        return

    os.makedirs("data", exist_ok=True)
    zip_path = "data/UCI_HAR_Dataset.zip"

    print(f"  [UCI-HAR] Dataset not found. Downloading from UCI repository ...")
    urllib.request.urlretrieve(_UCI_HAR_URL, zip_path)
    print(f"  [UCI-HAR] Extracting to data/ ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("data")
    os.remove(zip_path)
    print(f"  [UCI-HAR] Dataset ready at {DATA_DIR}")


# ── Raw data loading ──────────────────────────────────────────

def _load_raw(split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load inertial signals, 0-indexed labels, and subject IDs for one split.

    Returns:
        X:        (N, 9, 128) float32 — raw inertial channels
        y:        (N,) int32  — activity labels, 0-indexed (0=WALKING … 5=LAYING)
        subjects: (N,) int32  — subject IDs (1-30)
    """
    base    = os.path.join(DATA_DIR, split)
    sig_dir = os.path.join(base, "Inertial Signals")

    channels = []
    for name in _SIGNAL_NAMES:
        path = os.path.join(sig_dir, f"{name}_{split}.txt")
        channels.append(np.loadtxt(path, dtype=np.float32))  # (N, 128)
    X = np.stack(channels, axis=1)  # (N, 9, 128)

    y        = np.loadtxt(os.path.join(base, f"y_{split}.txt"),
                          dtype=np.int32) - 1          # convert 1-6 → 0-5
    subjects = np.loadtxt(os.path.join(base, f"subject_{split}.txt"),
                          dtype=np.int32)
    return X, y, subjects


def _get_data():
    """Load and cache both train and test splits."""
    if "all" not in _CACHE:
        _ensure_dataset()
        print(f"  [UCI-HAR] Loading from {DATA_DIR} ...")
        X_tr, y_tr, s_tr = _load_raw("train")
        X_te, y_te, s_te = _load_raw("test")
        _CACHE["all"] = (X_tr, y_tr, s_tr, X_te, y_te, s_te)
        print(f"  [UCI-HAR] Train: {len(X_tr)} samples | "
              f"Test: {len(X_te)} samples | "
              f"Channels: 9 | Timesteps: 128")
    return _CACHE["all"]


# ── Class balancing (ROS+RUS) ─────────────────────────────────

def _balance_ros_rus(X: np.ndarray, y: np.ndarray, beta: float = 1.0):
    """ROS+RUS balancing for one-hot encoded single-label data.

    y is one-hot (N, C). For each class:
      - Under-represented (count < target): oversample (ROS)
      - Over-represented  (count > target): undersample (RUS)
    Target = m_s + (m_l - m_s) * beta  [beta=1.0 → full equalization]
    """
    counts  = y.sum(0).astype(int)
    primary = np.argmax(y, axis=1)  # safe for single-label one-hot

    pc_counts = np.array([np.sum(primary == c) for c in range(NUM_CLASSES)])
    active    = pc_counts[pc_counts > 0]

    if len(active) == 0:
        return X, y

    m_l, m_s = int(active.max()), int(active.min())
    target   = int(m_s + (m_l - m_s) * beta)

    print(f"  [ROS+RUS] m_l={m_l}, m_s={m_s}, β={beta}, target={target}")
    print(f"  [ROS+RUS] Before: {dict(zip(SC_NAMES, pc_counts))}")

    bX, by = [], []
    for c in range(NUM_CLASSES):
        idx = np.where(primary == c)[0]
        if len(idx) == 0:
            continue
        if len(idx) < target:
            idx = np.random.choice(idx, target, replace=True)
        elif len(idx) > target:
            idx = np.random.choice(idx, target, replace=False)
        bX.append(X[idx])
        by.append(y[idx])

    bX   = np.concatenate(bX)
    by   = np.concatenate(by)
    perm = np.random.permutation(len(bX))

    print(f"  [ROS+RUS] After:  {dict(zip(SC_NAMES, by[perm].sum(0).astype(int)))}")
    return bX[perm], by[perm]


# ── Data augmentation ─────────────────────────────────────────

def _augment(x: torch.Tensor) -> torch.Tensor:
    """Inertial sensor signal augmentation.

    Simulates realistic variation in wearable sensor data:
    - Gaussian noise (sensor measurement noise)
    - Per-channel amplitude scaling (sensor calibration differences)
    - Temporal shift (activity phase offset between subjects)
    """
    # Gaussian noise
    if torch.rand(1).item() < 0.8:
        x = x + torch.randn_like(x) * 0.05

    # Per-channel amplitude scaling
    if torch.rand(1).item() < 0.5:
        scale = 0.8 + 0.4 * torch.rand(x.size(0), x.size(1), 1, device=x.device)
        x = x * scale

    # Temporal shift (~10 samples)
    if torch.rand(1).item() < 0.5:
        shift = torch.randint(-10, 11, (1,)).item()
        if shift > 0:
            x = F.pad(x[:, :, shift:], (0, shift))
        elif shift < 0:
            x = F.pad(x[:, :, :shift], (-shift, 0))

    return x


# ── DataLoader construction ───────────────────────────────────

def load_data(partition_id: int, num_partitions: int, beta: float = 1.0,
              batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Build train/test DataLoaders for one FL client partition.

    Training data is partitioned by subject ID (non-IID), giving each
    client data from a disjoint set of subjects. Test data is shared
    across all clients (global test set).

    Args:
        partition_id:    Client index (0 … num_partitions-1).
        num_partitions:  Total number of FL clients.
        beta:            ROS+RUS balance strength (0=no balancing, 1=full).
        batch_size:      DataLoader batch size.

    Returns:
        (trainloader, testloader)
    """
    X_tr, y_tr, s_tr, X_te, y_te, _ = _get_data()

    # Assign training subjects to this partition
    train_subjects = sorted(_TRAIN_SUBJECTS)
    chunks         = np.array_split(train_subjects, num_partitions)
    my_subjects    = set(int(s) for s in chunks[partition_id])

    my_idx = np.where(np.isin(s_tr, list(my_subjects)))[0]

    X_part = X_tr[my_idx].copy()
    y_part = y_tr[my_idx].copy()

    print(f"  [UCI-HAR] Partition {partition_id}: "
          f"subjects={sorted(my_subjects)}, samples={len(X_part)}")

    # Z-score normalize per channel using partition train statistics
    mu         = X_part.mean(axis=(0, 2), keepdims=True)
    sd         = X_part.std(axis=(0, 2), keepdims=True) + 1e-8
    X_part_n   = (X_part - mu) / sd
    X_te_n     = (X_te   - mu) / sd

    # One-hot encode labels
    y_part_oh = np.eye(NUM_CLASSES, dtype=np.float32)[y_part]  # (N, 6)
    y_te_oh   = np.eye(NUM_CLASSES, dtype=np.float32)[y_te]    # (M, 6)

    if beta > 0:
        X_part_n, y_part_oh = _balance_ros_rus(X_part_n, y_part_oh, beta=beta)

    X_tr_t = torch.tensor(X_part_n, dtype=torch.float32)
    y_tr_t = torch.tensor(y_part_oh, dtype=torch.float32)
    X_te_t = torch.tensor(X_te_n,   dtype=torch.float32)
    y_te_t = torch.tensor(y_te_oh,  dtype=torch.float32)

    # Weighted sampler for balanced mini-batches
    primary_classes = y_tr_t.argmax(dim=1).numpy()
    class_counts    = np.bincount(primary_classes, minlength=NUM_CLASSES).astype(float)
    class_counts    = np.maximum(class_counts, 1.0)
    sample_weights  = 1.0 / class_counts[primary_classes]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float64),
        num_samples=len(sample_weights),
        replacement=True,
    )

    trainloader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,
    )
    testloader = DataLoader(
        TensorDataset(X_te_t, y_te_t),
        batch_size=batch_size,
    )
    return trainloader, testloader
