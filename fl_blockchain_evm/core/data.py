"""MHEALTH data loading, caching, balancing, and augmentation.

Handles the full data pipeline from raw MHEALTH Mobile Health sensor logs
to PyTorch DataLoaders with class-balanced sampling.

Dataset: MHEALTH (Mobile Health) — Banos et al., 2014
- 10 subjects, 12 physical activity classes + null class (label 0, excluded)
- 23 sensor channels (chest acc, ECG x2, left-ankle acc/gyro/mag,
  right-arm acc/gyro/mag), sampled at 50 Hz
- Expected at: data/MHEALTHDATASET/mHealth_subject<N>.log

FL partitioning: subjects are split across clients by subject ID,
producing a realistic non-IID federated scenario.

Sliding-window segmentation: 256-sample windows with 128-sample stride
(≈ 5.12 s windows at 50 % overlap).
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from fl_blockchain_evm.core.constants import (
    NUM_CLASSES, NUM_CHANNELS, SC_NAMES, WINDOW_SIZE, WINDOW_STEP,
)

# ── Dataset location ──────────────────────────────────────────
# FL_DATA_DIR env var lets the path be set explicitly at runtime
# (needed when Flower extracts the FAB to a temp dir on SuperNodes).
# Fallback: absolute path relative to this module's location so it
# works when the package is installed in-place on the laptop/Pi.
_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get(
    "FL_DATA_DIR",
    os.path.join(_MODULE_ROOT, "data", "MHEALTHDATASET"),
)

# Pre-processed numpy cache (avoids slow re-parsing on every run)
_NPY_CACHE_DIR = os.path.join(DATA_DIR, ".npy_cache")

# 8 subjects for training, 2 held out for testing
_TRAIN_SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 8]
_TEST_SUBJECTS  = [9, 10]

_CACHE: dict = {}


# ── Raw file loading ──────────────────────────────────────────

def _load_subject(subject_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw sensor data for one subject.

    Uses a .npy disk cache for fast reloads after the first run.
    pandas.read_csv is used for the initial parse (10-50x faster than
    np.loadtxt on files this size).

    Returns:
        data:   (T, 23) float32 -- raw sensor readings
        labels: (T,)   int32   -- per-sample activity label
    """
    cache_data   = os.path.join(_NPY_CACHE_DIR, f"s{subject_id}_data.npy")
    cache_labels = os.path.join(_NPY_CACHE_DIR, f"s{subject_id}_labels.npy")

    if os.path.exists(cache_data) and os.path.exists(cache_labels):
        return np.load(cache_data), np.load(cache_labels)

    path = os.path.join(DATA_DIR, f"mHealth_subject{subject_id}.log")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MHEALTH subject file not found: {path}\n"
            f"Please place the MHEALTHDATASET folder in {DATA_DIR}"
        )

    # pandas.read_csv is 10-50x faster than np.loadtxt for large text files
    print(f"  [MHEALTH] Parsing subject {subject_id} (first run — building cache)...")
    df     = pd.read_csv(path, header=None, sep=r"\s+", dtype=np.float32)
    raw    = df.values           # (T, 24)
    data   = raw[:, :23]
    labels = raw[:, 23].astype(np.int32)

    os.makedirs(_NPY_CACHE_DIR, exist_ok=True)
    np.save(cache_data,   data)
    np.save(cache_labels, labels)
    return data, labels


def _sliding_windows(data: np.ndarray, labels: np.ndarray,
                     win: int, step: int) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a sliding window over continuous sensor data.

    Windows where the majority label is the null class (0) are discarded.

    Returns:
        X: (N, C, win) float32 -- windowed sensor channels
        y: (N,)        int32   -- majority label for each window (0-indexed)
    """
    T, C = data.shape
    X_wins, y_wins = [], []
    for start in range(0, T - win + 1, step):
        w_data   = data[start:start + win]       # (win, C)
        w_labels = labels[start:start + win]     # (win,)
        counts   = np.bincount(w_labels, minlength=13)
        majority = int(np.argmax(counts))
        if majority == 0:
            continue  # skip null-class windows
        X_wins.append(w_data.T)           # (C, win)
        y_wins.append(majority - 1)       # convert 1-12 -> 0-11

    if not X_wins:
        return np.empty((0, C, win), dtype=np.float32), np.empty((0,), dtype=np.int32)
    return np.stack(X_wins).astype(np.float32), np.array(y_wins, dtype=np.int32)


def _load_subjects(subject_ids: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, window, and concatenate data for a list of subjects.

    Returns:
        X:        (N, 23, win) float32
        y:        (N,) int32  -- 0-indexed labels
        subjects: (N,) int32  -- subject IDs
    """
    Xs, ys, ss = [], [], []
    for sid in subject_ids:
        data, labels = _load_subject(sid)
        X_w, y_w     = _sliding_windows(data, labels, WINDOW_SIZE, WINDOW_STEP)
        if len(X_w) == 0:
            continue
        Xs.append(X_w)
        ys.append(y_w)
        ss.append(np.full(len(X_w), sid, dtype=np.int32))
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss))


def _get_data():
    """Load and cache train/test splits."""
    if "all" not in _CACHE:
        print(f"  [MHEALTH] Loading from {DATA_DIR} ...")
        X_tr, y_tr, s_tr = _load_subjects(_TRAIN_SUBJECTS)
        X_te, y_te, s_te = _load_subjects(_TEST_SUBJECTS)
        _CACHE["all"] = (X_tr, y_tr, s_tr, X_te, y_te, s_te)
        print(f"  [MHEALTH] Train: {len(X_tr)} windows | "
              f"Test: {len(X_te)} windows | "
              f"Channels: {NUM_CHANNELS} | Window: {WINDOW_SIZE} samples")
    return _CACHE["all"]


# ── Class balancing (ROS+RUS) ─────────────────────────────────

def _balance_ros_rus(X: np.ndarray, y: np.ndarray, beta: float = 1.0):
    """ROS+RUS balancing for one-hot encoded single-label data.

    y is one-hot (N, C). For each class:
      - Under-represented (count < target): oversample (ROS)
      - Over-represented  (count > target): undersample (RUS)
    Target = m_s + (m_l - m_s) * beta  [beta=1.0 -> full equalization]
    """
    primary = np.argmax(y, axis=1)

    pc_counts = np.array([np.sum(primary == c) for c in range(NUM_CLASSES)])
    active    = pc_counts[pc_counts > 0]

    if len(active) == 0:
        return X, y

    m_l, m_s = int(active.max()), int(active.min())
    target   = int(m_s + (m_l - m_s) * beta)

    print(f"  [ROS+RUS] m_l={m_l}, m_s={m_s}, beta={beta}, target={target}")
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
              batch_size: int = 64) -> Tuple[DataLoader, DataLoader]:
    """Build train/test DataLoaders for one FL client partition.

    Training data is partitioned by subject ID (non-IID), giving each
    client data from a disjoint set of subjects. Test data (subjects 9-10)
    is shared across all clients.

    Args:
        partition_id:    Client index (0 ... num_partitions-1).
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

    print(f"  [MHEALTH] Partition {partition_id}: "
          f"subjects={sorted(my_subjects)}, windows={len(X_part)}")

    # Use shared normalization stats across all train subjects by default.
    # Per-partition normalization can put each client into a different feature
    # space, which hurts FedAvg on strongly non-IID subject splits.
    use_global_norm = os.getenv("FL_GLOBAL_NORM", "1") == "1"
    if use_global_norm:
        mu = X_tr.mean(axis=(0, 2), keepdims=True)
        sd = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
    else:
        mu = X_part.mean(axis=(0, 2), keepdims=True)
        sd = X_part.std(axis=(0, 2), keepdims=True) + 1e-8

    X_part_n = (X_part - mu) / sd
    X_te_n   = (X_te   - mu) / sd

    print(f"  [MHEALTH] Normalization: {'global-train' if use_global_norm else 'partition-local'}")

    # One-hot encode labels
    y_part_oh = np.eye(NUM_CLASSES, dtype=np.float32)[y_part]   # (N, 12)
    y_te_oh   = np.eye(NUM_CLASSES, dtype=np.float32)[y_te]     # (M, 12)

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
