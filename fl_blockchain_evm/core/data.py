"""PTB-XL data loading, caching, balancing, and augmentation.

Handles the full data pipeline from raw PTB-XL wfdb records to
PyTorch DataLoaders with class-balanced sampling.
"""

import os
import ast

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from typing import Tuple, Dict
import wfdb

from fl_blockchain_evm.core.constants import NUM_CLASSES, SC_NAMES, _SCP_TO_SC


# ── Class balancing (ROS+RUS) ─────────────────────────────────

def _balance_ros_rus(X, y, beta=1.0):
    """ROS+RUS per Jimenez et al. 2024, §3.5.

    For each class:
      - If under-represented (count < target): ROS (duplicate samples)
      - If over-represented (count > target): RUS (subsample)
    Target = m_s + τ where τ = (m_l - m_s) · β

    With β=1.0: target = m_l (full equalization).
    """
    counts = y.sum(0).astype(int)
    freqs = counts / (counts.sum() + 1e-8)
    primary = np.array([
        np.where(row == 1)[0][np.argmin(freqs[np.where(row == 1)[0]])]
        if row.sum() > 0 else 0 for row in y
    ])

    pc_counts = np.array([np.sum(primary == c) for c in range(NUM_CLASSES)])
    active_counts = pc_counts[pc_counts > 0]

    if len(active_counts) == 0:
        return X, y

    m_l = int(active_counts.max())
    m_s = int(active_counts.min())

    tau = (m_l - m_s) * beta
    target = int(m_s + tau)

    print(f"  [ROS+RUS] m_l={m_l}, m_s={m_s}, β={beta}, target={target}")
    print(f"  [ROS+RUS] Before: {dict(zip(SC_NAMES, pc_counts))}")

    bX, by = [], []
    for c in range(NUM_CLASSES):
        idx = np.where(primary == c)[0]
        if len(idx) == 0:
            continue
        n = len(idx)
        if n < target:
            idx = np.random.choice(idx, target, replace=True)
        elif n > target:
            idx = np.random.choice(idx, target, replace=False)
        bX.append(X[idx])
        by.append(y[idx])

    bX, by = np.concatenate(bX), np.concatenate(by)
    perm = np.random.permutation(len(bX))

    result_counts = by[perm].sum(0).astype(int)
    print(f"  [ROS+RUS] After:  {dict(zip(SC_NAMES, result_counts))}")
    return bX[perm], by[perm]


# ── Data augmentation ─────────────────────────────────────────

def _augment(x):
    # Gaussian noise (simulates sensor noise in IoT devices)
    if torch.rand(1).item() < 0.8:
        x = x + torch.randn_like(x) * 0.05

    # Per-lead amplitude scaling (simulates electrode impedance variation)
    if torch.rand(1).item() < 0.5:
        scale = 0.8 + 0.4 * \
            torch.rand(x.size(0), x.size(1), 1, device=x.device)
        x = x * scale

    # Temporal shift ~30 samples (simulates timing variation)
    if torch.rand(1).item() < 0.5:
        shift = torch.randint(-30, 31, (1,)).item()
        if shift > 0:
            x = F.pad(x[:, :, shift:], (0, shift))
        elif shift < 0:
            x = F.pad(x[:, :, :shift], (-shift, 0))

    # Baseline wander (low-freq sine — common ECG artifact)
    if torch.rand(1).item() < 0.3:
        freq = 0.1 + 0.4 * torch.rand(1).item()
        t = torch.linspace(0, 10*freq*6.283, x.size(2), device=x.device)
        amp = 0.1 * torch.rand(x.size(0), x.size(1), 1, device=x.device)
        x = x + amp * torch.sin(t).unsqueeze(0).unsqueeze(0)

    return x


# ── DataFrame and signal cache ────────────────────────────────

_DF_CACHE: Dict = {}


def _load_df(data_dir):
    if "df" not in _DF_CACHE:
        df = pd.read_csv(os.path.join(
            data_dir, "ptbxl_database.csv"), index_col="ecg_id")
        df.scp_codes = df.scp_codes.apply(ast.literal_eval)
        _DF_CACHE["df"] = df
    return _DF_CACHE["df"]


def _build_cache(data_dir):
    cdir = os.path.join(data_dir, "_cache_5class_v4")
    paths = [os.path.join(cdir, f)
             for f in ("ids.npy", "signals.npy", "labels.npy")]
    if all(os.path.exists(p) for p in paths):
        return [np.load(p) for p in paths]

    df = _load_df(data_dir)
    ids, sigs, labs = [], [], []
    for i, (idx, row) in enumerate(df.iterrows()):
        sig, _ = wfdb.rdsamp(os.path.join(data_dir, row["filename_lr"]))
        lbl = np.zeros(NUM_CLASSES, dtype=np.float32)
        has = False
        for code, confidence in row["scp_codes"].items():
            if code in _SCP_TO_SC and float(confidence) > 0:
                lbl[_SCP_TO_SC[code]] = 1.0
                has = True
        if not has:
            continue
        ids.append(idx)
        sigs.append(sig.T.astype(np.float32))
        labs.append(lbl)
        if (i + 1) % 5000 == 0:
            print(f"  [CACHE] {i+1}/{len(df)}...")

    ids, sigs, labs = np.array(ids), np.array(sigs), np.array(labs)
    os.makedirs(cdir, exist_ok=True)
    for p, arr in zip(paths, [ids, sigs, labs]):
        np.save(p, arr)
    print(f"  [CACHE] Saved {len(ids)} records")
    return ids, sigs, labs


# ── DataLoader construction ───────────────────────────────────

def load_data(partition_id: int, num_partitions: int, beta: float = 1.0,
              batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    data_dir = "data/ptb-xl"
    all_ids, all_sigs, all_labs = _build_cache(data_dir)
    folds = _load_df(data_dir).loc[all_ids, "strat_fold"].values

    train_idx = np.where(folds <= 8)[0]
    test_idx = np.where(folds >= 9)[0]

    np.random.seed(42)
    np.random.shuffle(train_idx)
    chunk = len(train_idx) // num_partitions
    start = partition_id * chunk
    end = start + chunk if partition_id < num_partitions - \
        1 else len(train_idx)
    my = train_idx[start:end]

    X_tr, y_tr = all_sigs[my].copy(), all_labs[my].copy()
    X_te, y_te = all_sigs[test_idx].copy(), all_labs[test_idx].copy()

    mu = X_tr.mean(axis=(0, 2), keepdims=True)
    sd = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
    X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd

    if beta > 0:
        X_tr, y_tr = _balance_ros_rus(X_tr, y_tr, beta=beta)

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)

    primary_classes = y_tr_t.argmax(dim=1).numpy()
    class_counts = np.bincount(
        primary_classes, minlength=NUM_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    sample_weights = 1.0 / class_counts[primary_classes]
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
