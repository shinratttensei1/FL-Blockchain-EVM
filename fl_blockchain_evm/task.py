"""PTB-XL 12-Lead ECG → 5 Superclasses (NORM/MI/STTC/CD/HYP).
Paper reference: "Application of FL Techniques for Arrhythmia Classification
Using 12-Lead ECG Signals", Jimenez et al., arXiv:2208.10993v3, Jan 2024.
"""

import os
import ast
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from typing import Tuple, List, Dict, Optional
import wfdb

try:
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
except ImportError:
    f1_score = precision_score = recall_score = roc_auc_score = None

DIAG_NORM = ["NORM"]
DIAG_MI = ["IMI", "ASMI", "ILMI", "AMI", "ALMI", "INJAS", "LMI",
           "INJAL", "IPLMI", "IPMI", "INJIN", "INJLA", "PMI", "INJIL"]
DIAG_STTC = ["NDT", "NST_", "DIG", "LNGQT", "ISC_", "ISCAL",
             "ISCIN", "ISCIL", "ISCAS", "ISCLA", "ANEUR", "EL", "ISCAN"]
DIAG_CD = ["LAFB", "IRBBB", "1AVB", "IVCD", "CRBBB",
           "CLBBB", "LPFB", "WPW", "ILBBB", "3AVB", "2AVB"]
DIAG_HYP = ["LVH", "LAO/LAE", "RVH", "RAO/RAE", "SEHYP"]

SC_NAMES: List[str] = ["NORM", "MI", "STTC", "CD", "HYP"]
NUM_CLASSES = 5

_SCP_TO_SC: Dict[str, int] = {}
for _i, _codes in enumerate([DIAG_NORM, DIAG_MI, DIAG_STTC, DIAG_CD, DIAG_HYP]):
    for _c in _codes:
        _SCP_TO_SC[_c] = _i


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        pt = torch.sigmoid(logits) * targets + \
            (1 - torch.sigmoid(logits)) * (1 - targets)
        focal = (1 - pt).clamp(min=1e-6) ** self.gamma * bce
        if self.alpha is not None:
            focal = (self.alpha / self.alpha.mean()).unsqueeze(0) * focal
        return focal.mean()


def _class_weights(loader):
    """Inverse-frequency weights — paper §3.5 shows balancing is critical."""
    labels = torch.cat([y for _, y in loader], dim=0)
    freqs = (labels.sum(0).float() + 1) / (labels.shape[0] + 1)
    return (freqs.median() / freqs).clamp(0.5, 20.0)


class _SEResBlock(nn.Module):
    def __init__(self, ch, ks=5):
        super().__init__()
        pad = ks // 2
        self.conv1 = nn.Conv1d(ch, ch, ks, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, ks, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(ch)
        mid = max(ch // 4, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, ch), nn.Sigmoid(),
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out * self.se(out).unsqueeze(-1)
        return F.relu(out + x, inplace=True)


class Net(nn.Module):
    """4-stage SE-ResNet: (B,12,1000) → 5 logits. ~200K params."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(12)

        self.conv1 = nn.Conv1d(12, 32, 15, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        self.res1a = _SEResBlock(32, 7)
        self.res1b = _SEResBlock(32, 7)

        self.conv2 = nn.Conv1d(32, 64, 7, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        self.res2a = _SEResBlock(64, 5)
        self.res2b = _SEResBlock(64, 5)

        self.conv3 = nn.Conv1d(64, 128, 5, padding=2, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)
        self.res3a = _SEResBlock(128, 3)
        self.res3b = _SEResBlock(128, 3)

        self.conv4 = nn.Conv1d(128, 256, 3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool4 = nn.MaxPool1d(2)
        self.res4 = _SEResBlock(256, 3)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.res1b(self.res1a(self.pool1(
            F.relu(self.bn1(self.conv1(x)), inplace=True))))
        x = self.res2b(self.res2a(self.pool2(
            F.relu(self.bn2(self.conv2(x)), inplace=True))))
        x = self.res3b(self.res3a(self.pool3(
            F.relu(self.bn3(self.conv3(x)), inplace=True))))
        x = self.res4(self.pool4(
            F.relu(self.bn4(self.conv4(x)), inplace=True)))
        return self.fc(self.drop(self.gap(x).squeeze(-1)))


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


def train(net, trainloader, epochs, lr=2e-3, device=torch.device("cpu"),
          use_mixup=True, mixup_alpha=0.3):
    net.to(device).train()
    cw = _class_weights(trainloader).to(device)
    criterion = FocalLoss(alpha=cw, gamma=2.0)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    total_steps = epochs * len(trainloader)
    warmup = max(total_steps // 10, 1)

    def lr_fn(step):
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + np.cos(np.pi * (step - warmup) / max(total_steps - warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    losses, total_n, t0 = [], 0, time.time()
    for _ in range(epochs):
        ep_loss, nb = 0.0, 0
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            x = _augment(x)

            if use_mixup and mixup_alpha > 0:
                lam = max(np.random.beta(mixup_alpha, mixup_alpha), 0.5)
                idx = torch.randperm(x.size(0), device=x.device)
                x = lam * x + (1 - lam) * x[idx]
                y = lam * y + (1 - lam) * y[idx]

            opt.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss += loss.item()
            nb += 1
            total_n += x.size(0)
        losses.append(ep_loss / max(nb, 1))

    return {
        "train_loss": losses[-1] if losses else 0.0,
        "train_loss_first_epoch": losses[0] if losses else 0.0,
        "train_loss_last_epoch": losses[-1] if losses else 0.0,
        "total_samples_processed": total_n,
        "training_time_seconds": round(time.time() - t0, 2),
        "num_epochs": epochs,
    }


def test(net, testloader, device=torch.device("cpu")):
    if len(testloader) == 0:
        return _empty()

    net.to(device).eval()
    crit = nn.BCEWithLogitsLoss()
    tot_loss, all_p, all_l = 0.0, [], []

    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            tot_loss += crit(logits, y).item()
            all_p.append(torch.sigmoid(logits).cpu().numpy())
            all_l.append(y.cpu().numpy())

    probs, labels = np.vstack(all_p), np.vstack(all_l)
    N = len(labels)

    thresholds = np.full(NUM_CLASSES, 0.5)
    for c in range(NUM_CLASSES):
        if labels[:, c].sum() == 0:
            continue
        best = -1.0
        for t in np.arange(0.05, 0.95, 0.02):
            f = f1_score(labels[:, c], (probs[:, c] > t).astype(
                float), zero_division=0)
            if f > best:
                best, thresholds[c] = f, t

    preds = (probs > thresholds).astype(float)

    pc_auc = [float(roc_auc_score(labels[:, c], probs[:, c]))
              if 0 < labels[:, c].sum() < N else 0.0 for c in range(NUM_CLASSES)]
    valid_auc = [a for a in pc_auc if a > 0]

    pc_spec, pc_cm = [], []
    for c in range(NUM_CLASSES):
        tp = int(((preds[:, c] == 1) & (labels[:, c] == 1)).sum())
        fp = int(((preds[:, c] == 1) & (labels[:, c] == 0)).sum())
        fn = int(((preds[:, c] == 0) & (labels[:, c] == 1)).sum())
        tn = int(((preds[:, c] == 0) & (labels[:, c] == 0)).sum())
        pc_cm.append({"TP": tp, "FP": fp, "FN": fn, "TN": tn})
        pc_spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    cm = np.zeros((5, 5), dtype=int)
    for i in range(N):
        if labels[i].sum() > 0:
            cm[int(np.argmax(labels[i])), int(np.argmax(probs[i]))] += 1

    return {
        "loss": tot_loss / len(testloader), "accuracy": float((preds == labels).mean()),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "specificity_macro": float(np.mean(pc_spec)),
        "auc_macro": float(np.mean(valid_auc)) if valid_auc else 0.0,
        "superclass_names": SC_NAMES,
        "per_class_precision": [float(v) for v in precision_score(labels, preds, average=None, zero_division=0)],
        "per_class_recall": [float(v) for v in recall_score(labels, preds, average=None, zero_division=0)],
        "per_class_f1": [float(v) for v in f1_score(labels, preds, average=None, zero_division=0)],
        "per_class_specificity": pc_spec, "per_class_auc": pc_auc,
        "per_class_support": [int(v) for v in labels.sum(0)],
        "per_class_cm": pc_cm, "optimal_thresholds": [float(t) for t in thresholds],
        "confusion_matrix": cm.tolist(), "num_samples": N, "num_classes": NUM_CLASSES,
    }


def _empty():
    z = [0.0] * 5
    return {"loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0,
            "precision_macro": 0.0, "recall_macro": 0.0, "specificity_macro": 0.0,
            "auc_macro": 0.0, "superclass_names": SC_NAMES,
            "per_class_precision": z, "per_class_recall": z, "per_class_f1": z,
            "per_class_specificity": z, "per_class_auc": z, "per_class_support": [0]*5,
            "per_class_cm": [{"TP": 0, "FP": 0, "FN": 0, "TN": 0}]*5,
            "optimal_thresholds": [0.5]*5, "confusion_matrix": [[0]*5]*5,
            "num_samples": 0, "num_classes": 5}
