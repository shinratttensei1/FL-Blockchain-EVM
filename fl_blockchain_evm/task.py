"""FL-Blockchain-EVM: Core ML Task — PTB-XL 12-Lead ECG Classification.

Implements:
  - **All 71 SCP statement codes** (44 diagnostic + 19 form + 12 rhythm)
  - Multi-label Focal Loss for class imbalance
  - Residual 1D-CNN architecture for 12-lead ECG
  - ROS+RUS hybrid balancing (Jimenez et al., 2024)
  - Comprehensive per-class and aggregate metrics
"""

import os
import ast
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import wfdb
from typing import Tuple, List, Dict
# В начало task.py
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

try:
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, roc_auc_score,
    )
except ImportError:
    f1_score = None
    precision_score = None
    recall_score = None
    roc_auc_score = None

# ── 44 Diagnostic codes (grouped by superclass) ──
DIAG_NORM = ['NORM']
DIAG_MI = ['IMI', 'ASMI', 'ILMI', 'AMI', 'ALMI', 'INJAS', 'LMI',
           'INJAL', 'IPLMI', 'IPMI', 'INJIN', 'INJLA', 'PMI', 'INJIL']
DIAG_STTC = ['NDT', 'NST_', 'DIG', 'LNGQT', 'ISC_', 'ISCAL', 'ISCIN',
             'ISCIL', 'ISCAS', 'ISCLA', 'ANEUR', 'EL', 'ISCAN']
DIAG_CD = ['LAFB', 'IRBBB', '1AVB', 'IVCD', 'CRBBB', 'CLBBB', 'LPFB',
           'WPW', 'ILBBB', '3AVB', '2AVB']
DIAG_HYP = ['LVH', 'LAO/LAE', 'RVH', 'RAO/RAE', 'SEHYP']

# ── 19 Form codes (morphology)  — 4 overlap with diagnostic ──
FORM_CODES = ['ABQRS', 'PVC', 'STD_', 'VCLVH', 'QWAVE', 'LOWT', 'NT_',
              'PAC', 'LPR', 'INVT', 'LVOLT', 'HVOLT', 'TAB_', 'STE_', 'PRC(S)',
              'NDT', 'NST_', 'DIG', 'LNGQT']   # overlap

# ── 12 Rhythm codes ──
RHYTHM_CODES = ['SR', 'AFIB', 'STACH', 'SARRH', 'SBRAD', 'PACE',
                'SVARR', 'BIGU', 'AFLT', 'SVTAC', 'PSVT', 'TRIGU']

# Taxonomy variables are now defined above their usage

# ──────────────────────────────────────────────────────────────────────
# Hybrid ROS+RUS Balancing Function
# ──────────────────────────────────────────────────────────────────────
def apply_ros_rus_balancing(X, y):
    """
    Гибридная балансировка ROS + RUS.
    X: (N, 12, 1000)
    y: (N, 5) - One-hot encoded superclasses
    """
    print(f"Балансировка: исходный размер {X.shape}")

    # 1. Превращаем y (one-hot) в одиночные метки для балансировщика
    # Берем argmax, считая, что у каждого примера есть основной класс
    y_labels = np.argmax(y, axis=1)

    # 2. Flatten X для imblearn: (N, 12, 1000) -> (N, 12000)
    n_samples, n_chn, n_len = X.shape
    X_flat = X.reshape(n_samples, -1)

    # --- Ограничение: не больше 3000 примеров на класс ---
    unique, counts = np.unique(y_labels, return_counts=True)
    target_count = min(max(counts), 3000)
    sampling_strategy = {k: max(v, target_count)
                         for k, v in zip(unique, counts)}
    ros = RandomOverSampler(
        sampling_strategy=sampling_strategy, random_state=42)
    X_res, y_res_labels = ros.fit_resample(X_flat, y_labels)

    # 4. (Опционально) RUS (Under-sampling), если данных стало слишком много
    # rus = RandomUnderSampler(random_state=42)
    # X_res, y_res_labels = rus.fit_resample(X_res, y_res_labels)

    # 5. Возвращаем форму обратно (N_new, 12, 1000)
    X_final = X_res.reshape(-1, n_chn, n_len)

    # 6. Восстанавливаем One-Hot для y
    # Создаем массив нулей
    # Используем dtype=np.float32, чтобы PyTorch сразу понимал, что это Float
    y_final = np.zeros((len(y_res_labels), y.shape[1]), dtype=np.float32)
    # Ставим 1 в нужные места
    y_final[np.arange(len(y_res_labels)), y_res_labels] = 1

    print(f"Балансировка завершена: новый размер {X_final.shape}")
    return X_final, y_final


# ──────────────────────────────────────────────────────────────────────
# ResidualBlock and Net Model Classes
_all_diag = DIAG_NORM + DIAG_MI + DIAG_STTC + DIAG_CD + DIAG_HYP  # 44
_form_only = [c for c in FORM_CODES if c not in _all_diag]           # 15
_rhythm_only = RHYTHM_CODES                                            # 12

ALL_SCP_CODES: List[str] = _all_diag + _form_only + _rhythm_only      # 71
NUM_CLASSES = len(ALL_SCP_CODES)                                     # 71
CODE_TO_IDX = {code: i for i, code in enumerate(ALL_SCP_CODES)}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=15,
                               stride=stride, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=15,
                               stride=1, padding=7, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Net(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(Net, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(12, 64, kernel_size=15,
                               stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(ResidualBlock(self.inplanes, planes, stride))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ══════════════════════════════════════════════════════════════════════
# Full PTB-XL label taxonomy — all 71 SCP statement codes
# ══════════════════════════════════════════════════════════════════════


# ── 44 Diagnostic codes (grouped by superclass) ──
DIAG_NORM = ['NORM']
DIAG_MI = ['IMI', 'ASMI', 'ILMI', 'AMI', 'ALMI', 'INJAS', 'LMI',
           'INJAL', 'IPLMI', 'IPMI', 'INJIN', 'INJLA', 'PMI', 'INJIL']
DIAG_STTC = ['NDT', 'NST_', 'DIG', 'LNGQT', 'ISC_', 'ISCAL', 'ISCIN',
             'ISCIL', 'ISCAS', 'ISCLA', 'ANEUR', 'EL', 'ISCAN']
DIAG_CD = ['LAFB', 'IRBBB', '1AVB', 'IVCD', 'CRBBB', 'CLBBB', 'LPFB',
           'WPW', 'ILBBB', '3AVB', '2AVB']
DIAG_HYP = ['LVH', 'LAO/LAE', 'RVH', 'RAO/RAE', 'SEHYP']

# ── 19 Form codes (morphology)  — 4 overlap with diagnostic ──
FORM_CODES = ['ABQRS', 'PVC', 'STD_', 'VCLVH', 'QWAVE', 'LOWT', 'NT_',
              'PAC', 'LPR', 'INVT', 'LVOLT', 'HVOLT', 'TAB_', 'STE_',
              'PRC(S)',
              'NDT', 'NST_', 'DIG', 'LNGQT']   # overlap

# ── 12 Rhythm codes ──
RHYTHM_CODES = ['SR', 'AFIB', 'STACH', 'SARRH', 'SBRAD', 'PACE',
                'SVARR', 'BIGU', 'AFLT', 'SVTAC', 'PSVT', 'TRIGU']

_all_diag = DIAG_NORM + DIAG_MI + DIAG_STTC + DIAG_CD + DIAG_HYP  # 44
_form_only = [c for c in FORM_CODES if c not in _all_diag]           # 15
_rhythm_only = RHYTHM_CODES                                            # 12

ALL_SCP_CODES: List[str] = _all_diag + _form_only + _rhythm_only      # 71
NUM_CLASSES = len(ALL_SCP_CODES)                                     # 71
CODE_TO_IDX = {code: i for i, code in enumerate(ALL_SCP_CODES)}

N_DIAG = len(_all_diag)     # 44
N_FORM = len(_form_only)    # 15
N_RHYTHM = len(_rhythm_only)  # 12

SUPERCLASSES = ALL_SCP_CODES

SUPERCLASS_MAP: Dict[str, str] = {}
for c in DIAG_NORM:
    SUPERCLASS_MAP[c] = 'NORM'
for c in DIAG_MI:
    SUPERCLASS_MAP[c] = 'MI'
for c in DIAG_STTC:
    SUPERCLASS_MAP[c] = 'STTC'
for c in DIAG_CD:
    SUPERCLASS_MAP[c] = 'CD'
for c in DIAG_HYP:
    SUPERCLASS_MAP[c] = 'HYP'
for c in _form_only:
    SUPERCLASS_MAP[c] = 'FORM'
for c in _rhythm_only:
    SUPERCLASS_MAP[c] = 'RHYTHM'

SC_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
SC_GROUPS = {
    'NORM': [CODE_TO_IDX[c] for c in DIAG_NORM],
    'MI':   [CODE_TO_IDX[c] for c in DIAG_MI],
    'STTC': [CODE_TO_IDX[c] for c in DIAG_STTC],
    'CD':   [CODE_TO_IDX[c] for c in DIAG_CD],
    'HYP':  [CODE_TO_IDX[c] for c in DIAG_HYP],
}


# ══════════════════════════════════════════════════════════════════════
# 1. Multi-Label Focal Loss
# ══════════════════════════════════════════════════════════════════════


class FocalLoss(nn.Module):
    """Per-class Weighted Focal Loss for multi-label classification.

    Combines focal modulation (Lin et al., 2017) with per-class inverse-
    frequency weighting so that rare SCP codes receive much stronger
    gradient signal than majority codes like SR or NORM.

    Args:
        alpha: Per-class weight tensor of shape (num_classes,).
               If None, falls back to uniform weighting.
               Recommended: median_freq / per_class_freq (clamped).
        gamma: Focal exponent — higher values focus more on hard examples.
    """

    def __init__(self, alpha=None, gamma: float = 2.0):
        super().__init__()
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Numerically stable multi-label focal loss with per-class alpha.
        - inputs: logits, shape (batch, num_classes)
        - targets: binary, shape (batch, num_classes)
        """
        # BCE with logits (numerically stable)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        # Probabilities for positive class
        probs = torch.sigmoid(inputs)
        # pt: prob of true class (for each label)
        pt = probs * targets + (1 - probs) * (1 - targets)
        # Focal modulation
        focal_mod = (1 - pt).clamp(min=1e-6, max=1.0) ** self.gamma
        loss = focal_mod * bce_loss
        # Normalize alpha to sum to num_classes (optional, for stability)
        if self.alpha is not None:
            alpha = self.alpha
            alpha = alpha / (alpha.sum() / alpha.numel())
            loss = alpha.unsqueeze(0) * loss
        return loss.mean()


def compute_class_weights(trainloader: 'DataLoader') -> torch.Tensor:
    """Compute inverse-frequency class weights from a DataLoader.

    Returns a (NUM_CLASSES,) tensor where:
        w_c = median_frequency / frequency_c    (clamped to [0.5, 10])

    This gives rare classes up to 10× the gradient weight of common ones,
    while preventing extreme weights for ultra-rare codes (< 5 samples).
    """
    all_labels = []
    for _, labels in trainloader:
        all_labels.append(labels)
    all_labels = torch.cat(all_labels, dim=0)  # (N, 71)
    n_samples = all_labels.shape[0]

    # Per-class positive count
    pos_counts = all_labels.sum(dim=0).float()  # (71,)

    # Frequencies (add smoothing to avoid division by zero)
    freqs = (pos_counts + 1.0) / (n_samples + 1.0)
    median_freq = freqs.median()

    # Inverse-frequency weights, clamped to prevent extremes
    weights = (median_freq / freqs).clamp(min=1.0, max=3.0)
    return weights


# ══════════════════════════════════════════════════════════════════════
# 2. 12-Lead 1D-CNN Architecture
# ══════════════════════════════════════════════════════════════════════


class _ResBlock(nn.Module):
    """Lightweight 1D residual block with skip connection."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class Net(nn.Module):
    """Residual 1D-CNN for 12-lead ECG — all 71 SCP codes.

    Input:  (B, 12, 1000)  →  12 leads, 10 s @ 100 Hz
    Flow:   InputBN → Conv-Pool → ResBlock → Conv-Pool → ResBlock → GAP → FC
    Output: 71 logits (one per SCP code)
    ~53 K parameters — fast on CPU, realistic for IoT / edge hardware.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(12)

        self.conv1 = nn.Conv1d(12, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)

        self.res1 = _ResBlock(32, kernel_size=5)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)

        self.res2 = _ResBlock(64, kernel_size=3)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)   # 64 → 71

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.res1(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.res2(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


# ══════════════════════════════════════════════════════════════════════
# 3. ROS + RUS Hybrid Balancing  (Jimenez et al., 2024)
# ══════════════════════════════════════════════════════════════════════


def balance_dataset_ros_rus(
    X_list: List[np.ndarray],
    y_list: List[np.ndarray],
    beta: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Hybrid Random Over-Sampling + Random Under-Sampling.

    τ = (m_l − m_s) · β   controls the degree of re-balancing.

    For 71 labels, many codes are very rare, so we only balance primary
    class assignments and leave very-rare classes to benefit from Focal Loss.

    The target_size is capped at the median class count to prevent the
    balanced dataset from exploding when the majority class (e.g. SR) is
    far larger than most classes.
    """
    X_np = np.array(X_list)
    y_np = np.array(y_list)

    class_counts = y_np.sum(axis=0).astype(int)
    active_mask = class_counts > 0
    if active_mask.sum() == 0:
        return X_np, y_np

    active_counts = class_counts[active_mask]
    ml = int(active_counts.max())
    ms = int(active_counts[active_counts > 0].min())

    target_size = int(ms + (ml - ms) * beta)
    target_size = max(target_size, 1)

    # Cap target_size at the 90th percentile of active class counts
    # to give rare classes more over-sampling while still preventing
    # the balanced dataset from exploding.
    p90_count = int(np.percentile(active_counts, 90))
    target_size = min(target_size, max(p90_count, ms + 1))

    # Assign each sample to its rarest positive class
    class_frequencies = class_counts / (class_counts.sum() + 1e-8)
    sample_primary_class = []
    for labels in y_np:
        pos = np.where(labels == 1)[0]
        if len(pos) > 0:
            rarest = pos[np.argmin(class_frequencies[pos])]
            sample_primary_class.append(rarest)
        else:
            sample_primary_class.append(0)
    sample_primary_class = np.array(sample_primary_class)

    balanced_X, balanced_y = [], []
    for c in range(NUM_CLASSES):
        c_idx = np.where(sample_primary_class == c)[0]
        if len(c_idx) == 0:
            continue
        n = len(c_idx)
        if n < target_size:
            resampled = np.random.choice(c_idx, size=target_size, replace=True)
        elif n > target_size:
            resampled = np.random.choice(
                c_idx, size=target_size, replace=False)
        else:
            resampled = c_idx
        balanced_X.extend(X_np[resampled])
        balanced_y.extend(y_np[resampled])

    balanced_X = np.array(balanced_X)
    balanced_y = np.array(balanced_y)
    perm = np.random.permutation(len(balanced_X))
    return balanced_X[perm], balanced_y[perm]


# ══════════════════════════════════════════════════════════════════════
# 4. PTB-XL Data Loader  — ALL 71 SCP codes (with disk cache)
# ══════════════════════════════════════════════════════════════════════

# In-memory cache so that each worker loads signals only once per process
_SIGNAL_CACHE: Dict[int, np.ndarray] = {}
_LABEL_CACHE: Dict[int, np.ndarray] = {}
_DF_CACHE = {}


def _load_df(data_dir: str):
    """Load and parse PTB-XL CSV (cached)."""
    if "df" not in _DF_CACHE:
        csv_path = os.path.join(data_dir, "ptbxl_database.csv")
        df = pd.read_csv(csv_path, index_col='ecg_id')
        df.scp_codes = df.scp_codes.apply(ast.literal_eval)
        _DF_CACHE["df"] = df
    return _DF_CACHE["df"]


def _get_disk_cache_path(data_dir: str) -> str:
    """Return base path for the preprocessed cache directory."""
    return os.path.join(data_dir, "_cache_71class")


def _build_full_cache(data_dir: str):
    """Build cached .npy files of all signals + labels to avoid repeated wfdb reads."""
    cache_dir = _get_disk_cache_path(data_dir)
    ids_path = os.path.join(cache_dir, "ids.npy")
    sig_path = os.path.join(cache_dir, "signals.npy")
    lab_path = os.path.join(cache_dir, "labels.npy")

    if os.path.exists(ids_path):
        return np.load(ids_path), np.load(sig_path), np.load(lab_path)

    df = _load_df(data_dir)
    ids_list, sig_list, lab_list = [], [], []

    total = len(df)
    print(f"  [CACHE] Building preprocessed cache from {total} wfdb files...")
    for i, (idx, row) in enumerate(df.iterrows()):
        fpath = os.path.join(data_dir, row["filename_lr"])
        signal, _ = wfdb.rdsamp(fpath)

        label_vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        scp_dict = row["scp_codes"]
        has_any_label = False
        for code in scp_dict:
            if code in CODE_TO_IDX:
                label_vec[CODE_TO_IDX[code]] = 1.0
                has_any_label = True

        if not has_any_label:
            continue

        ids_list.append(idx)
        sig_list.append(signal.T.astype(np.float32))
        lab_list.append(label_vec)

        if (i + 1) % 5000 == 0:
            print(f"  [CACHE] Processed {i+1}/{total} records...")

    ids_arr = np.array(ids_list)
    sig_arr = np.array(sig_list)
    lab_arr = np.array(lab_list)

    os.makedirs(cache_dir, exist_ok=True)
    np.save(ids_path, ids_arr)
    np.save(sig_path, sig_arr)
    np.save(lab_path, lab_arr)
    print(f"  [CACHE] Saved to {cache_dir}/ "
          f"({len(ids_arr)} records, {sig_arr.nbytes / 1e6:.1f} MB)")
    return ids_arr, sig_arr, lab_arr


def z_score_normalization(X):
    """
    Нормализация сигнала (Z-score) для каждого отведения отдельно.
    X shape: (N_samples, 12, 1000)
    """
    # Если X — torch.Tensor, переводим в numpy
    is_torch = False
    if isinstance(X, torch.Tensor):
        X_np = X.numpy()
        is_torch = True
    else:
        X_np = X
    # Вычисляем среднее и станд. отклонение по оси времени (axis=2)
    mean = np.mean(X_np, axis=2, keepdims=True)
    std = np.std(X_np, axis=2, keepdims=True)
    X_norm = (X_np - mean) / (std + 1e-8)
    # Возвращаем в исходном типе
    if is_torch:
        return torch.from_numpy(X_norm).type_as(X)
    else:
        return X_norm


def load_data(
    partition_id: int,
    num_partitions: int,
    beta: float = 0.2,
    ) -> Tuple[DataLoader, DataLoader]:
    """Load PTB-XL and partition for federated simulation (benchmark style).

    Uses disk cache. No ROS/RUS balancing. Global normalization (train mean/std).
    """
    data_dir = "data/ptb-xl"
    all_ids, all_signals, all_labels = _build_full_cache(data_dir)
    df = _load_df(data_dir)

    folds = df.loc[all_ids, "strat_fold"].values
    train_indices = np.where(folds <= 8)[0]
    test_indices = np.where(folds >= 9)[0]

    np.random.seed(42)
    np.random.shuffle(train_indices)
    size = len(train_indices) // num_partitions
    my_train_idx = train_indices[partition_id *
                                 size: (partition_id + 1) * size]

    X_train = torch.tensor(all_signals[my_train_idx], dtype=torch.float32)
    y_train = torch.tensor(all_labels[my_train_idx], dtype=torch.float32)
    X_test = torch.tensor(all_signals[test_indices], dtype=torch.float32)
    y_test = torch.tensor(all_labels[test_indices], dtype=torch.float32)

    # Global Z-Score normalization (benchmark style)
    mean = X_train.mean()
    std = X_train.std()
    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    return (
        DataLoader(TensorDataset(X_train, y_train),
                   batch_size=64, shuffle=True),
        DataLoader(TensorDataset(X_test, y_test), batch_size=64)
    )


# ══════════════════════════════════════════════════════════════════════
# 5. Training
# ══════════════════════════════════════════════════════════════════════


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Dict[str, float]:
    """Train local model and return rich training metrics."""
    net.to(device)

    # ── Compute per-class inverse-frequency weights ──
    class_weights = compute_class_weights(trainloader).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(device)

    # No LR scheduler — in FL, a fresh scheduler each round is harmful
    # because the optimizer is re-created from scratch every round.
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

    net.train()
    epoch_losses = []
    total_samples = 0
    t_start = time.time()

    for _epoch in range(epochs):
        running_loss = 0.0
        n_batches = 0
        for signals, labels in trainloader:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            total_samples += signals.size(0)
        avg_loss = running_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)

    train_time = time.time() - t_start
    return {
        "train_loss": float(epoch_losses[-1]) if epoch_losses else 0.0,
        "train_loss_first_epoch": float(epoch_losses[0]) if epoch_losses else 0.0,
        "train_loss_last_epoch": float(epoch_losses[-1]) if epoch_losses else 0.0,
        "total_samples_processed": total_samples,
        "training_time_seconds": float(train_time),
        "num_epochs": epochs,
    }


# ══════════════════════════════════════════════════════════════════════
# 6. Testing / Evaluation  — comprehensive metrics for 71 classes
# ══════════════════════════════════════════════════════════════════════


def test(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    """Evaluate model and return a rich metrics dictionary.

    Metrics span two granularities:
      1. **71-class fine-grained** — per SCP-code P / R / F1 / Spec / AUC
      2. **5-class superclass**   — NORM / MI / STTC / CD / HYP aggregates
                                     + 5×5 confusion matrix
    """
    if len(testloader) == 0:
        return _empty_metrics()

    net.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    total_loss = 0.0

    all_probs, all_preds, all_labels = [], [], []

    net.eval()
    with torch.no_grad():
        for signals, labels in testloader:
            signals, labels = signals.to(device), labels.to(device)
            logits = net(signals)
            total_loss += criterion(logits, labels).item()

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    n_samples = len(all_labels)
    avg_loss = total_loss / len(testloader)

    # Per-class thresholding
    best_thresholds = np.full(NUM_CLASSES, 0.3)
    for c in range(NUM_CLASSES):
        if ALL_SCP_CODES[c] in DIAG_MI or ALL_SCP_CODES[c] == 'NORM':
            best_thresholds[c] = 0.5
        else:
            best_thresholds[c] = 0.2
    all_preds = (all_probs > best_thresholds).astype(float)

    # ── Aggregate metrics (macro over all 71 classes) ──
    # Hamming accuracy: fraction of correctly predicted labels across
    # all samples and all 71 classes — the standard multi-label metric.
    # (exact-match accuracy is nearly 0 with 71 labels and is misleading)
    accuracy = float((all_preds == all_labels).mean())
    f1_macro = f1_score(all_labels, all_preds,
                        average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds,
                           average='weighted', zero_division=0)
    prec_macro = precision_score(
        all_labels, all_preds, average='macro', zero_division=0)
    rec_macro = recall_score(all_labels, all_preds,
                             average='macro', zero_division=0)

    # ── AUC-ROC per class, then macro ──
    try:
        per_class_auc = []
        for c in range(NUM_CLASSES):
            if 0 < all_labels[:, c].sum() < n_samples:
                per_class_auc.append(
                    roc_auc_score(all_labels[:, c], all_probs[:, c]))
            else:
                per_class_auc.append(0.0)
        valid = [a for a in per_class_auc if a > 0]
        auc_macro = float(np.mean(valid)) if valid else 0.0
    except Exception:
        per_class_auc = [0.0] * NUM_CLASSES
        auc_macro = 0.0

    # ── Per-class P / R / F1 / Spec / Support ──
    pc_prec = precision_score(all_labels, all_preds,
                              average=None, zero_division=0)
    pc_rec = recall_score(all_labels, all_preds, average=None, zero_division=0)
    pc_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    pc_support = all_labels.sum(axis=0).astype(int)

    pc_spec = []
    for c in range(NUM_CLASSES):
        tn = int(((all_preds[:, c] == 0) & (all_labels[:, c] == 0)).sum())
        fp = int(((all_preds[:, c] == 1) & (all_labels[:, c] == 0)).sum())
        pc_spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    specificity_macro = float(np.mean(pc_spec))

    # ── Per-class TP/FP/FN/TN ──
    cm_per_class = []
    for c in range(NUM_CLASSES):
        tp = int(((all_preds[:, c] == 1) & (all_labels[:, c] == 1)).sum())
        fp = int(((all_preds[:, c] == 1) & (all_labels[:, c] == 0)).sum())
        fn = int(((all_preds[:, c] == 0) & (all_labels[:, c] == 1)).sum())
        tn = int(((all_preds[:, c] == 0) & (all_labels[:, c] == 0)).sum())
        cm_per_class.append({"TP": tp, "FP": fp, "FN": fn, "TN": tn})

    # ══════════════════════════════════════════════════════════════════
    # Superclass-level aggregates (5 diagnostic super-classes)
    # ══════════════════════════════════════════════════════════════════
    sc_f1, sc_prec, sc_rec, sc_auc, sc_support = [], [], [], [], []
    for sc_name in SC_NAMES:
        idxs = SC_GROUPS[sc_name]
        sc_true = (all_labels[:, idxs].sum(axis=1) > 0).astype(float)
        sc_pred = (all_preds[:, idxs].sum(axis=1) > 0).astype(float)
        sc_prob = all_probs[:, idxs].max(axis=1)

        sc_f1.append(float(f1_score(sc_true, sc_pred, zero_division=0)))
        sc_prec.append(float(precision_score(
            sc_true, sc_pred, zero_division=0)))
        sc_rec.append(float(recall_score(sc_true, sc_pred, zero_division=0)))
        sc_support.append(int(sc_true.sum()))
        try:
            if 0 < sc_true.sum() < n_samples:
                sc_auc.append(float(roc_auc_score(sc_true, sc_prob)))
            else:
                sc_auc.append(0.0)
        except Exception:
            sc_auc.append(0.0)

    # ── 5×5 Superclass Confusion Matrix ──
    # Aggregate fine-grained predictions back to 5 diagnostic superclasses.
    diag_idxs_flat = []
    for idxs in SC_GROUPS.values():
        diag_idxs_flat.extend(idxs)

    cm_5x5 = np.zeros((5, 5), dtype=int)
    for i in range(n_samples):
        if all_labels[i, diag_idxs_flat].sum() == 0:
            continue  # skip non-diagnostic samples

        # True: superclass with the highest summed ground-truth labels
        true_scores = [float(all_labels[i, SC_GROUPS[sc]].sum())
                       for sc in SC_NAMES]
        t = int(np.argmax(true_scores))

        # Pred: superclass with the highest summed probabilities
        pred_scores = [float(all_probs[i, SC_GROUPS[sc]].sum())
                       for sc in SC_NAMES]
        p = int(np.argmax(pred_scores))

        cm_5x5[t, p] += 1

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "specificity_macro": float(specificity_macro),
        "auc_macro": float(auc_macro),
        # Fine-grained: all 71 SCP codes
        "per_class_precision": [float(v) for v in pc_prec],
        "per_class_recall": [float(v) for v in pc_rec],
        "per_class_f1": [float(v) for v in pc_f1],
        "per_class_specificity": [float(v) for v in pc_spec],
        "per_class_auc": [float(v) for v in per_class_auc],
        "per_class_support": [int(v) for v in pc_support],
        "per_class_cm": cm_per_class,
        # Superclass-level (5 diagnostic classes)
        "superclass_names": SC_NAMES,
        "superclass_f1": sc_f1,
        "superclass_precision": sc_prec,
        "superclass_recall": sc_rec,
        "superclass_auc": sc_auc,
        "superclass_support": sc_support,
        # 5×5 confusion matrix
        "confusion_matrix": cm_5x5,
        "num_samples": n_samples,
        "num_classes": NUM_CLASSES,
    }


def _empty_metrics() -> Dict:
    """Return zeroed-out metrics when testloader is empty."""
    return {
        "loss": 0.0, "accuracy": 0.0,
        "f1_macro": 0.0, "f1_weighted": 0.0,
        "precision_macro": 0.0, "recall_macro": 0.0,
        "specificity_macro": 0.0, "auc_macro": 0.0,
        "per_class_precision": [0.0] * NUM_CLASSES,
        "per_class_recall": [0.0] * NUM_CLASSES,
        "per_class_f1": [0.0] * NUM_CLASSES,
        "per_class_specificity": [0.0] * NUM_CLASSES,
        "per_class_auc": [0.0] * NUM_CLASSES,
        "per_class_support": [0] * NUM_CLASSES,
        "per_class_cm": [{"TP": 0, "FP": 0, "FN": 0, "TN": 0}] * NUM_CLASSES,
        "superclass_names": SC_NAMES,
        "superclass_f1": [0.0] * 5,
        "superclass_precision": [0.0] * 5,
        "superclass_recall": [0.0] * 5,
        "superclass_auc": [0.0] * 5,
        "superclass_support": [0] * 5,
        "confusion_matrix": np.zeros((5, 5)),
        "num_samples": 0, "num_classes": NUM_CLASSES,
    }
