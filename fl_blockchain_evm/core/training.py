"""Training and evaluation functions for the MHEALTH activity model.

Contains the train loop (with FocalLoss, Mixup, cosine LR schedule)
and the single-label evaluation function with per-class metrics.

Labels are one-hot encoded (N, 12) throughout the pipeline for
compatibility with FocalLoss and the balancing utilities.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from fl_blockchain_evm.core.constants import NUM_CLASSES, SC_NAMES
from fl_blockchain_evm.core.data import _augment
from fl_blockchain_evm.core.model import FocalLoss


def _class_weights(loader):
    """Inverse-frequency class weights from one-hot labels."""
    labels = torch.cat([y for _, y in loader], dim=0)
    freqs  = (labels.sum(0).float() + 1) / (labels.shape[0] + 1)
    return (freqs.median() / freqs).clamp(0.5, 20.0)


def _ts():
    """Current timestamp string for log lines."""
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def train(net, trainloader, epochs, lr=2e-3, device=torch.device("cpu"),
          use_mixup=True, mixup_alpha=0.3):
    net.to(device).train()
    cw        = _class_weights(trainloader).to(device)
    criterion = FocalLoss(alpha=cw, gamma=2.0)
    opt       = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    total_steps = epochs * len(trainloader)
    warmup      = max(total_steps // 10, 1)
    n_batches   = len(trainloader)

    def lr_fn(step):
        if step < warmup:
            return step / warmup
        return 0.5 * (1 + np.cos(
            np.pi * (step - warmup) / max(total_steps - warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    print(f"[{_ts()}] [TRAIN] Starting: epochs={epochs}  "
          f"batches/epoch={n_batches}  lr={lr:.4f}  "
          f"device={device}  mixup={use_mixup}")

    losses, total_n, t0 = [], 0, time.time()
    global_step = 0

    for ep in range(epochs):
        ep_loss, nb = 0.0, 0
        ep_t0 = time.time()

        for batch_idx, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            x = _augment(x)

            if use_mixup and mixup_alpha > 0:
                lam = max(np.random.beta(mixup_alpha, mixup_alpha), 0.5)
                idx = torch.randperm(x.size(0), device=x.device)
                x   = lam * x + (1 - lam) * x[idx]
                y   = lam * y + (1 - lam) * y[idx]

            opt.zero_grad()
            loss = criterion(net(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_loss  += loss.item()
            nb       += 1
            total_n  += x.size(0)
            global_step += 1

            # Log batch progress at 0%, 25%, 50%, 75%, 100%
            checkpoints = {0, n_batches // 4, n_batches // 2,
                           3 * n_batches // 4, n_batches - 1}
            if batch_idx in checkpoints:
                pct = int(100 * (batch_idx + 1) / n_batches)
                cur_lr = opt.param_groups[0]["lr"]
                print(f"[{_ts()}] [TRAIN]   ep {ep+1:3d}/{epochs}  "
                      f"batch {batch_idx+1:4d}/{n_batches}  "
                      f"({pct:3d}%)  "
                      f"loss={ep_loss / max(nb, 1):.5f}  "
                      f"lr={cur_lr:.6f}")

        ep_loss_avg = ep_loss / max(nb, 1)
        ep_elapsed  = time.time() - ep_t0
        losses.append(ep_loss_avg)

        print(f"[{_ts()}] [TRAIN] ── Epoch {ep+1:3d}/{epochs} complete ──  "
              f"loss={ep_loss_avg:.5f}  "
              f"samples={total_n:,}  "
              f"time={ep_elapsed:.1f}s  "
              f"({ep_elapsed/max(nb,1)*1000:.1f}ms/batch)")

    total_time = round(time.time() - t0, 2)
    print(f"[{_ts()}] [TRAIN] ══ Training complete ══  "
          f"epochs={epochs}  "
          f"total_samples={total_n:,}  "
          f"total_time={total_time}s  "
          f"loss_first={losses[0]:.5f}  "
          f"loss_last={losses[-1]:.5f}  "
          f"improvement={losses[0]-losses[-1]:.5f}")

    return {
        "train_loss":              losses[-1] if losses else 0.0,
        "train_loss_first_epoch":  losses[0]  if losses else 0.0,
        "train_loss_last_epoch":   losses[-1] if losses else 0.0,
        "total_samples_processed": total_n,
        "training_time_seconds":   total_time,
        "num_epochs":              epochs,
    }


def test(net, testloader, device=torch.device("cpu")):
    if len(testloader) == 0:
        return _empty()

    net.to(device).eval()
    crit     = nn.CrossEntropyLoss()
    tot_loss = 0.0
    all_p, all_l = [], []

    with torch.no_grad():
        for x, y in testloader:
            x, y    = x.to(device), y.to(device)
            logits  = net(x)
            y_idx   = y.argmax(dim=1)          # one-hot → class index for CE
            tot_loss += crit(logits, y_idx).item()
            all_p.append(F.softmax(logits, dim=1).cpu().numpy())
            all_l.append(y.cpu().numpy())

    probs  = np.vstack(all_p)   # (N, 6)  softmax probabilities
    labels = np.vstack(all_l)   # (N, 6)  one-hot ground truth
    N      = len(labels)

    true_idx = labels.argmax(1)   # (N,)
    pred_idx = probs.argmax(1)    # (N,)

    # One-hot predictions for per-class sklearn metrics
    preds = np.eye(NUM_CLASSES, dtype=float)[pred_idx]  # (N, 6)

    # Per-class AUC (OvR)
    pc_auc = []
    for c in range(NUM_CLASSES):
        if 0 < labels[:, c].sum() < N:
            pc_auc.append(float(roc_auc_score(labels[:, c], probs[:, c])))
        else:
            pc_auc.append(0.0)
    valid_auc = [a for a in pc_auc if a > 0]

    # Per-class specificity and binary confusion stats
    pc_spec, pc_cm = [], []
    for c in range(NUM_CLASSES):
        tp = int(((preds[:, c] == 1) & (labels[:, c] == 1)).sum())
        fp = int(((preds[:, c] == 1) & (labels[:, c] == 0)).sum())
        fn = int(((preds[:, c] == 0) & (labels[:, c] == 1)).sum())
        tn = int(((preds[:, c] == 0) & (labels[:, c] == 0)).sum())
        pc_cm.append({"TP": tp, "FP": fp, "FN": fn, "TN": tn})
        pc_spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)

    # N×N confusion matrix (argmax-based, single-label)
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for i in range(N):
        cm[true_idx[i], pred_idx[i]] += 1

    return {
        "loss":               tot_loss / len(testloader),
        "accuracy":           float((pred_idx == true_idx).mean()),
        "f1_macro":           float(f1_score(true_idx, pred_idx, average="macro",    zero_division=0)),
        "f1_weighted":        float(f1_score(true_idx, pred_idx, average="weighted", zero_division=0)),
        "precision_macro":    float(precision_score(true_idx, pred_idx, average="macro",    zero_division=0)),
        "recall_macro":       float(recall_score(true_idx, pred_idx, average="macro",    zero_division=0)),
        "specificity_macro":  float(np.mean(pc_spec)),
        "auc_macro":          float(np.mean(valid_auc)) if valid_auc else 0.0,
        "superclass_names":   SC_NAMES,
        "per_class_precision": [float(v) for v in np.asarray(precision_score(labels, preds, average=None, zero_division=0))],
        "per_class_recall":    [float(v) for v in np.asarray(recall_score(labels, preds, average=None, zero_division=0))],
        "per_class_f1":        [float(v) for v in np.asarray(f1_score(labels, preds, average=None, zero_division=0))],
        "per_class_specificity": pc_spec,
        "per_class_auc":         pc_auc,
        "per_class_support":     [int(v) for v in labels.sum(0)],
        "per_class_cm":          pc_cm,
        "optimal_thresholds":    [0.5] * NUM_CLASSES,
        "confusion_matrix":      cm.tolist(),
        "num_samples":           N,
        "num_classes":           NUM_CLASSES,
    }


def _empty():
    z = [0.0] * NUM_CLASSES
    return {
        "loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0,
        "precision_macro": 0.0, "recall_macro": 0.0, "specificity_macro": 0.0,
        "auc_macro": 0.0, "superclass_names": SC_NAMES,
        "per_class_precision": z, "per_class_recall": z, "per_class_f1": z,
        "per_class_specificity": z, "per_class_auc": z,
        "per_class_support": [0] * NUM_CLASSES,
        "per_class_cm": [{"TP": 0, "FP": 0, "FN": 0, "TN": 0}] * NUM_CLASSES,
        "optimal_thresholds": [0.5] * NUM_CLASSES,
        "confusion_matrix": [[0] * NUM_CLASSES] * NUM_CLASSES,
        "num_samples": 0, "num_classes": NUM_CLASSES,
    }
