"""FL-Blockchain-EVM: Server App — Federated Aggregation Coordinator.

Orchestrates 10 simulated IoT medical edge devices:
  - Global evaluation with comprehensive metrics after each round
  - Per-round JSON logging (results.json) for all metrics
  - Confusion matrix heatmaps (5×5 superclass) saved as PNG
  - Per-class bar charts for the top SCP codes
  - Final model checkpoint
"""

from fl_blockchain_evm.priority_strategy import MedicalFedAvg
from fl_blockchain_evm.task import (
    Net, test as test_fn, load_data,
    ALL_SCP_CODES, NUM_CLASSES, SC_NAMES, SC_GROUPS,
    SUPERCLASS_MAP, N_DIAG, N_FORM, N_RHYTHM,
)
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg
from flwr.serverapp import Grid, ServerApp
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import torch
from datetime import datetime
from typing import List, Tuple, Union, Optional, Dict

import matplotlib
matplotlib.use('Agg')

RED, GREEN, YELLOW, CYAN, RESET = (
    '\033[91m', '\033[92m', '\033[93m', '\033[96m', '\033[0m')

os.makedirs("outputs", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

SC_LABELS = [
    'NORM (Normal)', 'MI (Infarction)', 'STTC (ST/T Change)',
    'CD (Conduction Dist.)', 'HYP (Hypertrophy)',
]


# ──────────────────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────────────────


def plot_confusion_matrix(cm, round_num, accuracy, f1):
    """5×5 superclass confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=SC_LABELS, yticklabels=SC_LABELS)
    plt.title(
        f'Superclass Confusion Matrix — Round {round_num}\n'
        f'Accuracy: {accuracy:.2%} | F1-macro (71 codes): {f1:.4f}'
    )
    plt.ylabel('True Superclass')
    plt.xlabel('Predicted Superclass')
    fname = f"outputs/confusion_matrix_round_{round_num}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"{GREEN}   ✔ CM heatmap saved: {fname}{RESET}")


def plot_superclass_bar(metrics, round_num):
    """Bar chart of superclass-level P / R / F1."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(5)
    w = 0.2
    ax.bar(x - 1.5*w, metrics["superclass_precision"],
           w, label='Precision', color='#3498db')
    ax.bar(x - 0.5*w, metrics["superclass_recall"],
           w, label='Recall (Sens.)', color='#2ecc71')
    ax.bar(x + 0.5*w, metrics["superclass_f1"],
           w, label='F1-Score', color='#e67e22')
    ax.bar(x + 1.5*w, metrics["superclass_auc"],
           w, label='AUC-ROC', color='#9b59b6')
    ax.set_xticks(x)
    ax.set_xticklabels(SC_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(f'Superclass Metrics — Round {round_num}')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fname = f"outputs/superclass_round_{round_num}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"{GREEN}   ✔ Superclass chart saved: {fname}{RESET}")


def plot_top_codes_bar(metrics, round_num, top_k=20):
    """Bar chart of F1 for the top-K SCP codes by support."""
    supports = np.array(metrics["per_class_support"])
    f1s = np.array(metrics["per_class_f1"])

    # Pick top-K by support (so we show the most data-rich codes)
    top_idx = np.argsort(-supports)[:top_k]
    codes = [ALL_SCP_CODES[i] for i in top_idx]
    vals = [f1s[i] for i in top_idx]
    sups = [supports[i] for i in top_idx]

    # Color by category
    cat_colors = {'NORM': '#3498db', 'MI': '#e74c3c', 'STTC': '#2ecc71',
                  'CD': '#f39c12', 'HYP': '#9b59b6',
                  'FORM': '#1abc9c', 'RHYTHM': '#e67e22'}
    colors = [cat_colors.get(SUPERCLASS_MAP.get(c, ''), '#95a5a6')
              for c in codes]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(range(len(codes)), vals, color=colors)
    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels([f"{c}\n(n={s})" for c, s in zip(codes, sups)],
                       rotation=60, ha='right', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('F1-Score')
    ax.set_title(f'Top-{top_k} SCP Codes by F1 — Round {round_num}',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Legend for categories
    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=v, label=k) for k, v in cat_colors.items()]
    ax.legend(handles=legend_elems, loc='upper right', fontsize=8)

    fname = f"outputs/top_codes_round_{round_num}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"{GREEN}   ✔ Top-{top_k} codes chart saved: {fname}{RESET}")


def plot_device_dashboard(device_data: List[Dict], round_num: int,
                          phase: str = "train"):
    """Per-device dashboard proving 10 IoT edge devices participated.

    Generates a multi-panel figure showing each device's contribution:
      - Training loss per device  OR  Eval F1 per device
      - Samples processed per device
      - Training time per device
    """
    if not device_data:
        return

    n = len(device_data)
    ids = [d.get("client_id", i) for i, d in enumerate(device_data)]
    device_labels = [f"Device {i}" for i in ids]
    colors = plt.cm.tab10(np.linspace(0, 1, max(n, 1)))

    if phase == "train":
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"IoT Edge Device Training Dashboard — Round {round_num}\n"
            f"{n} Devices Participated in Federated Learning",
            fontsize=14, fontweight='bold', y=1.02,
        )

        # Panel 1: Training loss
        losses = [d.get("train_loss", 0) for d in device_data]
        axes[0].bar(range(n), losses, color=colors)
        axes[0].set_xticks(range(n))
        axes[0].set_xticklabels(
            device_labels, rotation=45, ha='right', fontsize=8)
        axes[0].set_ylabel('Training Loss')
        axes[0].set_title('Loss per Device')
        axes[0].grid(axis='y', alpha=0.3)
        for j, v in enumerate(losses):
            axes[0].text(j, v + 0.001, f'{v:.4f}',
                         ha='center', va='bottom', fontsize=7)

        # Panel 2: Samples processed
        samples = [d.get("num_examples", 0) for d in device_data]
        axes[1].bar(range(n), samples, color=colors)
        axes[1].set_xticks(range(n))
        axes[1].set_xticklabels(
            device_labels, rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('Training Samples')
        axes[1].set_title('Local Dataset Size per Device')
        axes[1].grid(axis='y', alpha=0.3)
        for j, v in enumerate(samples):
            axes[1].text(j, v + 10, str(v), ha='center',
                         va='bottom', fontsize=7)

        # Panel 3: Training time
        times = [d.get("training_time_seconds", 0) for d in device_data]
        axes[2].bar(range(n), times, color=colors)
        axes[2].set_xticks(range(n))
        axes[2].set_xticklabels(
            device_labels, rotation=45, ha='right', fontsize=8)
        axes[2].set_ylabel('Time (seconds)')
        axes[2].set_title('Training Time per Device')
        axes[2].grid(axis='y', alpha=0.3)
        for j, v in enumerate(times):
            axes[2].text(j, v + 0.1, f'{v:.1f}s',
                         ha='center', va='bottom', fontsize=7)

    else:  # eval
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(
            f"IoT Edge Device Evaluation Dashboard — Round {round_num}\n"
            f"{n} Devices Evaluated Independently",
            fontsize=14, fontweight='bold', y=1.02,
        )

        f1s = [d.get("eval_f1", 0) for d in device_data]
        axes[0].bar(range(n), f1s, color=colors)
        axes[0].set_xticks(range(n))
        axes[0].set_xticklabels(
            device_labels, rotation=45, ha='right', fontsize=8)
        axes[0].set_ylabel('F1-macro')
        axes[0].set_title('F1 Score per Device')
        axes[0].set_ylim(0, 1.05)
        axes[0].grid(axis='y', alpha=0.3)
        for j, v in enumerate(f1s):
            axes[0].text(j, v + 0.01, f'{v:.3f}',
                         ha='center', va='bottom', fontsize=7)

        accs = [d.get("eval_acc", 0) for d in device_data]
        axes[1].bar(range(n), accs, color=colors)
        axes[1].set_xticks(range(n))
        axes[1].set_xticklabels(
            device_labels, rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy per Device')
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(axis='y', alpha=0.3)
        for j, v in enumerate(accs):
            axes[1].text(j, v + 0.01, f'{v:.3f}',
                         ha='center', va='bottom', fontsize=7)

        aucs = [d.get("eval_auc", 0) for d in device_data]
        axes[2].bar(range(n), aucs, color=colors)
        axes[2].set_xticks(range(n))
        axes[2].set_xticklabels(
            device_labels, rotation=45, ha='right', fontsize=8)
        axes[2].set_ylabel('AUC-ROC')
        axes[2].set_title('AUC per Device')
        axes[2].set_ylim(0, 1.05)
        axes[2].grid(axis='y', alpha=0.3)
        for j, v in enumerate(aucs):
            axes[2].text(j, v + 0.01, f'{v:.3f}',
                         ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    fname = f"outputs/device_dashboard_{phase}_round_{round_num}.png"
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"{GREEN}   ✔ Device dashboard saved: {fname}{RESET}")


# ──────────────────────────────────────────────────────────────────────
# Global evaluation callback
# ──────────────────────────────────────────────────────────────────────


def _get_device() -> torch.device:
    """Select best available accelerator: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def global_evaluate(server_round: int, arrays: ArrayRecord,
                    config: ConfigRecord = None):
    """Evaluate the aggregated global model on the shared test set."""
    model = Net(num_classes=NUM_CLASSES)
    device = _get_device()
    model.load_state_dict(arrays.to_torch_state_dict())
    model.to(device)

    # Читаем num_partitions из конфига, если есть, иначе 10
    num_partitions = 10
    if config is not None and hasattr(config, 'to_dict'):
        conf_dict = config.to_dict()
        num_partitions = int(conf_dict.get('num-partitions', 10))
    _, server_testloader = load_data(
        partition_id=0, num_partitions=num_partitions)
    metrics = test_fn(model, server_testloader, device)

    # ── Console output ──
    print(
        f"\n{YELLOW}═══════════════════════════════════════════════════════════════{RESET}")
    print(
        f"{YELLOW}  [ROUND {server_round}] GLOBAL MODEL — 71 SCP Codes{RESET}")
    print(
        f"{YELLOW}═══════════════════════════════════════════════════════════════{RESET}")
    print(f"   Loss       : {metrics['loss']:.4f}")
    print(f"   Accuracy   : {metrics['accuracy']:.4f}")
    print(f"   F1-macro   : {metrics['f1_macro']:.4f}  (over 71 codes)")
    print(f"   F1-weighted: {metrics['f1_weighted']:.4f}")
    print(f"   Precision  : {metrics['precision_macro']:.4f}")
    print(f"   Recall     : {metrics['recall_macro']:.4f}  (Sensitivity)")
    print(f"   Specificity: {metrics['specificity_macro']:.4f}")
    print(f"   AUC-ROC    : {metrics['auc_macro']:.4f}")

    # Superclass breakdown
    print(f"   ── Superclass breakdown ──")
    for i, sc in enumerate(SC_NAMES):
        print(
            f"     {sc:5s}  P={metrics['superclass_precision'][i]:.3f}  "
            f"R={metrics['superclass_recall'][i]:.3f}  "
            f"F1={metrics['superclass_f1'][i]:.3f}  "
            f"AUC={metrics['superclass_auc'][i]:.3f}  "
            f"(n={metrics['superclass_support'][i]})"
        )

    # Top-10 fine-grained codes by F1
    f1_arr = np.array(metrics["per_class_f1"])
    sup_arr = np.array(metrics["per_class_support"])
    # Only show codes with support > 0
    active = np.where(sup_arr > 0)[0]
    if len(active) > 0:
        top10 = active[np.argsort(-f1_arr[active])[:10]]
        print(f"   ── Top-10 SCP codes by F1 ──")
        for idx in top10:
            print(
                f"     {ALL_SCP_CODES[idx]:8s}  F1={f1_arr[idx]:.3f}  "
                f"P={metrics['per_class_precision'][idx]:.3f}  "
                f"R={metrics['per_class_recall'][idx]:.3f}  "
                f"(n={sup_arr[idx]})"
            )

    # ── Save plots ──
    plot_confusion_matrix(
        metrics["confusion_matrix"], server_round,
        metrics["accuracy"], metrics["f1_macro"],
    )
    plot_superclass_bar(metrics, server_round)
    plot_top_codes_bar(metrics, server_round)

    # ── JSON log ──
    log_entry = {
        "round": server_round,
        "type": "global",
        "timestamp": datetime.now().isoformat(),
        "num_classes": metrics["num_classes"],
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "specificity_macro": metrics["specificity_macro"],
        "auc_macro": metrics["auc_macro"],
        # Superclass aggregates
        "superclass_names": metrics["superclass_names"],
        "superclass_f1": metrics["superclass_f1"],
        "superclass_precision": metrics["superclass_precision"],
        "superclass_recall": metrics["superclass_recall"],
        "superclass_auc": metrics["superclass_auc"],
        "superclass_support": metrics["superclass_support"],
        # Per SCP-code (71 entries each)
        "scp_codes": list(ALL_SCP_CODES),
        "per_class_f1": metrics["per_class_f1"],
        "per_class_precision": metrics["per_class_precision"],
        "per_class_recall": metrics["per_class_recall"],
        "per_class_auc": metrics["per_class_auc"],
        "per_class_support": metrics["per_class_support"],
        # Confusion matrix (5×5 superclass)
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "num_samples": metrics["num_samples"],
    }
    with open("outputs/results.json", "a") as f:
        json.dump(log_entry, f)
        f.write("\n")

    return {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "auc_macro": metrics["auc_macro"],
    }


# ──────────────────────────────────────────────────────────────────────
# Client-metric weighted average
# ──────────────────────────────────────────────────────────────────────


# Track the current round for the aggregation callbacks
_current_round = {"train": 0, "eval": 0}


def train_metrics_aggregation(metrics_list: List[RecordDict],
                              weighting_key: str) -> MetricRecord:
    """Aggregate per-client TRAINING metrics and print device table."""
    _current_round["train"] += 1
    rnd = _current_round["train"]

    # ── Console: Per-Device Training Summary ──
    print(
        f"\n{CYAN}┌─────────────────────────────────────────────────────────────────┐{RESET}")
    print(
        f"{CYAN}│  ROUND {rnd} — TRAINING REPORT: "
        f"{len(metrics_list)} IoT Edge Devices"
        f"{' ' * (32 - len(str(rnd)) - len(str(len(metrics_list))))}│{RESET}")
    print(
        f"{CYAN}├────────┬──────────┬──────────┬────────────┬───────┬────────────┤{RESET}")
    print(
        f"{CYAN}│ Device │   Loss   │ Samples  │  Time (s)  │ SCP # │ Loss Δ     │{RESET}")
    print(
        f"{CYAN}├────────┼──────────┼──────────┼────────────┼───────┼────────────┤{RESET}")

    device_data = []
    total_samples = 0
    total_time = 0.0

    for m in sorted(metrics_list,
                    key=lambda x: int(x["metrics"].get("client_id", 0))):
        met = m["metrics"]
        cid = int(met.get("client_id", 0))
        loss = float(met.get("train_loss", 0.0))
        n_ex = int(met.get("num-examples", 0))
        t_sec = float(met.get("training_time_seconds", 0.0))
        active = int(met.get("active_scp_codes", 0))
        loss_first = float(met.get("train_loss_first_epoch", loss))
        loss_last = float(met.get("train_loss_last_epoch", loss))
        delta = loss_last - loss_first
        delta_str = f"{delta:+.4f}"

        total_samples += n_ex
        total_time += t_sec

        print(
            f"{CYAN}│{RESET}   {cid:2d}   "
            f"{CYAN}│{RESET} {loss:.4f}   "
            f"{CYAN}│{RESET}  {n_ex:5d}   "
            f"{CYAN}│{RESET}   {t_sec:6.1f}   "
            f"{CYAN}│{RESET}  {active:2d}  "
            f"{CYAN}│{RESET} {delta_str:>10s} "
            f"{CYAN}│{RESET}")

        device_data.append({
            "client_id": cid, "train_loss": loss,
            "num_examples": n_ex, "training_time_seconds": t_sec,
            "active_scp_codes": active,
        })

    print(
        f"{CYAN}├────────┼──────────┼──────────┼────────────┼───────┼────────────┤{RESET}")
    print(
        f"{CYAN}│{RESET}  ALL   "
        f"{CYAN}│{RESET}          "
        f"{CYAN}│{RESET}  {total_samples:5d}   "
        f"{CYAN}│{RESET}   {total_time:6.1f}   "
        f"{CYAN}│{RESET}       "
        f"{CYAN}│{RESET}            "
        f"{CYAN}│{RESET}")
    print(
        f"{CYAN}└────────┴──────────┴──────────┴────────────┴───────┴────────────┘{RESET}")

    # ── Save per-device training dashboard plot ──
    plot_device_dashboard(device_data, rnd, phase="train")

    # ── JSON log ──
    log_entry = {
        "round": rnd,
        "type": "device_training",
        "timestamp": datetime.now().isoformat(),
        "num_devices": len(device_data),
        "total_samples": total_samples,
        "total_training_time": round(total_time, 2),
        "devices": device_data,
    }
    with open("outputs/results.json", "a") as f:
        json.dump(log_entry, f)
        f.write("\n")

    # Return simple aggregate
    losses = [d["train_loss"] for d in device_data]
    return MetricRecord({
        "train_loss_avg": float(np.mean(losses)),
        "num_devices": float(len(device_data)),
    })


def weighted_average(metrics_list: List[RecordDict],
                     weighting_key: str) -> MetricRecord:
    """Compute weighted average of per-client evaluation metrics."""
    _current_round["eval"] += 1
    rnd = _current_round["eval"]

    total = sum(int(m["metrics"]["num-examples"]) for m in metrics_list)
    if total == 0:
        return MetricRecord({"eval_acc": 0.0, "eval_f1": 0.0})

    def _wavg(key):
        return sum(
            float(m["metrics"][key]) * int(m["metrics"]["num-examples"])
            for m in metrics_list
        ) / total

    agg = MetricRecord({
        "eval_acc": _wavg("eval_acc"),
        "eval_f1": _wavg("eval_f1"),
        "eval_f1_weighted": _wavg("eval_f1_weighted"),
        "eval_precision": _wavg("eval_precision"),
        "eval_recall": _wavg("eval_recall"),
        "eval_specificity": _wavg("eval_specificity"),
        "eval_auc": _wavg("eval_auc"),
        # Add per-class metrics if present
        **{k: _wavg(k) for k in [
            "per_class_precision", "per_class_recall", "per_class_f1", "per_class_specificity", "per_class_auc", "per_class_support"
        ] if k in metrics_list[0]["metrics"]}
    })

    # ── Console: Per-Device Evaluation Summary ──
    print(
        f"\n{GREEN}┌───────────────────────────────────────────────────────────────┐{RESET}")
    print(
        f"{GREEN}│  ROUND {rnd} — EVALUATION: "
        f"{len(metrics_list)} IoT Edge Devices"
        f"{' ' * (34 - len(str(rnd)) - len(str(len(metrics_list))))}│{RESET}")
    print(
        f"{GREEN}├────────┬──────────┬──────────┬──────────┬──────────┬──────────┤{RESET}")
    print(
        f"{GREEN}│ Device │   Loss   │   Acc    │ F1-macro │  AUC-ROC │ Samples  │{RESET}")
    print(
        f"{GREEN}├────────┼──────────┼──────────┼──────────┼──────────┼──────────┤{RESET}")

    device_data = []
    for m in sorted(metrics_list,
                    key=lambda x: int(x["metrics"]["client_id"])):
        met = m["metrics"]
        cid = int(met["client_id"])
        eloss = float(met["eval_loss"])
        eacc = float(met["eval_acc"])
        ef1 = float(met["eval_f1"])
        eauc = float(met["eval_auc"])
        n_ex = int(met["num-examples"])

        print(
            f"{GREEN}│{RESET}   {cid:2d}   "
            f"{GREEN}│{RESET} {eloss:.4f}   "
            f"{GREEN}│{RESET} {eacc:.4f}   "
            f"{GREEN}│{RESET} {ef1:.4f}   "
            f"{GREEN}│{RESET} {eauc:.4f}   "
            f"{GREEN}│{RESET}  {n_ex:5d}   "
            f"{GREEN}│{RESET}")

        device_data.append({
            "client_id": cid, "eval_loss": eloss, "eval_acc": eacc,
            "eval_f1": ef1, "eval_auc": eauc, "num_examples": n_ex,
        })

    print(
        f"{GREEN}└────────┴──────────┴──────────┴──────────┴──────────┴──────────┘{RESET}")

    # ── Save per-device eval dashboard plot ──
    plot_device_dashboard(device_data, rnd, phase="eval")

    # ── Log per-client metrics ──
    client_entries = []
    for m in metrics_list:
        entry = {
            "type": "client_eval",
            "round": rnd,
            "timestamp": datetime.now().isoformat(),
            "client_id": int(m["metrics"]["client_id"]),
            "eval_loss": float(m["metrics"]["eval_loss"]),
            "eval_acc": float(m["metrics"]["eval_acc"]),
            "eval_f1": float(m["metrics"]["eval_f1"]),
            "eval_precision": float(m["metrics"]["eval_precision"]),
            "eval_recall": float(m["metrics"]["eval_recall"]),
            "eval_specificity": float(m["metrics"]["eval_specificity"]),
            "eval_auc": float(m["metrics"]["eval_auc"]),
            "num_examples": int(m["metrics"]["num-examples"]),
        }
        # Per-superclass F1 & AUC (if present)
        for sc in SC_NAMES:
            for prefix in ("eval_f1_", "eval_auc_"):
                key = f"{prefix}{sc}"
                if key in m["metrics"]:
                    entry[key] = float(m["metrics"][key])
        client_entries.append(entry)

    with open("outputs/results.json", "a") as f:
        json.dump(client_entries, f)
        f.write("\n")

    return agg


# ──────────────────────────────────────────────────────────────────────
# Server application
# ──────────────────────────────────────────────────────────────────────


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    num_partitions = context.run_config.get("num-partitions", 10)

    results_path = "outputs/results.json"
    if os.path.exists(results_path):
        os.remove(results_path)

    global_model = Net(num_classes=NUM_CLASSES)
    arrays = ArrayRecord(global_model.state_dict())

    strategy = MedicalFedAvg(
        fraction_train=fraction_train,
        train_metrics_aggr_fn=train_metrics_aggregation,
        evaluate_metrics_aggr_fn=weighted_average,
    )

    print(f"\n{CYAN}{'═'*65}{RESET}")
    print(f"{CYAN}  FL-Blockchain-EVM: Federated ECG Classification{RESET}")
    print(f"{CYAN}  10 IoT Devices  |  PTB-XL  |  ALL 71 SCP Codes{RESET}")
    print(f"{CYAN}  Rounds: {num_rounds}  |  LR: {lr}  |  Frac: {fraction_train}{RESET}")
    dev = _get_device()
    print(f"{CYAN}  Device: {dev} {'(Apple Silicon MPS GPU)' if dev.type == 'mps' else ''}{RESET}")
    print(f"{CYAN}{'═'*65}{RESET}\n")

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    print(f"\n{'═'*65}")
    print(f"{GREEN}  SIMULATION COMPLETE{RESET}")
    print(f"{'═'*65}")
    print(f"{GREEN}  Saving final model to final_model.pt ...{RESET}")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
    print(f"{GREEN}  ✔ Model saved successfully!{RESET}")
    print(f"{GREEN}  ✔ Metrics logged to outputs/results.json{RESET}")
    print(f"{GREEN}  ✔ Plots saved to outputs/ and metrics/{RESET}")
