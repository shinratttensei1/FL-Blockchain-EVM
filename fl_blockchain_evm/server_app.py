import torch
import numpy as np
import json
import os
from typing import List, Dict
from datetime import datetime
import matplotlib.pyplot as plt
from fl_blockchain_evm.priority_strategy import MedicalFedAvg
from fl_blockchain_evm.task import Net, test as test_fn, load_data, NUM_CLASSES, SC_NAMES
from fl_blockchain_evm.blockchain import EVMBlockchain as FLBlockchain
from fl_blockchain_evm import live_state
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
import seaborn as sns
import matplotlib
matplotlib.use('Agg')

G, Y, C, R = '\033[92m', '\033[93m', '\033[96m', '\033[0m'
os.makedirs("outputs", exist_ok=True)

SC_LABELS = ['NORM', 'MI (Infarction)', 'STTC (ST/T)',
             'CD (Conduction)', 'HYP (Hypertrophy)']

_blockchain = FLBlockchain()

_round_state: Dict = {
    "train_results":  [],
    "current_round":  0,
    "loss_mean":      0.0,
    "loss_std":       0.0,
    "threshold":      0.0,
}


def _dev():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _plot_cm(cm, rnd, acc, f1):
    if isinstance(cm, list):
        cm = np.array(cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=SC_LABELS, yticklabels=SC_LABELS)
    plt.title(f'Confusion Matrix — Round {rnd}\nAcc: {acc:.2%} | F1: {f1:.4f}')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(f"outputs/cm_round_{rnd}.png", dpi=150, bbox_inches='tight')
    plt.close()


def global_evaluate(server_round, arrays, config=None):
    live_state.evaluating(server_round)

    model = Net()
    dev = _dev()
    model.load_state_dict(arrays.to_torch_state_dict())
    model.to(dev)

    _, testloader = load_data(0, 10, beta=0)
    m = test_fn(model, testloader, dev)

    print(f"\n{Y}{'═'*60}{R}")
    print(f"{Y}  [ROUND {server_round}] GLOBAL — 5 Superclasses{R}")
    print(f"{Y}{'═'*60}{R}")
    for k in ["loss", "accuracy", "f1_macro", "f1_weighted",
              "precision_macro", "recall_macro", "specificity_macro", "auc_macro"]:
        print(f"   {k:20s}: {m[k]:.4f}")
    print(f"   {'── Per-Superclass ──':20s}")
    for i, sc in enumerate(SC_NAMES):
        print(f"     {sc:5s}  P={m['per_class_precision'][i]:.3f}  "
              f"R={m['per_class_recall'][i]:.3f}  "
              f"F1={m['per_class_f1'][i]:.3f}  "
              f"AUC={m['per_class_auc'][i]:.3f}  "
              f"(n={m['per_class_support'][i]})")

    _plot_cm(m["confusion_matrix"], server_round, m["accuracy"], m["f1_macro"])

    num_clients = len(_round_state["train_results"])
    _blockchain.add_global_model_block(
        fl_round=server_round,
        model_state_dict=arrays.to_torch_state_dict(),
        accuracy=m["accuracy"],
        f1_macro=m["f1_macro"],
        auc_macro=m["auc_macro"],
        loss=m["loss"],
        num_clients=num_clients,
    )

    _round_state["train_results"] = []

    chain_length = _blockchain.get_chain_length()
    chain_valid = _blockchain.verify_chain()

    summary = _blockchain.get_round_summary(server_round)
    print(f"\n{C}  [BLOCKCHAIN] Round {server_round} complete — "
          f"3 blocks written (LOCAL + VOTE + GLOBAL) | "
          f"Chain length: {summary['total_blocks']}{R}")

    # ── Live state update ──
    live_state.round_complete(server_round, m, chain_length, chain_valid)

    log = {k: m[k] for k in [
        "loss", "accuracy", "f1_macro", "f1_weighted",
        "precision_macro", "recall_macro", "specificity_macro", "auc_macro",
        "per_class_f1", "per_class_precision", "per_class_recall",
        "per_class_auc", "per_class_support", "confusion_matrix",
        "num_samples", "num_classes",
    ]}
    log.update({
        "round":               server_round,
        "type":                "global",
        "timestamp":           datetime.now().isoformat(),
        "superclass_names":    SC_NAMES,
        "optimal_thresholds":  m.get("optimal_thresholds", [0.5] * 5),
        "blockchain_blocks":   summary["total_blocks"],
    })
    # Include IPFS CIDs if available
    round_cids = _blockchain.get_round_cids(server_round)
    if round_cids:
        log["ipfs_cids"] = round_cids
        live_state.ipfs_pinned(server_round, round_cids)
    with open("outputs/results.json", "a") as f:
        json.dump(log, f)
        f.write("\n")

    return {
        "loss":      m["loss"],
        "accuracy":  m["accuracy"],
        "f1_macro":  m["f1_macro"],
        "auc_macro": m["auc_macro"],
    }


_rnd = {"train": 0, "eval": 0}


def _print_table(header, rows, cols):
    widths = [max(len(str(r[i])) for r in [header] + rows) + 2
              for i in range(len(cols))]
    sep = "+" + "+".join("-" * w for w in widths) + "+"

    def _row(r):
        return "|" + "|".join(str(r[i]).center(w)
                              for i, w in enumerate(widths)) + "|"
    print(sep)
    print(_row(header))
    print(sep)
    for r in rows:
        print(_row(r))
    print(sep)


def train_metrics_aggregation(metrics_list, weighting_key):
    _rnd["train"] += 1
    rnd = _rnd["train"]
    _round_state["current_round"] = rnd

    live_state.round_started(rnd)

    data = []
    for m in sorted(metrics_list,
                    key=lambda x: int(x["metrics"].get("client_id", 0))):
        met = m["metrics"]
        data.append({
            "client_id":      int(met.get("client_id", 0)),
            "train_loss":     float(met.get("train_loss", 0)),
            "num_examples":   int(met.get("num-examples", 0)),
            "training_time":  float(met.get("training_time_seconds", 0)),
            "active_classes": int(met.get("active_classes", 0)),
        })

    _round_state["train_results"] = data

    print(f"\n{C}  ROUND {rnd} TRAINING: {len(data)} devices{R}")
    _print_table(
        ["Device", "Loss", "Samples", "Time(s)", "Cls"],
        [[d["client_id"], f"{d['train_loss']:.4f}", d["num_examples"],
          f"{d['training_time']:.1f}", d["active_classes"]] for d in data],
        ["Device", "Loss", "Samples", "Time(s)", "Cls"],
    )

    losses = [d["train_loss"] for d in data]
    loss_mean = float(np.mean(losses))
    loss_std = float(np.std(losses)) if len(losses) > 1 else 0.0
    threshold = loss_mean + loss_std

    _round_state["loss_mean"] = loss_mean
    _round_state["loss_std"] = loss_std
    _round_state["threshold"] = threshold

    print(f"  Loss  mean={loss_mean:.4f}  std={loss_std:.4f}  "
          f"threshold={threshold:.4f}")

    print(f"\n{C}  [BLOCKCHAIN] Firing LOCAL + VOTE blocks for Round {rnd} "
          f"(async)...{R}")

    votes = _blockchain.add_round_summary_block(
        fl_round=rnd,
        clients=data,
        loss_mean=loss_mean,
        loss_std=loss_std,
        threshold=threshold,
    )

    _print_table(
        ["Device", "Loss", "Verdict"],
        [[v["client_id"], f"{v['loss']:.4f}", v["vote"]] for v in votes],
        ["Device", "Loss", "Verdict"],
    )

    # ── Live state update ──
    live_state.clients_trained(data, loss_mean, loss_std, threshold, votes)

    with open("outputs/results.json", "a") as f:
        json.dump({
            "round":      rnd,
            "type":       "device_training",
            "timestamp":  datetime.now().isoformat(),
            "loss_mean":  loss_mean,
            "loss_std":   loss_std,
            "threshold":  threshold,
            "devices":    data,
        }, f)
        f.write("\n")

    return MetricRecord({
        "train_loss_avg": float(np.mean(losses)),
        "num_devices":    float(len(data)),
    })


def weighted_average(metrics_list, weighting_key):
    _rnd["eval"] += 1
    rnd = _rnd["eval"]
    total = sum(int(m["metrics"]["num-examples"]) for m in metrics_list)
    if total == 0:
        return MetricRecord({"eval_acc": 0.0, "eval_f1": 0.0})

    def wavg(k):
        return sum(
            float(m["metrics"][k]) * int(m["metrics"]["num-examples"])
            for m in metrics_list
        ) / total

    data = []
    for m in sorted(metrics_list,
                    key=lambda x: int(x["metrics"]["client_id"])):
        met = m["metrics"]
        data.append({
            k: float(met[k]) if k != "client_id" else int(met[k])
            for k in ["client_id", "eval_loss", "eval_acc",
                      "eval_f1", "eval_auc", "num-examples"]
        })

    print(f"\n{G}  ROUND {rnd} EVALUATION: {len(data)} devices{R}")
    _print_table(
        ["Device", "Loss", "Acc", "F1", "AUC", "N"],
        [[d["client_id"], f"{d['eval_loss']:.4f}", f"{d['eval_acc']:.4f}",
          f"{d['eval_f1']:.4f}", f"{d['eval_auc']:.4f}", int(d["num-examples"])]
         for d in data],
        ["Device", "Loss", "Acc", "F1", "AUC", "N"],
    )

    with open("outputs/results.json", "a") as f:
        json.dump([{
            "type":      "client_eval",
            "round":     rnd,
            "timestamp": datetime.now().isoformat(),
            **d,
        } for d in data], f)
        f.write("\n")

    return MetricRecord({
        k: wavg(k) for k in [
            "eval_acc", "eval_f1", "eval_f1_weighted",
            "eval_precision", "eval_recall",
            "eval_specificity", "eval_auc",
        ]
    })


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context):
    lr = context.run_config["lr"]
    num_rounds = context.run_config["num-server-rounds"]
    frac = context.run_config["fraction-train"]

    if os.path.exists("outputs/results.json"):
        os.remove("outputs/results.json")

    # ── Init live state ──
    live_state.init(num_rounds, _blockchain.contract_address)

    model = Net()
    strategy = MedicalFedAvg(
        fraction_train=frac,
        train_metrics_aggr_fn=train_metrics_aggregation,
        evaluate_metrics_aggr_fn=weighted_average,
    )

    ipfs_status = "enabled" if _blockchain.ipfs_enabled else "disabled"
    print(f"\n{C}{'═'*60}")
    print(f"  5 Superclasses: {', '.join(SC_NAMES)}")
    print(f"  Rounds: {num_rounds} | LR: {lr} | Device: {_dev()}")
    print(f"  Blockchain: 3 tx per round (LOCAL + VOTE + GLOBAL)")
    print(f"  IPFS:       {ipfs_status}")
    print(f"  Dashboard:  open dashboard.html in your browser")
    print(f"{'═'*60}{R}\n")

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(model.state_dict()),
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    live_state.done(_blockchain.get_chain_length())
    _blockchain.print_chain_summary()

    torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
    print(f"\n{G}   Done. Model  -> final_model.pt")
    print(f"    Metrics -> outputs/results.json{R}")
