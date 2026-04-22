import time
import datetime

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fl_blockchain_evm.task import Net, load_data, train as train_fn, test as test_fn
from fl_blockchain_evm.utils import get_device

try:
    import psutil as _psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


def _ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get_device_stats() -> dict:
    """Collect lightweight system metrics for the current device.

    Returns cpu_percent, ram_used_mb, cpu_temp_c (-1.0 if unavailable).
    Safe on RP4 (reads /sys/class/thermal/thermal_zone0/temp) and on any
    Linux/macOS host where psutil is installed.
    """
    if not _HAS_PSUTIL:
        return {"cpu_percent": -1.0, "ram_used_mb": -1.0, "cpu_temp_c": -1.0}
    cpu_pct = float(_psutil.cpu_percent(interval=0.2))
    ram_mb  = float(_psutil.virtual_memory().used / 1024 ** 2)
    temp    = -1.0
    # RP4 Linux thermal zone (also works on most ARM SBCs)
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as _f:
            temp = float(_f.read().strip()) / 1000.0
    except Exception:
        # Fallback: psutil sensors (Linux/Windows)
        try:
            sensors = _psutil.sensors_temperatures()
            for _key in ("cpu_thermal", "coretemp", "k10temp"):
                if _key in sensors and sensors[_key]:
                    temp = float(sensors[_key][0].current)
                    break
        except Exception:
            pass
    return {
        "cpu_percent": round(cpu_pct, 1),
        "ram_used_mb": round(ram_mb, 1),
        "cpu_temp_c":  round(temp, 1),
    }


def _log(tag: str, msg: str):
    print(f"[{_ts()}] [{tag}] {msg}", flush=True)


app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    pid       = context.node_config["partition-id"]
    n_parts   = context.node_config["num-partitions"]
    lr        = float(msg.content["config"].get("lr", 0.002))
    epochs     = int(context.run_config.get("local-epochs", 1))
    batch_size = int(context.run_config.get("batch-size", 256))
    device    = get_device()

    _log(f"CLIENT-{pid}", "══════════════════════════════════════════════")
    _log(f"CLIENT-{pid}", f"TRAIN  partition={pid}/{n_parts}  "
                          f"epochs={epochs}  lr={lr}  batch={batch_size}  device={device}")

    # ── Load model ────────────────────────────────────────────
    model = Net()
    state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(state_dict)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    _log(f"CLIENT-{pid}", f"Model loaded: {total_params:,} parameters")

    # ── Load data ─────────────────────────────────────────────
    t_data = time.time()
    _log(f"CLIENT-{pid}", f"Loading data partition {pid} (num_partitions={n_parts})...")
    trainloader, _ = load_data(pid, n_parts, beta=1.0, batch_size=batch_size)
    n_batches   = len(trainloader)
    n_samples   = len(trainloader.dataset)
    _log(f"CLIENT-{pid}", f"Data loaded: {n_samples} samples, "
                          f"{n_batches} batches  [{time.time()-t_data:.1f}s]")

    # ── Train ──────────────────────────────────────────────────
    _log(f"CLIENT-{pid}", f"Starting local training ({epochs} epochs)...")
    t_train = time.time()
    m = train_fn(model, trainloader, epochs=epochs, lr=lr, device=device)
    train_elapsed = time.time() - t_train

    # Class activity summary
    y       = trainloader.dataset.tensors[1]
    counts  = y.sum(0).cpu().numpy().astype(int)
    active  = int((counts > 0).sum())

    hw = _get_device_stats()

    _log(f"CLIENT-{pid}", f"Training complete: "
                          f"loss={m['train_loss']:.5f}  "
                          f"first_epoch={m['train_loss_first_epoch']:.5f}  "
                          f"last_epoch={m['train_loss_last_epoch']:.5f}  "
                          f"improvement={m['train_loss_first_epoch']-m['train_loss_last_epoch']:.5f}  "
                          f"time={train_elapsed:.1f}s  "
                          f"active_classes={active}/12  "
                          f"cpu={hw['cpu_percent']}%  "
                          f"ram={hw['ram_used_mb']:.0f}MB  "
                          f"temp={hw['cpu_temp_c']}°C")
    _log(f"CLIENT-{pid}", f"Class counts: {counts.tolist()}")
    _log(f"CLIENT-{pid}", "══════════════════════════════════════════════")

    if device.type == "mps":
        torch.mps.empty_cache()

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord({
                "train_loss":              float(m["train_loss"]),
                "train_loss_first_epoch":  float(m["train_loss_first_epoch"]),
                "train_loss_last_epoch":   float(m["train_loss_last_epoch"]),
                "client_id":               int(pid),
                "active_classes":          int(active),
                "num-examples":            int(y.shape[0]),
                "training_time_seconds":   float(m["training_time_seconds"]),
                "cpu_percent":             hw["cpu_percent"],
                "ram_used_mb":             hw["ram_used_mb"],
                "cpu_temp_c":              hw["cpu_temp_c"],
            }),
        }),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context):
    pid     = context.node_config["partition-id"]
    n_parts = context.node_config["num-partitions"]
    device  = get_device()

    _log(f"CLIENT-{pid}", "══════════════════════════════════════════════")
    _log(f"CLIENT-{pid}", f"EVALUATE  partition={pid}/{n_parts}  device={device}")

    # ── Load model ─────────────────────────────────────────────
    model = Net()
    state_dict = msg.content["arrays"].to_torch_state_dict()
    model.load_state_dict(state_dict)
    model.to(device)

    # ── Load test data ─────────────────────────────────────────
    t_data = time.time()
    batch_size = int(context.run_config.get("batch-size", 256))
    _, valloader = load_data(pid, n_parts, beta=0, batch_size=batch_size)
    n_samples = len(valloader.dataset)
    _log(f"CLIENT-{pid}", f"Test data loaded: {n_samples} samples  [{time.time()-t_data:.1f}s]")

    # ── Evaluate ───────────────────────────────────────────────
    t_eval = time.time()
    r = test_fn(model, valloader, device)
    eval_elapsed = time.time() - t_eval

    hw = _get_device_stats()

    _log(f"CLIENT-{pid}", f"Evaluation complete: "
                          f"loss={r['loss']:.5f}  "
                          f"acc={r['accuracy']:.4f}  "
                          f"f1={r['f1_macro']:.4f}  "
                          f"auc={r['auc_macro']:.4f}  "
                          f"time={eval_elapsed:.1f}s  "
                          f"cpu={hw['cpu_percent']}%  "
                          f"ram={hw['ram_used_mb']:.0f}MB  "
                          f"temp={hw['cpu_temp_c']}°C")
    _log(f"CLIENT-{pid}", "══════════════════════════════════════════════")

    if device.type == "mps":
        torch.mps.empty_cache()

    return Message(
        content=RecordDict({
            "arrays": ArrayRecord(model.state_dict()),
            "metrics": MetricRecord({
                "eval_loss":             float(r["loss"]),
                "eval_acc":              float(r["accuracy"]),
                "eval_f1":               float(r["f1_macro"]),
                "eval_f1_weighted":      float(r["f1_weighted"]),
                "eval_precision":        float(r["precision_macro"]),
                "eval_recall":           float(r["recall_macro"]),
                "eval_specificity":      float(r["specificity_macro"]),
                "eval_auc":              float(r["auc_macro"]),
                "num-examples":          int(r["num_samples"]),
                "client_id":             int(pid),
                "eval_time_seconds":     float(eval_elapsed),
                "cpu_percent":           hw["cpu_percent"],
                "ram_used_mb":           hw["ram_used_mb"],
                "cpu_temp_c":            hw["cpu_temp_c"],
            }),
        }),
        reply_to=msg,
    )
