"""FL-Blockchain-EVM: Client App — IoT Medical Edge Device.

Each client simulates an IoT medical edge device that:
  1. Loads its local partition of PTB-XL ECG data
  2. Applies ROS+RUS balancing locally (Jimenez et al., 2024)
  3. Trains the CRNN model for `local-epochs`
  4. Reports per-device training & evaluation metrics
"""

import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from fl_blockchain_evm.task import (
    Net, load_data, train as train_fn, test as test_fn,
    ALL_SCP_CODES, NUM_CLASSES, SC_NAMES, apply_ros_rus_balancing
)


# Unified device selection for MacBook M4 and others
def _get_device() -> torch.device:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Local training on an IoT edge device with ROS+RUS balancing."""
    model = Net(num_classes=NUM_CLASSES)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = _get_device()
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Let task.py handle all balancing and normalization
    trainloader, _ = load_data(partition_id, num_partitions, beta=0.8)

    # Train directly
    train_metrics = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Collect class distribution for reporting
    y_train = trainloader.dataset.tensors[1]
    class_counts = y_train.sum(dim=0).cpu().numpy().astype(int)
    active_codes = int((class_counts > 0).sum())
    num_examples = int(y_train.shape[0])

    # Package response with all required metrics
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": float(train_metrics["train_loss"]),
        "train_loss_first_epoch": float(train_metrics.get("train_loss_first_epoch", train_metrics["train_loss"])),
        "train_loss_last_epoch": float(train_metrics.get("train_loss_last_epoch", train_metrics["train_loss"])),
        "client_id": int(partition_id),
        "active_scp_codes": active_codes,
        "num-examples": num_examples,
        "training_time_seconds": float(train_metrics.get("training_time_seconds", 0.0)),
    }

    # Free up MPS memory after training
    if device.type == "mps":
        torch.mps.empty_cache()

    return Message(
        content=RecordDict({
            "arrays": model_record,
            "metrics": MetricRecord(metrics)
        }),
        reply_to=msg
    )


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Local evaluation on the shared test set."""
    model = Net(num_classes=NUM_CLASSES)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = _get_device()
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

    eval_result = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": float(eval_result["loss"]),
        "eval_acc": float(eval_result["accuracy"]),
        "eval_f1": float(eval_result["f1_macro"]),
        "eval_f1_weighted": float(eval_result["f1_weighted"]),
        "eval_precision": float(eval_result["precision_macro"]),
        "eval_recall": float(eval_result["recall_macro"]),
        "eval_specificity": float(eval_result["specificity_macro"]),
        "eval_auc": float(eval_result["auc_macro"]),
        "num-examples": int(eval_result["num_samples"]),
        "client_id": int(partition_id),
    }

    model_record = ArrayRecord(model.state_dict())

    # Free up MPS memory after evaluation
    if device.type == "mps":
        torch.mps.empty_cache()

    return Message(
        content=RecordDict({
            "arrays": model_record,
            "metrics": MetricRecord(metrics)
        }),
        reply_to=msg
    )
