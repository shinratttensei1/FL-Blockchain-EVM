"""FL-Blockchain-EVM: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from typing import List, Tuple, Dict
from flwr.common import Metrics

from fl_blockchain_evm.task import Net, load_data
from fl_blockchain_evm.task import test as test_fn
from fl_blockchain_evm.task import train as train_fn

# Flower ClientApp
app = ClientApp()

RED = '\033[91m'
RESET = '\033[0m'


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data with emergency detection.

    Simulates medical IoT edge device that scans local data for critical pathologies.
    In this FEMNIST simulation:
    - Classes 10-61 (letters A-Z, a-z) = "Pathologies" (simulated X-ray anomalies)
    - Classes 0-9 (digits) = "Normal" (healthy scans)

    If pathologies detected, triggers is_emergency flag for priority aggregation.
    """

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    all_labels = []
    for batch in trainloader:
        labels = batch["character"]
        all_labels.extend(labels.tolist())

    pathology_labels = [label for label in all_labels if label >= 50]
    is_emergency = len(pathology_labels) > 0

    pathology_classes = sorted(list(set(pathology_labels)))

    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "client_id": context.node_config["partition-id"],
        "num-examples": len(trainloader.dataset),
        "is_emergency": int(is_emergency),
        "pathology_count": len(pathology_labels),
        "normal_count": len(all_labels) - len(pathology_labels),
    }

    if is_emergency:
        print(
            f"{RED}CLIENT {partition_id}: Detected {len(pathology_labels)} pathologies, sending is_emergency={int(is_emergency)}{RESET}")

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
        "client_id": int(partition_id)
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
