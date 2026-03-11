import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from fl_blockchain_evm.task import Net, load_data, train as train_fn, test as test_fn
from fl_blockchain_evm.utils import get_device


app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()

    pid = context.node_config["partition-id"]
    trainloader, _ = load_data(
        pid, context.node_config["num-partitions"], beta=1.0)

    m = train_fn(model, trainloader, epochs=context.run_config["local-epochs"],
                 lr=msg.content["config"]["lr"], device=device)

    y = trainloader.dataset.tensors[1]
    counts = y.sum(0).cpu().numpy().astype(int)

    if device.type == "mps":
        torch.mps.empty_cache()
    return Message(content=RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({
            "train_loss": float(m["train_loss"]),
            "train_loss_first_epoch": float(m["train_loss_first_epoch"]),
            "train_loss_last_epoch": float(m["train_loss_last_epoch"]),
            "client_id": int(pid),
            "active_classes": int((counts > 0).sum()),
            "num-examples": int(y.shape[0]),
            "training_time_seconds": float(m["training_time_seconds"]),
        }),
    }), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = get_device()

    pid = context.node_config["partition-id"]
    _, valloader = load_data(
        pid, context.node_config["num-partitions"], beta=0)
    r = test_fn(model, valloader, device)

    if device.type == "mps":
        torch.mps.empty_cache()
    return Message(content=RecordDict({
        "arrays": ArrayRecord(model.state_dict()),
        "metrics": MetricRecord({
            "eval_loss": float(r["loss"]),
            "eval_acc": float(r["accuracy"]),
            "eval_f1": float(r["f1_macro"]),
            "eval_f1_weighted": float(r["f1_weighted"]),
            "eval_precision": float(r["precision_macro"]),
            "eval_recall": float(r["recall_macro"]),
            "eval_specificity": float(r["specificity_macro"]),
            "eval_auc": float(r["auc_macro"]),
            "num-examples": int(r["num_samples"]),
            "client_id": int(pid),
        }),
    }), reply_to=msg)
