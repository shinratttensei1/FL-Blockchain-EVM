"""FL-Blockchain-EVM: A Flower / PyTorch app."""

from datetime import datetime
import json
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from fl_blockchain_evm.task import Net, test as test_fn, load_data as load_server_data
from typing import List

from fl_blockchain_evm.task import Net


def global_evaluate(server_round: int, arrays: ArrayRecord, config: ConfigRecord = None):
    model = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(arrays.to_torch_state_dict())
    model.to(device)

    _, server_testloader = load_server_data(partition_id=0, num_partitions=1)
    loss, accuracy = test_fn(model, server_testloader, device)

    return {"loss": float(loss), "accuracy": float(accuracy)}


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy with new aggregation functions
    strategy = FedAvg(
        fraction_train=fraction_train,
        train_metrics_aggr_fn=weighted_loss_average,
        evaluate_metrics_aggr_fn=weighted_average,
    )

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    print("\n--- Final Results ---")
    if result.evaluate_metrics_clientapp:
        # Get the latest round's metrics
        last_round = max(result.evaluate_metrics_clientapp.keys())
        final_metrics = result.evaluate_metrics_clientapp[last_round]

        print(
            f"Final Aggregated Accuracy (Round {last_round}): {final_metrics['accuracy']}")
    else:
        print("No distributed evaluation metrics found.")

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")


def weighted_average(metrics: List[RecordDict], weighting_key: str) -> MetricRecord:
    round_results = []
    total_examples = sum(int(m["metrics"]["num-examples"]) for m in metrics)

    for m in metrics:
        cid = m["metrics"].get("client_id", "Unknown")
        acc = m["metrics"]["eval_acc"]

        round_results.append({
            "client_id": cid,
            "accuracy": float(acc),
            "samples": int(m["metrics"]["num-examples"]),
            "timestamp": datetime.now().isoformat()
        })
        print(f"Client {cid} -> Accuracy: {acc:.4f}")

    with open("results.json", "a") as f:
        json.dump(round_results, f)
        f.write("\n")

    weighted_acc = sum(float(m["accuracy"]) * m["samples"]
                       for m in round_results) / total_examples
    return MetricRecord({"accuracy": weighted_acc})


def weighted_loss_average(metrics: List[RecordDict], weighting_key: str) -> MetricRecord:
    total_examples = sum(int(m["metrics"]["num-examples"]) for m in metrics)
    weighted_loss = sum(
        float(m["metrics"]["train_loss"]) * int(m["metrics"]["num-examples"])
        for m in metrics
    ) / total_examples
    return MetricRecord({"train_loss": weighted_loss})
