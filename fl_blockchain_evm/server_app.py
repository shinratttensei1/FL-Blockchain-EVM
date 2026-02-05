"""FL-Blockchain-EVM: A Flower / PyTorch app with Priority-Based Aggregation."""

from datetime import datetime
import json
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp
from fl_blockchain_evm.task import Net, test as test_fn, load_data as load_server_data
from typing import List
from fl_blockchain_evm.priority_strategy import PriorityMedicalFedAvg

# ANSI Color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def global_evaluate(server_round: int, arrays: ArrayRecord, config: ConfigRecord = None):
    model = Net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(arrays.to_torch_state_dict())
    model.to(device)

    _, server_testloader = load_server_data(partition_id=0, num_partitions=1)
    loss, accuracy = test_fn(model, server_testloader, device)

    global_result = {
        "round": server_round,
        "type": "global",
        "accuracy": float(accuracy),
        "loss": float(loss),
        "timestamp": datetime.now().isoformat()
    }

    with open("outputs/results.json", "a") as f:
        json.dump(global_result, f)
        f.write("\n")

    return {"loss": loss, "accuracy": accuracy}


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

    # Initialize Priority-Based Medical FedAvg strategy
    # Emergency clients receive 2x weight (e=2.0) for "Emergency Acuity"
    strategy = PriorityMedicalFedAvg(
        fraction_train=fraction_train,
        evaluate_metrics_aggr_fn=weighted_average,
        emergency_multiplier=2.0,  # Priority weight for emergency updates
        alert_log_path="outputs/emergency_alerts.json",  # Immutable audit trail
    )

    # Set the train_metrics_aggr_fn with access to strategy
    strategy.train_metrics_aggr_fn = lambda metrics, key: weighted_loss_average(
        metrics, key, strategy)

    # Start strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    print("\n" + "="*60)
    print(f"{GREEN}--- Final Results ---{RESET}")
    print("="*60)

    if result.evaluate_metrics_clientapp:
        # Get the latest round's metrics
        last_round = max(result.evaluate_metrics_clientapp.keys())
        final_metrics = result.evaluate_metrics_clientapp[last_round]

        print(
            f"{GREEN}Final Aggregated Accuracy (Round {last_round}): {final_metrics['accuracy']}{RESET}")
    else:
        print("No distributed evaluation metrics found.")

    # Display emergency alert statistics
    print(f"\n{GREEN}--- Emergency Alert Statistics ---{RESET}")
    print(f"{GREEN}Total emergency alerts detected: {strategy.total_emergency_count}{RESET}")
    print(
        f"{GREEN}Rounds with emergencies: {len([c for c in strategy.round_emergency_count.values() if c > 0])}{RESET}")
    if strategy.round_emergency_count:
        for round_num, count in sorted(strategy.round_emergency_count.items()):
            if count > 0:
                print(
                    f"{GREEN}  Round {round_num}: {count} emergency client(s){RESET}")
    print(f"{GREEN}Emergency multiplier used: {strategy.emergency_multiplier}x{RESET}")
    print(f"{GREEN}Alert log saved to: {strategy.alert_log_path}{RESET}")

    print(f"\n{GREEN}Saving final model to disk...{RESET}")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
    print(f"{GREEN}Model saved successfully!{RESET}")


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

    with open("outputs/results.json", "a") as f:
        json.dump(round_results, f)
        f.write("\n")

    weighted_acc = sum(float(m["accuracy"]) * m["samples"]
                       for m in round_results) / total_examples
    return MetricRecord({"accuracy": weighted_acc})


def weighted_loss_average(metrics: List[RecordDict], weighting_key: str, strategy: PriorityMedicalFedAvg = None) -> MetricRecord:
    """Aggregate training metrics with priority weighting for emergencies.

    This function is called by Flower to aggregate training metrics from clients.
    We apply emergency detection and priority weighting here.
    """
    # Extract emergency information and apply priority multiplier
    emergency_multiplier = 2.0 if strategy is None else strategy.emergency_multiplier
    emergency_count = 0
    total_pathologies = 0
    emergency_clients = []
    normal_clients = []

    # Get current round from strategy if available
    current_round = getattr(strategy, '_current_round', 0) if strategy else 0

    weighted_loss_sum = 0.0
    total_effective_weight = 0.0

    for m in metrics:
        client_metrics = m["metrics"]
        num_examples = int(client_metrics["num-examples"])
        train_loss = float(client_metrics["train_loss"])
        client_id = client_metrics.get("client_id", "unknown")

        # Check for emergency (sent as int: 1=yes, 0=no)
        is_emergency = bool(client_metrics.get("is_emergency", 0))

        # Apply priority multiplier for emergencies
        if is_emergency:
            effective_weight = num_examples * emergency_multiplier
            emergency_count += 1
            pathology_count = client_metrics.get("pathology_count", 0)
            total_pathologies += pathology_count

            emergency_clients.append({
                "client_id": client_id,
                "num_examples": num_examples,
                "effective_weight": effective_weight,
                "pathology_count": pathology_count,
            })

            print(
                f"{RED}[EMERGENCY] Client {client_id} - {pathology_count} pathologies, weight {num_examples} → {effective_weight}{RESET}")
        else:
            effective_weight = num_examples
            normal_clients.append({
                "client_id": client_id,
                "num_examples": num_examples,
                "effective_weight": effective_weight,
            })

        weighted_loss_sum += train_loss * effective_weight
        total_effective_weight += effective_weight

    weighted_loss = weighted_loss_sum / \
        total_effective_weight if total_effective_weight > 0 else 0.0

    # Log emergency statistics
    if emergency_count > 0:
        print(
            f"\n{RED}[Round Summary] {emergency_count}/{len(metrics)} clients with emergencies ({total_pathologies} total pathologies){RESET}")

        # Log to file if strategy is available
        if strategy:
            strategy._log_emergency_alert(
                current_round, emergency_clients, normal_clients)
            strategy.total_emergency_count += emergency_count
            strategy.round_emergency_count[current_round] = emergency_count

    return MetricRecord({"train_loss": weighted_loss})
