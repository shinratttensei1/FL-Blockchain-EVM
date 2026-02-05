import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def load_results(file_path="outputs/results.json"):
    """Load and parse results.json file"""
    client_rounds = {}
    global_metrics = {}
    client_losses = {}

    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)

                if isinstance(data, list):
                    round_num = len(client_rounds) + 1
                    avg_acc = sum(c['accuracy'] for c in data) / len(data)
                    avg_loss = sum(c.get('loss', 0) for c in data) / \
                        len(data) if any('loss' in c for c in data) else None
                    client_rounds[round_num] = avg_acc
                    if avg_loss is not None:
                        client_losses[round_num] = avg_loss

                elif isinstance(data, dict) and data.get("type") == "global":
                    r_num = data.get("round")
                    global_metrics[r_num] = {
                        'accuracy': data.get("accuracy"),
                        'loss': data.get("loss")
                    }
            except json.JSONDecodeError:
                continue

    return client_rounds, global_metrics, client_losses


def plot_convergence_curves(client_rounds, global_metrics, save_path='metrics/convergence_curves.png'):
    """Plot accuracy and loss convergence over rounds"""
    rounds = sorted(client_rounds.keys())
    c_acc = [client_rounds[r] for r in rounds]
    g_acc = [global_metrics.get(r, {}).get('accuracy', 0) for r in rounds]
    g_loss = [global_metrics.get(r, {}).get('loss', 0) for r in rounds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy plot
    ax1.plot(rounds, c_acc, label='Aggregated Client Eval',
             color='#3498db', linewidth=2, alpha=0.7)
    ax1.plot(rounds, g_acc, label='Global Server Eval',
             color='#e74c3c', linestyle='--', linewidth=2.5)
    ax1.set_title('Accuracy Convergence (FEMNIST)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Communication Round', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Loss plot
    ax2.plot(rounds, g_loss, label='Global Loss',
             color='#9b59b6', linewidth=2.5)
    ax2.set_title('Global Loss Convergence', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Communication Round', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return rounds[-1], g_acc[-1], g_loss[-1]


def generate_summary_report(client_rounds, global_metrics):
    """Generate a text summary report"""
    pass


def plot_all_visualizations():
    """Generate convergence visualization"""
    # Create metrics directory if it doesn't exist
    metrics_dir = Path('metrics')
    metrics_dir.mkdir(exist_ok=True)

    # Load data
    client_rounds, global_metrics, client_losses = load_results(
        "outputs/results.json")

    # Generate convergence plot
    final_round, final_acc, final_loss = plot_convergence_curves(
        client_rounds, global_metrics)

    # Generate summary
    generate_summary_report(client_rounds, global_metrics)


def plot_comparison(file_path="results.json"):
    """Legacy function - kept for backward compatibility"""
    client_rounds, global_metrics, _ = load_results(file_path)

    rounds = sorted(client_rounds.keys())
    c_acc = [client_rounds[r] for r in rounds]
    g_acc = [global_metrics.get(r, {}).get('accuracy', 0) for r in rounds]

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, c_acc, label='Aggregated Client Eval (Reported)',
             color='#3498db', linewidth=2)
    plt.plot(rounds, g_acc, label='Global Server Eval (Verified)',
             color='#e74c3c', linestyle='--', linewidth=2)

    plt.title(
        'Comparison: Client Reporting vs. Global Verification (FEMNIST)', fontsize=14)
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('eval_vs_global.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    # Generate all visualizations for professor demonstration
    plot_all_visualizations()
