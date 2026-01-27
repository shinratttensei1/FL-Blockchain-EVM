import json
import matplotlib.pyplot as plt


def plot_fl_results(manifest_path="results.json"):
    rounds = []
    avg_accuracies = []

    with open(manifest_path, "r") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            round_acc = sum(c['accuracy'] for c in data) / len(data)
            rounds.append(i + 1)
            avg_accuracies.append(round_acc)

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, avg_accuracies, label='Aggregated Client Accuracy',
             color='#2c3e50', linewidth=2)

    plt.title(
        'Federated Learning Convergence: Rounds vs. Accuracy (FEMNIST)', fontsize=14)
    plt.xlabel('Communication Rounds', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.savefig('rounds_vs_accuracy.png', dpi=300)
    print("Diagram saved as rounds_vs_accuracy.png")
    plt.show()


if __name__ == "__main__":
    plot_fl_results()
