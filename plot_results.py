import json
import matplotlib.pyplot as plt


def plot_comparison(file_path="results.json"):
    client_rounds = {}
    global_accuracies = {}

    with open(file_path, "r") as f:
        for line in f:
            try:
                data = json.loads(line)

                if isinstance(data, list):
                    round_num = len(client_rounds) + 1
                    avg_acc = sum(c['accuracy'] for c in data) / len(data)
                    client_rounds[round_num] = avg_acc

                elif isinstance(data, dict) and data.get("type") == "global":
                    r_num = data.get("round")
                    global_accuracies[r_num] = data.get("accuracy")
                    
            except json.JSONDecodeError:
                continue

    rounds = sorted(client_rounds.keys())
    c_acc = [client_rounds[r] for r in rounds]
    g_acc = [global_accuracies.get(r, 0) for r in rounds]

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
    print(f"Plot saved! Processed {len(rounds)} rounds.")
    plt.show()


if __name__ == "__main__":
    plot_comparison()
