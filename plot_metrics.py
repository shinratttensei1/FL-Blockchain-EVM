"""Quick convergence plot (Accuracy + F1 + Loss) from results.json.

This is the lightweight version of plot_results.py for quick checks.
"""

import matplotlib.pyplot as plt
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')

RESULTS_PATH = "outputs/results.json"

if not os.path.exists(RESULTS_PATH):
    print(f"Error: {RESULTS_PATH} not found. Run `flwr run .` first!")
    sys.exit(1)

rounds, acc, f1, loss, auc = [], [], [], [], []

with open(RESULTS_PATH) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("type") == "global":
            rounds.append(data["round"])
            acc.append(data["accuracy"])
            f1.append(data.get("f1_macro", data.get("f1_score", 0)))
            loss.append(data["loss"])
            auc.append(data.get("auc_macro", 0))

if not rounds:
    print("No global metrics found.")
    sys.exit(1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    'Medical IoT Swarm: 10 Edge Devices — FL Performance',
    fontsize=14, fontweight='bold',
)

# Accuracy & F1
ax = axes[0]
ax.plot(rounds, acc, 'b-o', label='Accuracy', linewidth=2)
ax.plot(rounds, f1, 'g-s', label='F1-macro', linewidth=2)
ax.set_title('Learning Performance')
ax.set_xlabel('Communication Round')
ax.set_ylabel('Score (0–1)')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend()

# Loss
ax = axes[1]
ax.plot(rounds, loss, 'r-^', label='Global Loss', linewidth=2)
ax.set_title('Convergence Rate (Focal Loss)')
ax.set_xlabel('Communication Round')
ax.set_ylabel('Loss')
ax.grid(True, alpha=0.3)
ax.legend()

# AUC
ax = axes[2]
ax.plot(rounds, auc, 'm-D', label='AUC-ROC macro', linewidth=2)
ax.set_title('AUC-ROC Convergence')
ax.set_xlabel('Communication Round')
ax.set_ylabel('AUC-ROC')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.93])
out = "outputs/simulation_metrics.png"
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print(f"✔ Graph saved to: {out}")
