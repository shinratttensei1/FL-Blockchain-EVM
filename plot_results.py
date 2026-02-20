"""Plot all FL simulation metrics from outputs/results.json.

Generates publication-ready figures for 71-SCP-code classification:
  1. Convergence curves (Accuracy, F1, Loss, AUC over rounds)
  2. Superclass F1 evolution over rounds  (5 classes)
  3. Top-20 SCP codes F1 evolution over rounds
  4. Per-client evaluation comparison (all 10 IoT devices)
  5. Final-round superclass performance bar chart
  6. Final-round top-20 SCP codes bar chart
  7. Summary statistics + markdown report
"""

import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')

SC_NAMES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)
RESULTS_PATH = "outputs/results.json"


def load_results(path=RESULTS_PATH):
    """Parse results.json into global rounds and client-eval rounds."""
    global_rounds = {}
    client_evals = defaultdict(list)

    if not os.path.exists(path):
        print(f"Error: {path} not found.  Run `flwr run .` first!")
        sys.exit(1)

    round_counter = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict) and data.get("type") == "global":
                global_rounds[data["round"]] = data
            elif isinstance(data, list):
                round_counter += 1
                client_evals[round_counter] = data

    return global_rounds, client_evals


# ──────────────────────────────────────────────────────────────────────
# 1. Convergence Curves
# ──────────────────────────────────────────────────────────────────────


def plot_convergence(global_rounds):
    """4-panel convergence: Accuracy, F1-macro, Loss, AUC-ROC."""
    rounds = sorted(global_rounds.keys())
    acc = [global_rounds[r]["accuracy"] for r in rounds]
    f1m = [global_rounds[r].get("f1_macro", 0) for r in rounds]
    loss = [global_rounds[r]["loss"] for r in rounds]
    auc = [global_rounds[r].get("auc_macro", 0) for r in rounds]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        'FL-Blockchain-EVM: 10 IoT Devices — 71 SCP Codes',
        fontsize=14, fontweight='bold', y=0.98)

    for ax, vals, title, ylabel, color, marker in [
        (axes[0, 0], acc,  'Accuracy',            'Accuracy', 'b', 'o'),
        (axes[0, 1], f1m,  'F1-Score (macro, 71)', 'F1-macro', 'g', 's'),
        (axes[1, 0], loss, 'Global Loss (BCE)',    'Loss',     'r', '^'),
        (axes[1, 1], auc,  'AUC-ROC (macro)',      'AUC-ROC', 'm', 'D'),
    ]:
        ax.plot(rounds, vals, f'{color}-{marker}', linewidth=2, markersize=6)
        ax.set_title(title)
        ax.set_xlabel('Communication Round')
        ax.set_ylabel(ylabel)
        if ylabel != 'Loss':
            ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fname = METRICS_DIR / 'convergence_curves.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✔ Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# 2. Superclass F1 Evolution
# ──────────────────────────────────────────────────────────────────────


def plot_superclass_f1_evolution(global_rounds):
    """Line plot of superclass-level F1 across rounds."""
    rounds = sorted(global_rounds.keys())
    if "superclass_f1" not in global_rounds[rounds[0]]:
        print("⚠  superclass_f1 not in results — skipping.")
        return

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, sc in enumerate(SC_NAMES):
        vals = [global_rounds[r]["superclass_f1"][i] for r in rounds]
        ax.plot(rounds, vals, '-o', color=colors[i],
                linewidth=2, label=sc, markersize=5)

    ax.set_title('Superclass F1-Score Evolution Over FL Rounds',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('F1-Score')
    ax.set_ylim(0, 1.05)
    ax.legend(title='Diagnostic Superclass')
    ax.grid(True, alpha=0.3)
    fname = METRICS_DIR / 'superclass_f1_evolution.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✔ Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# 3. Top-20 SCP Codes F1 Evolution
# ──────────────────────────────────────────────────────────────────────


def plot_top_codes_f1_evolution(global_rounds, top_k=15):
    """Line plot of the top-K SCP codes (by final support) over rounds."""
    rounds = sorted(global_rounds.keys())
    last = global_rounds[rounds[-1]]

    if "per_class_f1" not in last or "scp_codes" not in last:
        print("⚠  per_class_f1 / scp_codes not in results — skipping.")
        return

    codes = last["scp_codes"]
    supports = np.array(last["per_class_support"])
    top_idx = np.argsort(-supports)[:top_k]

    fig, ax = plt.subplots(figsize=(14, 7))
    cmap = plt.cm.get_cmap('tab20', top_k)
    for rank, idx in enumerate(top_idx):
        vals = [global_rounds[r]["per_class_f1"][idx] for r in rounds]
        ax.plot(rounds, vals, '-o', color=cmap(rank),
                linewidth=1.5, label=f"{codes[idx]} (n={supports[idx]})",
                markersize=4)

    ax.set_title(f'Top-{top_k} SCP Codes — F1 Evolution',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('F1-Score')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2, loc='lower right')
    ax.grid(True, alpha=0.3)
    fname = METRICS_DIR / 'top_codes_f1_evolution.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✔ Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# 4. Per-Client Evaluation
# ──────────────────────────────────────────────────────────────────────


def plot_per_client_metrics(client_evals):
    """Bar chart of per-client metrics for the last round."""
    if not client_evals:
        print("⚠  No client eval data — skipping per-client plot.")
        return

    last_round = max(client_evals.keys())
    entries = client_evals[last_round]
    if not entries or not isinstance(entries[0], dict):
        print("⚠  Client eval data malformed — skipping.")
        return

    entries = sorted(entries, key=lambda e: e.get("client_id", 0))
    ids = [f"Device {e['client_id']}" for e in entries]

    acc = [e.get("eval_acc", 0) for e in entries]
    f1 = [e.get("eval_f1", 0) for e in entries]
    auc = [e.get("eval_auc", 0) for e in entries]

    x = np.arange(len(ids))
    w = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, acc, w, label='Accuracy', color='#3498db')
    ax.bar(x,     f1,  w, label='F1-macro (71)', color='#2ecc71')
    ax.bar(x + w, auc, w, label='AUC-ROC',  color='#e67e22')

    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score')
    ax.set_title(
        f'Per-Device Evaluation — Round {last_round} '
        f'({len(entries)} IoT Medical Edge Devices)',
        fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fname = METRICS_DIR / 'per_client_metrics.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✔ Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# 5. Final Superclass Bar Chart
# ──────────────────────────────────────────────────────────────────────


def plot_final_superclass(global_rounds):
    """Grouped bar chart for superclass metrics in the final round."""
    last = max(global_rounds.keys())
    g = global_rounds[last]
    if "superclass_f1" not in g:
        print("⚠  superclass data not in results — skipping.")
        return

    x = np.arange(5)
    w = 0.18
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x - 2*w, g["superclass_precision"], w,
           label='Precision',  color='#3498db')
    ax.bar(x - w,   g["superclass_recall"],    w,
           label='Recall',     color='#2ecc71')
    ax.bar(x,       g["superclass_f1"],        w,
           label='F1-Score',   color='#e67e22')
    ax.bar(x + w,   g["superclass_auc"],       w,
           label='AUC-ROC',   color='#9b59b6')

    # Add support counts as text
    for i in range(5):
        ax.text(i, max(g["superclass_f1"][i], g["superclass_auc"][i]) + 0.03,
                f'n={g["superclass_support"][i]}', ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(SC_NAMES)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score')
    ax.set_title(f'Superclass Performance — Final Round {last}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    fname = METRICS_DIR / 'final_superclass_metrics.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✔ Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# 6. Final Top SCP Codes Bar Chart
# ──────────────────────────────────────────────────────────────────────


def plot_final_top_codes(global_rounds, top_k=25):
    """Horizontal bar chart of top-K SCP codes by F1 in the final round."""
    last = max(global_rounds.keys())
    g = global_rounds[last]
    if "per_class_f1" not in g or "scp_codes" not in g:
        print("⚠  per_class_f1 / scp_codes missing — skipping.")
        return

    codes = g["scp_codes"]
    f1s = np.array(g["per_class_f1"])
    supports = np.array(g["per_class_support"])

    # Only codes with support > 0
    active = np.where(supports > 0)[0]
    active = active[np.argsort(-f1s[active])][:top_k]

    fig, ax = plt.subplots(figsize=(10, max(8, top_k * 0.35)))
    y_pos = np.arange(len(active))
    bars = ax.barh(y_pos, [f1s[i] for i in active], color='#3498db')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [f"{codes[i]} (n={supports[i]})" for i in active], fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('F1-Score')
    ax.set_title(f'Top-{top_k} SCP Codes by F1 — Final Round {last}',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    fname = METRICS_DIR / 'final_top_codes_f1.png'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✔ Saved {fname}")


# ──────────────────────────────────────────────────────────────────────
# 7. Summary Report
# ──────────────────────────────────────────────────────────────────────


def print_summary(global_rounds, client_evals):
    """Print a final summary table."""
    if not global_rounds:
        print("No global results to summarize.")
        return

    last = max(global_rounds.keys())
    g = global_rounds[last]
    n_codes = g.get("num_classes", 71)

    print("\n" + "═" * 70)
    print("  FL-Blockchain-EVM — SIMULATION SUMMARY")
    print("═" * 70)
    print(f"  Total communication rounds : {len(global_rounds)}")
    print(f"  Number of IoT devices      : 10")
    print(f"  Dataset                     : PTB-XL (12-lead ECG)")
    print(f"  Total SCP codes classified  : {n_codes}")
    print(f"  Balancing                   : ROS+RUS (Jimenez et al., 2024)")
    print("─" * 70)
    print(f"  Final Accuracy              : {g['accuracy']:.4f}")
    print(f"  Final F1-macro ({n_codes} codes)   : {g.get('f1_macro', 0):.4f}")
    print(f"  Final F1-weighted           : {g.get('f1_weighted', 0):.4f}")
    print(f"  Final Precision (macro)     : {g.get('precision_macro', 0):.4f}")
    print(f"  Final Recall (macro)        : {g.get('recall_macro', 0):.4f}")
    print(
        f"  Final Specificity (macro)   : {g.get('specificity_macro', 0):.4f}")
    print(f"  Final AUC-ROC (macro)       : {g.get('auc_macro', 0):.4f}")
    print(f"  Final Loss                  : {g['loss']:.4f}")

    if "superclass_f1" in g:
        print("─" * 70)
        print("  Superclass breakdown:")
        for i, sc in enumerate(SC_NAMES):
            print(f"    {sc:5s}  F1={g['superclass_f1'][i]:.4f}  "
                  f"P={g['superclass_precision'][i]:.4f}  "
                  f"R={g['superclass_recall'][i]:.4f}  "
                  f"AUC={g['superclass_auc'][i]:.4f}  "
                  f"(n={g['superclass_support'][i]})")

    if "per_class_f1" in g and "scp_codes" in g:
        codes = g["scp_codes"]
        f1s = np.array(g["per_class_f1"])
        sups = np.array(g["per_class_support"])
        active = np.where(sups > 0)[0]
        top15 = active[np.argsort(-f1s[active])[:15]]
        print("─" * 70)
        print("  Top-15 SCP codes by F1:")
        for idx in top15:
            print(f"    {codes[idx]:8s}  F1={f1s[idx]:.4f}  (n={sups[idx]})")

    print("═" * 70)

    # ── Markdown report ──
    report = METRICS_DIR / "summary_report.md"
    with open(report, "w") as f:
        f.write("# FL-Blockchain-EVM — Simulation Summary\n\n")
        f.write(f"| Metric | Value |\n|--------|-------|\n")
        f.write(f"| Rounds | {len(global_rounds)} |\n")
        f.write(f"| IoT Devices | 10 |\n")
        f.write(f"| Dataset | PTB-XL (12-lead ECG) |\n")
        f.write(f"| SCP codes classified | {n_codes} |\n")
        f.write(f"| Balancing | ROS+RUS (β=0.8) |\n")
        f.write(f"| Final Accuracy | {g['accuracy']:.4f} |\n")
        f.write(f"| Final F1-macro | {g.get('f1_macro', 0):.4f} |\n")
        f.write(f"| Final F1-weighted | {g.get('f1_weighted', 0):.4f} |\n")
        f.write(f"| Final Precision | {g.get('precision_macro', 0):.4f} |\n")
        f.write(f"| Final Recall | {g.get('recall_macro', 0):.4f} |\n")
        f.write(
            f"| Final Specificity | {g.get('specificity_macro', 0):.4f} |\n")
        f.write(f"| Final AUC-ROC | {g.get('auc_macro', 0):.4f} |\n")
        f.write(f"| Final Loss | {g['loss']:.4f} |\n\n")

        if "superclass_f1" in g:
            f.write("## Superclass Breakdown (Final Round)\n\n")
            f.write("| Class | Precision | Recall | F1 | AUC | Support |\n")
            f.write("|-------|-----------|--------|----|-----|---------|\n")
            for i, sc in enumerate(SC_NAMES):
                f.write(
                    f"| {sc} | {g['superclass_precision'][i]:.4f} "
                    f"| {g['superclass_recall'][i]:.4f} "
                    f"| {g['superclass_f1'][i]:.4f} "
                    f"| {g['superclass_auc'][i]:.4f} "
                    f"| {g['superclass_support'][i]} |\n")

        if "per_class_f1" in g and "scp_codes" in g:
            codes = g["scp_codes"]
            f.write("\n## All SCP Codes (Final Round)\n\n")
            f.write("| Code | F1 | Precision | Recall | AUC | Support |\n")
            f.write("|------|----|-----------|--------|-----|---------|\n")
            sups = np.array(g["per_class_support"])
            for idx in np.argsort(-sups):
                if sups[idx] == 0:
                    continue
                f.write(
                    f"| {codes[idx]} "
                    f"| {g['per_class_f1'][idx]:.4f} "
                    f"| {g['per_class_precision'][idx]:.4f} "
                    f"| {g['per_class_recall'][idx]:.4f} "
                    f"| {g['per_class_auc'][idx]:.4f} "
                    f"| {sups[idx]} |\n")

    print(f"\n✔ Summary report written to {report}")


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────


def main():
    global_rounds, client_evals = load_results()
    if not global_rounds:
        print("No global metrics found in results.json.")
        sys.exit(1)

    plot_convergence(global_rounds)
    plot_superclass_f1_evolution(global_rounds)
    plot_top_codes_f1_evolution(global_rounds)
    plot_per_client_metrics(client_evals)
    plot_final_superclass(global_rounds)
    plot_final_top_codes(global_rounds)
    print_summary(global_rounds, client_evals)


if __name__ == "__main__":
    main()
