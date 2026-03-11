"""Plot data distribution across 10 simulated IoT medical edge devices.

Generates TWO plots:
  1. Superclass distribution (stacked bar — 5 groups)
  2. Top-20 SCP code distribution (grouped bar — per device)
"""

from fl_blockchain_evm.core.constants import (
    SC_NAMES, DIAG_NORM, DIAG_MI, DIAG_STTC, DIAG_CD, DIAG_HYP,
)
import matplotlib.pyplot as plt
import os
import ast
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')


NUM_CLIENTS = 10
DATA_DIR = "data/ptb-xl"

_all_diag = DIAG_NORM + DIAG_MI + DIAG_STTC + DIAG_CD + DIAG_HYP

FORM_CODES = ['ABQRS', 'PVC', 'STD_', 'VCLVH', 'QWAVE', 'LOWT', 'NT_',
              'PAC', 'LPR', 'INVT', 'LVOLT', 'HVOLT', 'TAB_', 'STE_',
              'PRC(S)', 'NDT', 'NST_', 'DIG', 'LNGQT']
RHYTHM_CODES = ['SR', 'AFIB', 'STACH', 'SARRH', 'SBRAD', 'PACE',
                'SVARR', 'BIGU', 'AFLT', 'SVTAC', 'PSVT', 'TRIGU']

_form_only = [c for c in FORM_CODES if c not in _all_diag]
_rhythm_only = RHYTHM_CODES

ALL_SCP_CODES = _all_diag + _form_only + _rhythm_only
CODE_TO_IDX = {c: i for i, c in enumerate(ALL_SCP_CODES)}

SC_GROUPS = {
    'NORM': DIAG_NORM, 'MI': DIAG_MI, 'STTC': DIAG_STTC,
    'CD': DIAG_CD, 'HYP': DIAG_HYP,
}


def load_partitions():
    """Replicate data-loading logic from task.py."""
    csv_path = os.path.join(DATA_DIR, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    df = pd.read_csv(csv_path, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    # Training fold only
    train_ids = df[df.strat_fold <= 8].index.values.copy()
    np.random.seed(42)
    np.random.shuffle(train_ids)

    size = len(train_ids) // NUM_CLIENTS

    partitions_sc = {}      # superclass counts per client
    partitions_code = {}    # per-SCP-code counts per client

    for cid in range(NUM_CLIENTS):
        start = cid * size
        end = (cid + 1) * size
        my_ids = train_ids[start:end]

        sc_counts = {sc: 0 for sc in SC_NAMES}
        code_counts = {c: 0 for c in ALL_SCP_CODES}

        for idx in my_ids:
            scp_dict = df.loc[idx].scp_codes
            for code in scp_dict:
                if code in CODE_TO_IDX:
                    code_counts[code] += 1
            # Superclass
            for sc, codes_list in SC_GROUPS.items():
                if any(c in scp_dict for c in codes_list):
                    sc_counts[sc] += 1

        partitions_sc[cid] = sc_counts
        partitions_code[cid] = code_counts

    return partitions_sc, partitions_code


def plot_superclass_distribution(partitions_sc):
    """Stacked bar chart of 5 superclasses across 10 devices."""
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    fig, ax = plt.subplots(figsize=(14, 6))

    bottom = np.zeros(NUM_CLIENTS)
    for i, sc in enumerate(SC_NAMES):
        vals = [partitions_sc[cid][sc] for cid in range(NUM_CLIENTS)]
        ax.bar(range(NUM_CLIENTS), vals, bottom=bottom,
               label=sc, color=colors[i])
        bottom += np.array(vals)

    ax.set_xlabel('IoT Medical Edge Device ID', fontsize=12)
    ax.set_ylabel('Number of ECG Records', fontsize=12)
    ax.set_title(
        'Superclass Distribution Across 10 IoT Devices\n'
        '(Training Set — Before ROS+RUS Balancing)',
        fontsize=13, fontweight='bold')
    ax.set_xticks(range(NUM_CLIENTS))
    ax.set_xticklabels([f'Device {i}' for i in range(NUM_CLIENTS)],
                       rotation=45, ha='right')
    ax.legend(title='Superclass', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fname = "outputs/device_superclass_distribution.png"
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✔ Saved {fname}")


def plot_top_codes_distribution(partitions_code, top_k=20):
    """Heatmap of top-K SCP codes across 10 devices."""
    # Aggregate across devices to find top-K
    total = {c: sum(partitions_code[cid][c] for cid in range(NUM_CLIENTS))
             for c in ALL_SCP_CODES}
    top_codes = sorted(total, key=total.get, reverse=True)[:top_k]

    # Build matrix
    matrix = np.zeros((top_k, NUM_CLIENTS), dtype=int)
    for j, cid in enumerate(range(NUM_CLIENTS)):
        for i, code in enumerate(top_codes):
            matrix[i, j] = partitions_code[cid][code]

    fig, ax = plt.subplots(figsize=(14, max(8, top_k * 0.4)))
    import seaborn as sns
    sns.heatmap(matrix, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=[f'Dev {i}' for i in range(NUM_CLIENTS)],
                yticklabels=[f"{c} (Σ={total[c]})" for c in top_codes],
                ax=ax)
    ax.set_title(f'Top-{top_k} SCP Codes — Distribution Across 10 Devices',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('IoT Device')

    fname = "outputs/device_code_distribution_heatmap.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✔ Saved {fname}")


def main():
    print("Loading PTB-XL data distribution for 10 IoT devices ...")
    partitions_sc, partitions_code = load_partitions()
    plot_superclass_distribution(partitions_sc)
    plot_top_codes_distribution(partitions_code)

    # Summary
    total_codes_active = sum(
        1 for c in ALL_SCP_CODES
        if sum(partitions_code[cid][c] for cid in range(NUM_CLIENTS)) > 0
    )
    print(
        f"\n  Total unique SCP codes with ≥1 sample: {total_codes_active}/{len(ALL_SCP_CODES)}")


if __name__ == "__main__":
    main()
