"""Shared utility functions for the FL-Blockchain-EVM project.

Common helpers used across the server/client apps, dashboard, and scripts.
"""

import json
import os
from typing import List, Dict, Any, Optional

import torch


def get_device(force: Optional[str] = None) -> torch.device:
    """Select the best available compute device.

    Args:
        force: If provided, use this device string directly.
    """
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_table(header, rows, cols):
    """Print a formatted ASCII table to stdout."""
    widths = [max(len(str(r[i])) for r in [header] + rows) + 2
              for i in range(len(cols))]
    sep = "+" + "+".join("-" * w for w in widths) + "+"

    def _row(r):
        return "|" + "|".join(str(r[i]).center(w)
                              for i, w in enumerate(widths)) + "|"
    print(sep)
    print(_row(header))
    print(sep)
    for r in rows:
        print(_row(r))
    print(sep)


def load_results(path: str = "outputs/results.json") -> List[Dict[str, Any]]:
    """Read a JSONL results file, returning a flat list of records.

    Handles both single JSON objects and JSON arrays per line,
    flattening arrays into the returned list.
    """
    if not os.path.exists(path):
        return []
    records: List[Dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, list):
                    records.extend(obj)
                else:
                    records.append(obj)
            except json.JSONDecodeError:
                pass
    return records
