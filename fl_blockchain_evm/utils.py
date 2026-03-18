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

    # Print CUDA information for debugging
    if torch.cuda.is_available():
        print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        try:
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
            # Check CUDA version info
            try:
                import torch.version
                print(f"PyTorch CUDA version: {torch.version.cuda}")
            except:
                print("Could not get PyTorch CUDA version")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
        except Exception as e:
            print(f"Error getting CUDA info: {e}")

        # Try CUDA with more comprehensive testing
        try:
            # Test CUDA by creating and operating on tensors
            test_tensor = torch.randn(10, 10, device='cuda:0')
            result = test_tensor @ test_tensor.t()  # Matrix multiplication
            torch.cuda.empty_cache()
            print("CUDA test passed - using GPU")
            return torch.device("cuda:0")
        except RuntimeError as e:
            print(f"CUDA device available but not usable: {e}")
            print("This usually means:")
            print("1. PyTorch was compiled for different CUDA version")
            print("2. GPU architecture not supported by this PyTorch build")
            print("3. CUDA driver/runtime mismatch")
            print("Falling back to CPU")
        except Exception as e:
            print(f"CUDA test failed: {e}")
            print("Falling back to CPU")

    # Try MPS (Apple Silicon) - but user wants NVIDIA GPU
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS available but user requested NVIDIA GPU - skipping")
        # Don't use MPS since user wants NVIDIA

    print("Using CPU - if you want GPU, fix CUDA compatibility")
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
