#!/usr/bin/env python3
"""Prepare MHEALTH data for federated learning (synthetic or real)."""

import os
import sys
from pathlib import Path

def prepare_synthetic():
    """Generate synthetic data for testing."""
    import numpy as np

    print("\n[1] Creating data directory...")
    data_dir = Path("data/MHEALTHDATASET/.npy_cache")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"    ✓ {data_dir}")

    print("\n[2] Generating synthetic data...")
    subjects = [1, 2, 9, 10]

    for subject in subjects:
        print(f"    → Subject {subject}...", end=" ")
        X = np.random.randn(10000, 23).astype(np.float32)
        y = np.random.randint(0, 12, 10000).astype(np.int32)

        np.save(f"data/MHEALTHDATASET/.npy_cache/s{subject}_data.npy", X)
        np.save(f"data/MHEALTHDATASET/.npy_cache/s{subject}_labels.npy", y)
        print("✓")

    print("\n[3] Summary:")
    for subject in subjects:
        size = os.path.getsize(f"data/MHEALTHDATASET/.npy_cache/s{subject}_data.npy") / 1024 / 1024
        print(f"    Subject {subject}: {size:.1f} MB")

def prepare_real():
    """Download real MHEALTH data."""
    import urllib.request

    print("\n[1] Creating data directory...")
    data_dir = Path("data/MHEALTHDATASET")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"    ✓ {data_dir}")

    print("\n[2] Downloading MHEALTH data...")
    print("    (This takes ~5-10 minutes, ~150MB)\n")

    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/MHEALTH"
    subjects = [1, 2, 9, 10]

    for subject in subjects:
        filename = f"mHealth_subject{subject}.log"
        filepath = data_dir / filename
        url = f"{base_url}/{filename}"

        print(f"    → {filename}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, filepath)
            size = os.path.getsize(filepath) / 1024 / 1024
            print(f"✓ ({size:.1f} MB)")
        except Exception as e:
            print(f"✗ Error: {e}")
            return False

    print("\n[3] Summary:")
    for subject in subjects:
        filepath = data_dir / f"mHealth_subject{subject}.log"
        if filepath.exists():
            size = os.path.getsize(filepath) / 1024 / 1024
            print(f"    Subject {subject}: {size:.1f} MB")

    return True

def main():
    print("="*50)
    print("  MHEALTH Data Preparation")
    print("="*50)

    print("\nOptions:")
    print("  [1] Synthetic (fast, for testing)")
    print("  [2] Real MHEALTH (slow, for training)")
    print("  [3] Exit")
    print("")

    choice = input("Choose [1-3]: ").strip()

    if choice == "1":
        prepare_synthetic()
        print("\n✓ Ready! You can now deploy to Pis.")
    elif choice == "2":
        if prepare_real():
            print("\n✓ Ready! You can now deploy to Pis.")
        else:
            print("\n✗ Download failed. Check your internet connection.")
            return 1
    elif choice == "3":
        print("Cancelled.")
        return 0
    else:
        print("Invalid choice.")
        return 1

    print("\nNext step:")
    print("  bash scripts/deploy_to_pis.sh")
    print("")
    return 0

if __name__ == "__main__":
    sys.exit(main())
