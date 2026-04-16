#!/usr/bin/env python3
"""Validate that distributed FL setup is ready to deploy."""

import os
import sys
from pathlib import Path


def check(condition, message):
    """Print check result."""
    status = "✓" if condition else "✗"
    print(f"{status} {message}")
    return condition


def main():
    print("\n" + "="*60)
    print("  FL-Blockchain-EVM: Setup Validation")
    print("="*60 + "\n")

    all_good = True
    project_root = Path(__file__).parent.parent

    # Check Python
    print("[1] Python & Dependencies")
    all_good &= check(sys.version_info >= (3, 9), "Python 3.9+")

    try:
        import torch
        all_good &= check(True, "PyTorch installed")
    except ImportError:
        all_good &= check(False, "PyTorch installed")

    try:
        import flwr
        all_good &= check(True, "Flower installed")
    except ImportError:
        all_good &= check(False, "Flower installed")

    # Check files
    print("\n[2] Required Files")
    files_to_check = [
        "run_server.py",
        "run_dashboard.py",
        "fl_blockchain_evm/client_app.py",
        "fl_blockchain_evm/server_app.py",
        "requirements.txt",
        "requirements-pi.txt",
        "requirements-server.txt",
        "scripts/setup_pi.sh",
        "scripts/deploy_to_pis.sh",
        "scripts/start_server.sh",
        "scripts/start_client_pi.sh",
        "DEPLOYMENT.md",
        "DATA_PREPARATION.md",
    ]

    for fname in files_to_check:
        fpath = project_root / fname
        exists = fpath.exists()
        all_good &= check(exists, f"File exists: {fname}")

    # Check executability
    print("\n[3] Script Permissions")
    scripts = [
        "scripts/setup_pi.sh",
        "scripts/deploy_to_pis.sh",
        "scripts/start_server.sh",
        "scripts/start_client_pi.sh",
    ]

    for script in scripts:
        spath = project_root / script
        if spath.exists():
            is_exec = os.access(spath, os.X_OK)
            all_good &= check(is_exec, f"Executable: {script}")
        else:
            all_good &= check(False, f"Executable: {script}")

    # Check data
    print("\n[4] Data & Configuration")

    data_dir = project_root / "data" / "MHEALTHDATASET"
    has_data = data_dir.exists() and len(list(data_dir.glob("*.log"))) > 0
    check(has_data, "MHEALTH dataset found (optional)")

    env_file = project_root / ".env"
    check(env_file.exists(), ".env file exists (optional)")

    # Check imports
    print("\n[5] Module Imports")

    try:
        from fl_blockchain_evm.core.model import Net, FocalLoss
        all_good &= check(True, "Can import model")
    except Exception as e:
        all_good &= check(False, f"Can import model: {e}")

    try:
        from fl_blockchain_evm.client_app import app as client_app
        all_good &= check(True, "Can import client app")
    except Exception as e:
        all_good &= check(False, f"Can import client app: {e}")

    try:
        from fl_blockchain_evm.server_app import app as server_app
        all_good &= check(True, "Can import server app")
    except Exception as e:
        all_good &= check(False, f"Can import server app: {e}")

    # Summary
    print("\n" + "="*60)
    if all_good:
        print("  ✓ Setup validation PASSED")
        print("  Ready to deploy to Pis!")
    else:
        print("  ✗ Setup validation FAILED")
        print("  Some checks failed - review above")
    print("="*60 + "\n")

    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main())
