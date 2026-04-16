#!/usr/bin/env python3
"""Run Flower server for distributed federated learning (not simulation).

This server listens on the network and accepts connections from remote
Flower clients (e.g., on Raspberry Pis). The server handles:
  - Model aggregation (MedicalFedAvg strategy)
  - Global evaluation
  - Blockchain recording (async fire-and-wait)
  - Dashboard state updates
"""

import os
import sys
from pathlib import Path

import flwr as fl

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fl_blockchain_evm.server_app import app as server_app
from fl_blockchain_evm.utils import get_device


def main():
    """Start Flower server in distributed mode (not simulation)."""

    # Parse config
    num_rounds = int(os.getenv("NUM_ROUNDS", "5"))
    server_address = os.getenv("SERVER_ADDRESS", "0.0.0.0:8080")

    print("\n" + "="*70)
    print("  FEDERATED LEARNING SERVER (Distributed Mode)")
    print("="*70)
    print(f"  Server listening on: {server_address}")
    print(f"  Number of rounds: {num_rounds}")
    print(f"  Device: {get_device()}")
    print("="*70)
    print("  Waiting for clients to connect...")
    print("  Pis should run:")
    print("    flower-client-app --server-address=<YOUR_IP>:8080 --client-id=0")
    print("="*70 + "\n")

    # Start server
    fl.server.start_server(
        server_app=server_app,
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )


if __name__ == "__main__":
    main()
