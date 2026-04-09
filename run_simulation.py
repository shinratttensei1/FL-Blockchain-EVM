#!/usr/bin/env python3
"""Run Flower simulation directly."""

import flwr as fl
from fl_blockchain_evm.server_app import app as server_app
from fl_blockchain_evm.client_app import app as client_app


def main():
    """Run the simulation."""
    print("Starting Flower simulation...")

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=8,
        backend_config={
            "client_resources": {
                "num_cpus": 2,
                "num_gpus": 0.0,
            }
        },
    )
    print("Simulation completed successfully!")


if __name__ == "__main__":
    main()
