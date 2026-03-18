#!/usr/bin/env python3
"""Run Flower simulation directly."""

import flwr as fl
from fl_blockchain_evm.server_app import app as server_app
from fl_blockchain_evm.client_app import app as client_app


def main():
    """Run the simulation."""
    print("Starting Flower simulation...")

    # Run simulation with ClientApp API
    try:
        fl.simulation.run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=10,
            backend_config={
                "client_resources": {
                    "num_cpus": 1,
                    "num_gpus": 0
                }
            },
        )
        print("Simulation completed successfully!")
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
