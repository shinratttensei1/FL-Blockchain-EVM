#!/usr/bin/env python3
"""Run the FL Blockchain Dashboard."""

import sys


def main():
    """Run the dashboard server."""
    print("Starting FL Blockchain Dashboard...")
    print("Dashboard will be available at: http://localhost:8000")
    print("Press Ctrl+C to stop")

    try:
        # Run the dashboard server
        from fl_blockchain_evm.dashboard.server import app
        import uvicorn

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False
        )
    except ImportError as e:
        print(f"Import error: {e}")
        print("\nMake sure all dependencies are installed:")
        print("  pip install fastapi uvicorn web3 python-dotenv")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDashboard stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
