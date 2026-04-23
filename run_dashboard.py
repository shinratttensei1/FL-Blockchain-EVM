#!/usr/bin/env python3
"""Run the FL-Blockchain Management Server.

Serves two UIs on http://localhost:8080:
  /          → Management console  (devices, contracts, training control)
  /monitor   → Live metrics dashboard (topology, charts, blockchain ledger)

API reference:
  /docs      → Interactive Swagger UI
  /redoc     → ReDoc reference
"""

import sys
import os


def main() -> None:
  port = int(os.getenv("DASHBOARD_PORT", os.getenv("PORT", "8080")))

    print("=" * 58)
    print("  FL·CHAIN — Management Server")
    print("=" * 58)
  print(f"  Management console : http://localhost:{port}/")
  print(f"  Live monitor       : http://localhost:{port}/monitor")
  print(f"  API docs           : http://localhost:{port}/docs")
    print("  Press Ctrl+C to stop")
    print("=" * 58)

    try:
        import uvicorn
        from fl_blockchain_evm.management.server import app

        uvicorn.run(app, host="0.0.0.0", port=port, reload=False)
    except ImportError as exc:
        print(f"\nImport error: {exc}")
        print("\nInstall missing packages:")
        print("  pip install fastapi uvicorn web3 python-dotenv pydantic")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
