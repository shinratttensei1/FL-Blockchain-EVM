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


def main() -> None:
    print("=" * 58)
    print("  FL·CHAIN — Management Server")
    print("=" * 58)
    print("  Management console : http://localhost:8080/")
    print("  Live monitor       : http://localhost:8080/monitor")
    print("  API docs           : http://localhost:8080/docs")
    print("  Press Ctrl+C to stop")
    print("=" * 58)

    try:
        import uvicorn
        from fl_blockchain_evm.management.server import app

        uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
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
