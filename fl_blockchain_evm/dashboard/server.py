"""
fl_dashboard_server.py
----------------------
Lightweight FastAPI backend for the FL training dashboard.

Reads outputs/results.json (written by server_app.py) and queries the
deployed SimpleFLBlockchain smart contract to serve live data to the
frontend via a simple REST + SSE API.

Run:
    pip install fastapi uvicorn web3 python-dotenv
    python fl_dashboard_server.py
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse

from dotenv import load_dotenv
from web3 import Web3
from fl_blockchain_evm.utils import load_results as _load_results

load_dotenv()

app = FastAPI(title="FL Blockchain Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_FILE = Path("outputs/results.json")

# ─────────────────────────────────────────────────────────────
# Blockchain connection (optional — gracefully degrades if unavailable)
# ─────────────────────────────────────────────────────────────


def _init_blockchain():
    rpc_url = os.getenv("BASE_SEPOLIA_RPC_URL")
    contract_addr = os.getenv("CONTRACT_ADDRESS")
    if not rpc_url or not contract_addr:
        return None, None
    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not w3.is_connected():
            return None, None
        abi_path = Path("contracts/FLBlockchain_abi.json")
        if not abi_path.exists():
            return None, None
        with open(abi_path, encoding="utf-8") as f:
            abi = json.load(f)
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(contract_addr), abi=abi
        )
        return w3, contract
    except Exception:
        return None, None


_w3, _contract = _init_blockchain()

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


def _blockchain_state() -> Dict[str, Any]:
    if _contract is None:
        return {"connected": False, "chain_length": 0, "chain_valid": None, "recent_blocks": []}
    try:
        chain_length = _contract.functions.getBlockCount().call()
        chain_valid = _contract.functions.verifyChain().call()
        # Fetch last 6 blocks (skip genesis if chain short)
        recent = []
        start = max(0, chain_length - 6)
        for i in range(start, chain_length):
            # Use blocks mapping instead of getBlock function
            b = _contract.functions.blocks(i).call()
            recent.append({
                "index":       b[0],
                "fl_round":    b[1],
                "block_type":  b[2],
                "content_hash": b[3].hex(),
                "prev_hash":   b[4].hex(),
                "timestamp":   b[5],
                "submitter":   b[6],
            })
        return {
            "connected":    True,
            "chain_length": chain_length,
            "chain_valid":  chain_valid,
            "recent_blocks": list(reversed(recent)),
        }
    except Exception as e:
        return {"connected": False, "chain_length": 0, "chain_valid": None,
                "recent_blocks": [], "error": str(e)}


def _build_dashboard_state() -> Dict[str, Any]:
    import numpy as np
    records = _load_results(str(RESULTS_FILE))

    rounds: Dict[int, Dict] = {}
    for r in records:
        rnd = r.get("round", 0)
        if rnd not in rounds:
            rounds[rnd] = {"round": rnd, "training": None,
                           "global": None, "client_evals": []}

        t = r.get("type")
        if t == "device_training":
            devices = r.get("devices", [])
            # Normalise training_time field: original code writes
            # "training_time_seconds", refactored writes "training_time"
            for d in devices:
                if "training_time" not in d and "training_time_seconds" in d:
                    d["training_time"] = d["training_time_seconds"]
            # Compute threshold if not stored (original server_app omits it)
            if "threshold" not in r and devices:
                losses = [d["train_loss"] for d in devices]
                mean = float(np.mean(losses))
                std = float(np.std(losses)) if len(losses) > 1 else 0.0
                r["threshold"] = mean + std
                r["loss_mean"] = mean
                r["loss_std"] = std
            rounds[rnd]["training"] = r

        elif t == "global":
            rounds[rnd]["global"] = r

        elif t == "client_eval":
            rounds[rnd]["client_evals"].append(r)

    # Status logic:
    #   idle     — no results.json data yet
    #   training — latest round has training record but no global record yet
    #   complete — all rounds have global records
    latest_round = max(rounds.keys()) if rounds else 0
    latest = rounds.get(latest_round, {})

    if not records:
        status = "idle"
    elif latest.get("global") is not None:
        status = "complete" if all(
            rounds[r].get("global") is not None for r in rounds
        ) else "training"
    else:
        status = "training"

    # History for the line chart
    history = []
    for rnd in sorted(rounds.keys()):
        g = rounds[rnd].get("global")
        if g:
            history.append({
                "round":     rnd,
                "accuracy":  g.get("accuracy",  0),
                "f1_macro":  g.get("f1_macro",  0),
                "auc_macro": g.get("auc_macro", 0),
                "loss":      g.get("loss",      0),
            })

    blockchain = _blockchain_state()

    # IPFS summary — extract CIDs from global round records
    ipfs_summary = {"enabled": False, "total_pins": 0, "rounds": {}}
    for rnd_key in sorted(rounds.keys()):
        g = rounds[rnd_key].get("global")
        if g and g.get("ipfs_cids"):
            ipfs_summary["enabled"] = True
            ipfs_summary["rounds"][str(rnd_key)] = g["ipfs_cids"]
            ipfs_summary["total_pins"] += len(g["ipfs_cids"])

    return {
        "status":       status,
        "latest_round": latest_round,
        "total_rounds": len(rounds),
        "rounds":       {str(k): v for k, v in rounds.items()},
        "history":      history,
        "blockchain":   blockchain,
        "ipfs":         ipfs_summary,
        "last_updated": datetime.now().isoformat(),
    }

# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────


@app.get("/api/state")
def get_state():
    return _build_dashboard_state()


@app.get("/api/blockchain")
def get_blockchain(limit: int = 6):
    """Get blockchain data."""
    blockchain_state = _blockchain_state()
    # Limit the number of blocks returned
    if blockchain_state.get("recent_blocks"):
        blockchain_state["recent_blocks"] = blockchain_state["recent_blocks"][-limit:]
    blockchain_state["blocks"] = blockchain_state.pop("recent_blocks", [])
    return blockchain_state


@app.get("/api/stream")
def stream():
    """Server-Sent Events endpoint — pushes state every 3 seconds."""
    def event_generator():
        while True:
            try:
                data = json.dumps(_build_dashboard_state())
                yield f"data: {data}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            time.sleep(3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the dashboard HTML."""
    html_path = Path(__file__).parent / "fl_dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    # Fallback to topology-only view
    html_path = Path(__file__).parent / "fl_topology.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse("<h1>Dashboard HTML not found</h1>")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)