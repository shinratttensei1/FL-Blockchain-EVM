"""FL Management Server — FastAPI backend.

Provides a unified API for:
  - Edge device registry   (add / remove / toggle / list)
  - Smart contract registry (add / remove / activate / list)
  - Training configuration  (get / update)
  - Training lifecycle      (launch / stop / status)
  - Live FL metrics         (state / SSE stream)
  - Blockchain ledger       (recent blocks / chain verification)
  - System health           (health check / env inspection)

Run:
    python run_dashboard.py
    → http://localhost:8080          management console
    → http://localhost:8080/monitor  live metrics dashboard
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from web3 import Web3

from fl_blockchain_evm.management import store


def _load_results(path: str) -> list:
    """Read JSONL results file — avoids importing torch via utils."""
    import json as _json
    import os as _os
    if not _os.path.exists(path):
        return []
    records: list = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
                records.extend(obj) if isinstance(obj, list) else records.append(obj)
            except _json.JSONDecodeError:
                pass
    return records

load_dotenv()

app = FastAPI(title="FL-Blockchain Management API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RESULTS_FILE = Path("outputs/results.json")

# ── Training process state ─────────────────────────────────────

_training_proc: Optional[subprocess.Popen] = None
_training_log:  list = []          # last N lines of stdout
_training_start: Optional[float] = None


def _proc_status() -> Dict[str, Any]:
    global _training_proc
    if _training_proc is None:
        return {"running": False, "pid": None, "exit_code": None, "elapsed_s": None}
    rc = _training_proc.poll()
    elapsed = round(time.time() - _training_start, 1) if _training_start else None
    return {
        "running":    rc is None,
        "pid":        _training_proc.pid,
        "exit_code":  rc,
        "elapsed_s":  elapsed,
    }


# ── Blockchain helpers ─────────────────────────────────────────

def _make_w3_contract(rpc_url: str, contract_addr: str):
    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 6}))
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


def _blockchain_state(limit: int = 6) -> Dict[str, Any]:
    active = store.get_active_contract()
    if not active:
        env = store.read_env()
        rpc_url       = env.get("BASE_SEPOLIA_RPC_URL") or os.getenv("BASE_SEPOLIA_RPC_URL", "")
        contract_addr = env.get("CONTRACT_ADDRESS")     or os.getenv("CONTRACT_ADDRESS", "")
    else:
        rpc_url       = active["rpc_url"]
        contract_addr = active["address"]

    if not rpc_url or not contract_addr:
        return {"connected": False, "chain_length": 0, "chain_valid": None,
                "recent_blocks": [], "active_contract": None}

    _w3, _contract = _make_w3_contract(rpc_url, contract_addr)
    if _contract is None:
        return {"connected": False, "chain_length": 0, "chain_valid": None,
                "recent_blocks": [], "active_contract": contract_addr}

    try:
        chain_length = _contract.functions.getBlockCount().call()
        chain_valid  = _contract.functions.verifyChain().call()
        recent: list = []
        start = max(0, chain_length - limit)
        for i in range(start, chain_length):
            b = _contract.functions.blocks(i).call()
            recent.append({
                "index":        b[0],
                "fl_round":     b[1],
                "block_type":   b[2],
                "content_hash": b[3].hex(),
                "prev_hash":    b[4].hex(),
                "timestamp":    b[5],
                "submitter":    b[6],
            })
        return {
            "connected":       True,
            "chain_length":    chain_length,
            "chain_valid":     chain_valid,
            "recent_blocks":   list(reversed(recent)),
            "active_contract": contract_addr,
        }
    except Exception as exc:
        return {"connected": False, "chain_length": 0, "chain_valid": None,
                "recent_blocks": [], "error": str(exc),
                "active_contract": contract_addr}


# ── Dashboard state builder ────────────────────────────────────

def _build_dashboard_state() -> Dict[str, Any]:
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
            for d in devices:
                if "training_time" not in d and "training_time_seconds" in d:
                    d["training_time"] = d["training_time_seconds"]
            if "threshold" not in r and devices:
                losses = [d["train_loss"] for d in devices]
                mean = float(np.mean(losses))
                std  = float(np.std(losses)) if len(losses) > 1 else 0.0
                r["threshold"] = mean + std
                r["loss_mean"] = mean
                r["loss_std"]  = std
            rounds[rnd]["training"] = r
        elif t == "global":
            rounds[rnd]["global"] = r
        elif t == "client_eval":
            rounds[rnd]["client_evals"].append(r)

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

    # Override with live process status
    ps = _proc_status()
    if ps["running"]:
        status = "training"
    elif ps["exit_code"] is not None and status == "idle":
        status = "complete"

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

    ipfs_summary: Dict[str, Any] = {"enabled": False, "total_pins": 0, "rounds": {}}
    for rnd_key in sorted(rounds.keys()):
        g = rounds[rnd_key].get("global")
        if g and g.get("ipfs_cids"):
            ipfs_summary["enabled"] = True
            ipfs_summary["rounds"][str(rnd_key)] = g["ipfs_cids"]
            ipfs_summary["total_pins"] += len(g["ipfs_cids"])

    return {
        "status":        status,
        "latest_round":  latest_round,
        "total_rounds":  len(rounds),
        "rounds":        {str(k): v for k, v in rounds.items()},
        "history":       history,
        "blockchain":    blockchain,
        "ipfs":          ipfs_summary,
        "training_proc": ps,
        "last_updated":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ═══════════════════════════════════════════════════════════════
# REQUEST MODELS
# ═══════════════════════════════════════════════════════════════

class DeviceCreate(BaseModel):
    name:        str = Field(..., min_length=1, max_length=64)
    host:        str = Field(..., min_length=1, max_length=255)
    port:        int = Field(8765, ge=1, le=65535)
    type:        str = Field("sensor_node")
    description: str = Field("")

class DeviceUpdate(BaseModel):
    name:        Optional[str] = None
    host:        Optional[str] = None
    port:        Optional[int] = None
    type:        Optional[str] = None
    description: Optional[str] = None
    enabled:     Optional[bool] = None

class ContractCreate(BaseModel):
    address:  str = Field(..., min_length=42, max_length=42)
    network:  str = Field("Base Sepolia")
    rpc_url:  str = Field("https://sepolia.base.org")
    chain_id: int = Field(84532)
    name:     str = Field("")
    abi_path: str = Field("contracts/FLBlockchain_abi.json")

class TrainingConfig(BaseModel):
    # Common
    num_rounds:          Optional[int]   = None
    lr:                  Optional[float] = None
    local_epochs:        Optional[int]   = None
    fraction_train:      Optional[float] = None
    beta:                Optional[float] = None
    batch_size:          Optional[int]   = None
    # Mode
    run_mode:            Optional[str]   = None  # "simulation" | "federation"
    # Simulation-only
    num_supernodes:      Optional[int]   = None
    # Federation-only
    superlink_address:   Optional[str]   = None  # "host:port"
    superlink_insecure:  Optional[bool]  = None

class TrainingLaunch(BaseModel):
    """Override any config field at launch time (same fields as TrainingConfig)."""
    num_rounds:          Optional[int]   = None
    lr:                  Optional[float] = None
    local_epochs:        Optional[int]   = None
    fraction_train:      Optional[float] = None
    run_mode:            Optional[str]   = None
    num_supernodes:      Optional[int]   = None
    superlink_address:   Optional[str]   = None
    superlink_insecure:  Optional[bool]  = None


# ═══════════════════════════════════════════════════════════════
# DEVICE ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/devices", tags=["Devices"])
def get_devices():
    """List all registered edge devices."""
    return store.list_devices()


@app.post("/api/devices", status_code=201, tags=["Devices"])
def create_device(body: DeviceCreate):
    """Register a new edge device."""
    return store.add_device(
        name=body.name,
        host=body.host,
        port=body.port,
        device_type=body.type,
        description=body.description,
    )


@app.delete("/api/devices/{device_id}", tags=["Devices"])
def delete_device(device_id: str):
    """Remove a device by ID."""
    if not store.remove_device(device_id):
        raise HTTPException(status_code=404, detail="Device not found")
    return {"ok": True}


@app.patch("/api/devices/{device_id}/toggle", tags=["Devices"])
def toggle_device(device_id: str):
    """Toggle enabled/disabled state."""
    result = store.toggle_device(device_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Device not found")
    return result


@app.patch("/api/devices/{device_id}", tags=["Devices"])
def update_device(device_id: str, body: DeviceUpdate):
    """Update device fields (name, host, port, type, description, enabled)."""
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    result = store.update_device(device_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail="Device not found")
    return result


# ═══════════════════════════════════════════════════════════════
# CONTRACT ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/contracts", tags=["Contracts"])
def get_contracts():
    """List all registered smart contracts."""
    return store.list_contracts()


@app.get("/api/contracts/active", tags=["Contracts"])
def get_active_contract():
    """Return the currently active smart contract."""
    c = store.get_active_contract()
    if not c:
        raise HTTPException(status_code=404, detail="No active contract")
    return c


@app.post("/api/contracts", status_code=201, tags=["Contracts"])
def create_contract(body: ContractCreate):
    """Register a new smart contract."""
    # Basic checksum validation
    try:
        Web3.to_checksum_address(body.address)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Invalid Ethereum address (must be checksummed or lowercase hex)"
        )
    return store.add_contract(
        address=body.address,
        network=body.network,
        rpc_url=body.rpc_url,
        chain_id=body.chain_id,
        name=body.name,
        abi_path=body.abi_path,
    )


@app.delete("/api/contracts/{contract_id}", tags=["Contracts"])
def delete_contract(contract_id: str):
    """Remove a contract by ID."""
    if not store.remove_contract(contract_id):
        raise HTTPException(status_code=404, detail="Contract not found")
    return {"ok": True}


@app.put("/api/contracts/{contract_id}/activate", tags=["Contracts"])
def activate_contract(contract_id: str):
    """Set a contract as the active one and update .env."""
    result = store.activate_contract(contract_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Contract not found")
    # Sync .env so FL training picks up the new contract
    store.write_env({
        "BASE_SEPOLIA_RPC_URL": result["rpc_url"],
        "CONTRACT_ADDRESS":     result["address"],
    })
    return result


# ═══════════════════════════════════════════════════════════════
# TRAINING CONFIG
# ═══════════════════════════════════════════════════════════════

@app.get("/api/config", tags=["Training"])
def get_config():
    """Return current training configuration."""
    return store.get_config()


@app.put("/api/config", tags=["Training"])
def update_config(body: TrainingConfig):
    """Persist training configuration."""
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=422, detail="No fields to update")
    return store.update_config(updates)


# ═══════════════════════════════════════════════════════════════
# TRAINING LIFECYCLE
# ═══════════════════════════════════════════════════════════════

@app.post("/api/training/start", tags=["Training"])
def start_training(body: TrainingLaunch = TrainingLaunch()):
    """Launch `flwr run .` in simulation or real-federation mode.

    Simulation  → flwr run . local-simulation   (virtual supernodes, no HW needed)
    Federation  → flwr run . remote-federation  (real SuperLink + registered devices)
    """
    global _training_proc, _training_start, _training_log

    if _training_proc is not None and _training_proc.poll() is None:
        raise HTTPException(status_code=409, detail="Training already running")

    cfg = store.get_config()

    # Merge per-launch overrides into cfg (and persist)
    overrides = {k: v for k, v in body.model_dump().items() if v is not None}
    cfg.update(overrides)
    store.update_config(cfg)

    mode = cfg.get("run_mode", "simulation")

    # Common --run-config flags (app-level hyper-parameters)
    run_cfg = " ".join([
        f"num-server-rounds={cfg['num_rounds']}",
        f"lr={cfg['lr']}",
        f"local-epochs={cfg['local_epochs']}",
        f"fraction-train={cfg['fraction_train']}",
    ])

    if mode == "simulation":
        # ── Simulation branch ──────────────────────────────────
        # Uses the local-simulation federation defined in pyproject.toml.
        # num-supernodes controls how many virtual FL clients are spawned.
        override = (
            f"federations.local-simulation.options.num-supernodes="
            f"{cfg['num_supernodes']}"
        )
        cmd = [
            sys.executable, "-m", "flwr", "run", ".", "local-simulation",
            "--run-config", run_cfg,
            "--override", override,
        ]

    else:
        # ── Real federation branch ─────────────────────────────
        # Uses the remote-federation federation defined in pyproject.toml.
        # The SuperLink address (host:port) must be configured.
        address = cfg.get("superlink_address", "").strip()
        if not address:
            raise HTTPException(
                status_code=422,
                detail="superlink_address must be set for federation mode "
                       "(e.g. '192.168.1.100:9093')"
            )
        insecure = bool(cfg.get("superlink_insecure", True))

        # Patch pyproject.toml so the remote-federation section is correct
        store.update_pyproject_superlink(address, insecure)

        override_parts = [
            f"federations.remote-federation.address={address}",
        ]
        if insecure:
            override_parts.append("federations.remote-federation.insecure=true")

        cmd = [
            sys.executable, "-m", "flwr", "run", ".", "remote-federation",
            "--run-config", run_cfg,
            "--override", " ".join(override_parts),
        ]

    # Reset log and results from any previous run
    _training_log = []
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_FILE.exists():
        RESULTS_FILE.unlink()

    _training_proc  = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    _training_start = time.time()

    return {
        "ok":     True,
        "mode":   mode,
        "pid":    _training_proc.pid,
        "cmd":    " ".join(cmd),
        "config": cfg,
    }


@app.post("/api/training/stop", tags=["Training"])
def stop_training():
    """Send SIGTERM to the running training process."""
    global _training_proc
    if _training_proc is None or _training_proc.poll() is not None:
        raise HTTPException(status_code=409, detail="No training process is running")

    pid = _training_proc.pid
    try:
        os.kill(pid, signal.SIGTERM)
        _training_proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    return {"ok": True, "pid": pid}


@app.get("/api/training/status", tags=["Training"])
def training_status():
    """Return live process status and last log lines."""
    return {
        "process": _proc_status(),
        "log":     _training_log[-40:],
    }


@app.get("/api/training/log", tags=["Training"])
def training_log(lines: int = 80):
    """Stream recent stdout lines from the training process."""
    global _training_proc, _training_log

    if _training_proc is None:
        return {"lines": []}

    # Non-blocking drain of stdout into _training_log
    if _training_proc.stdout:
        try:
            import select
            ready, _, _ = select.select([_training_proc.stdout], [], [], 0)
            while ready:
                line = _training_proc.stdout.readline()
                if not line:
                    break
                _training_log.append(line.rstrip())
                if len(_training_log) > 500:
                    _training_log = _training_log[-400:]
                ready, _, _ = select.select([_training_proc.stdout], [], [], 0)
        except Exception:
            pass

    return {"lines": _training_log[-lines:]}


# ═══════════════════════════════════════════════════════════════
# MONITORING ENDPOINTS  (replicate / extend dashboard/server.py)
# ═══════════════════════════════════════════════════════════════

@app.get("/api/state", tags=["Monitoring"])
def get_state():
    """Full FL metrics state (used by live monitoring dashboard)."""
    return _build_dashboard_state()


@app.get("/api/stream", tags=["Monitoring"])
def stream():
    """Server-Sent Events — pushes merged state every 3 seconds."""
    def _gen():
        while True:
            try:
                payload = json.dumps(_build_dashboard_state())
                yield f"data: {payload}\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            time.sleep(3)

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/blockchain", tags=["Monitoring"])
def get_blockchain(limit: int = 6):
    """Blockchain ledger endpoint."""
    bc = _blockchain_state(limit=limit)
    bc["blocks"] = bc.pop("recent_blocks", [])
    return bc


# ═══════════════════════════════════════════════════════════════
# SYSTEM
# ═══════════════════════════════════════════════════════════════

@app.get("/api/health", tags=["System"])
def health():
    return {
        "ok":       True,
        "devices":  len(store.list_devices()),
        "contracts": len(store.list_contracts()),
        "training": _proc_status(),
    }


@app.get("/api/env", tags=["System"])
def get_env():
    """Return non-secret .env keys (masks PRIVATE_KEY and JWT values)."""
    raw = store.read_env()
    safe = {}
    for k, v in raw.items():
        if any(x in k.upper() for x in ("KEY", "SECRET", "JWT", "PASSWORD")):
            safe[k] = "***"
        else:
            safe[k] = v
    return safe


@app.put("/api/env", tags=["System"])
def update_env(updates: Dict[str, str]):
    """Update .env entries (use with care — rewrites the file)."""
    store.write_env(updates)
    return {"ok": True, "updated_keys": list(updates.keys())}


# ═══════════════════════════════════════════════════════════════
# HTML ROUTES
# ═══════════════════════════════════════════════════════════════

_HERE = Path(__file__).parent


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def management_ui():
    """Serve the management console."""
    p = _HERE / "management.html"
    if p.exists():
        return HTMLResponse(p.read_text())
    return HTMLResponse("<h1>management.html not found</h1>", status_code=500)


@app.get("/monitor", response_class=HTMLResponse, include_in_schema=False)
def monitor_ui():
    """Serve the live metrics dashboard."""
    p = _HERE.parent / "dashboard" / "fl_dashboard.html"
    if p.exists():
        return HTMLResponse(p.read_text())
    return HTMLResponse("<h1>fl_dashboard.html not found</h1>", status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
