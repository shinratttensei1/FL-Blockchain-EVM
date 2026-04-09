"""Persistent file-backed store for FL management state.

Stores devices, smart contracts, and training configuration in a single
JSON file (management_store.json at the project root). All operations
are thread-safe via a module-level lock.

Schema
------
{
  "devices": [
    {
      "id":          "uuid4",
      "name":        "Device 0",
      "host":        "192.168.1.10",
      "port":        8765,
      "type":        "sensor_node",   # sensor_node | raspberry_pi | cloud
      "description": "",
      "enabled":     true,
      "created_at":  "ISO-8601"
    }, ...
  ],
  "contracts": [
    {
      "id":         "uuid4",
      "name":       "MHEALTH v1",
      "address":    "0x...",
      "network":    "Base Sepolia",
      "rpc_url":    "https://sepolia.base.org",
      "chain_id":   84532,
      "abi_path":   "contracts/FLBlockchain_abi.json",
      "active":     true,
      "created_at": "ISO-8601"
    }, ...
  ],
  "config": {
    "num_rounds":      10,
    "lr":              0.002,
    "local_epochs":    5,
    "fraction_train":  1.0,
    "num_supernodes":  10,
    "beta":            1.0,
    "batch_size":      64
  }
}
"""

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_STORE_PATH = Path("management_store.json")
_lock = threading.Lock()

_DEFAULT: Dict[str, Any] = {
    "devices": [],
    "contracts": [],
    "config": {
        # ── Common ───────────────────────────────────────────────
        "num_rounds":         10,
        "lr":                 0.002,
        "local_epochs":       5,
        "fraction_train":     1.0,
        "beta":               1.0,
        "batch_size":         64,
        # ── Run mode ─────────────────────────────────────────────
        # "simulation"  → flwr local-simulation (virtual supernodes, no real HW)
        # "federation"  → flwr remote-federation (real SuperLink + edge devices)
        "run_mode":           "simulation",
        # ── Simulation-only ──────────────────────────────────────
        "num_supernodes":     10,
        # ── Federation-only ──────────────────────────────────────
        "superlink_address":  "",   # e.g. "192.168.1.100:9093"
        "superlink_insecure": True, # set False to require TLS
    },
}


# ── Low-level I/O ─────────────────────────────────────────────

def _load() -> Dict[str, Any]:
    if not _STORE_PATH.exists():
        return json.loads(json.dumps(_DEFAULT))
    with open(_STORE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    # Back-fill any missing keys from defaults
    for k, v in _DEFAULT.items():
        if k not in data:
            data[k] = json.loads(json.dumps(v))
    for k, v in _DEFAULT["config"].items():
        data["config"].setdefault(k, v)
    return data


def _save(data: Dict[str, Any]) -> None:
    _STORE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _now() -> str:
    return datetime.utcnow().isoformat() + "Z"


# ── Devices ───────────────────────────────────────────────────

def list_devices() -> List[Dict]:
    with _lock:
        return _load()["devices"]


def add_device(
    name: str,
    host: str,
    port: int,
    device_type: str = "sensor_node",
    description: str = "",
) -> Dict:
    with _lock:
        data = _load()
        device = {
            "id":          str(uuid.uuid4()),
            "name":        name,
            "host":        host,
            "port":        port,
            "type":        device_type,
            "description": description,
            "enabled":     True,
            "created_at":  _now(),
        }
        data["devices"].append(device)
        _save(data)
        return device


def remove_device(device_id: str) -> bool:
    with _lock:
        data = _load()
        before = len(data["devices"])
        data["devices"] = [d for d in data["devices"] if d["id"] != device_id]
        if len(data["devices"]) == before:
            return False
        _save(data)
        return True


def toggle_device(device_id: str) -> Optional[Dict]:
    with _lock:
        data = _load()
        for d in data["devices"]:
            if d["id"] == device_id:
                d["enabled"] = not d["enabled"]
                _save(data)
                return d
        return None


def update_device(device_id: str, updates: Dict) -> Optional[Dict]:
    """Patch allowed fields on a device."""
    allowed = {"name", "host", "port", "type", "description", "enabled"}
    with _lock:
        data = _load()
        for d in data["devices"]:
            if d["id"] == device_id:
                for k, v in updates.items():
                    if k in allowed:
                        d[k] = v
                _save(data)
                return d
        return None


# ── Contracts ─────────────────────────────────────────────────

def list_contracts() -> List[Dict]:
    with _lock:
        return _load()["contracts"]


def add_contract(
    address: str,
    network: str,
    rpc_url: str,
    chain_id: int,
    name: str = "",
    abi_path: str = "contracts/FLBlockchain_abi.json",
) -> Dict:
    with _lock:
        data = _load()
        # Deactivate all others if this is the first
        is_first = len(data["contracts"]) == 0
        contract = {
            "id":         str(uuid.uuid4()),
            "name":       name or f"{network} contract",
            "address":    address,
            "network":    network,
            "rpc_url":    rpc_url,
            "chain_id":   chain_id,
            "abi_path":   abi_path,
            "active":     is_first,
            "created_at": _now(),
        }
        data["contracts"].append(contract)
        _save(data)
        return contract


def remove_contract(contract_id: str) -> bool:
    with _lock:
        data = _load()
        before = len(data["contracts"])
        removed = next((c for c in data["contracts"] if c["id"] == contract_id), None)
        data["contracts"] = [c for c in data["contracts"] if c["id"] != contract_id]
        if len(data["contracts"]) == before:
            return False
        # If removed contract was active and others remain, activate the first
        if removed and removed.get("active") and data["contracts"]:
            data["contracts"][0]["active"] = True
        _save(data)
        return True


def activate_contract(contract_id: str) -> Optional[Dict]:
    with _lock:
        data = _load()
        target = None
        for c in data["contracts"]:
            c["active"] = (c["id"] == contract_id)
            if c["active"]:
                target = c
        if target is None:
            return None
        _save(data)
        return target


def get_active_contract() -> Optional[Dict]:
    with _lock:
        for c in _load()["contracts"]:
            if c.get("active"):
                return c
        return None


# ── Training config ───────────────────────────────────────────

def get_config() -> Dict:
    with _lock:
        return _load()["config"]


def update_config(updates: Dict) -> Dict:
    allowed = {
        "num_rounds", "lr", "local_epochs", "fraction_train",
        "num_supernodes", "beta", "batch_size",
        "run_mode", "superlink_address", "superlink_insecure",
    }
    with _lock:
        data = _load()
        for k, v in updates.items():
            if k in allowed:
                data["config"][k] = v
        _save(data)
        return data["config"]


# ── Env file helpers ──────────────────────────────────────────

def _env_path() -> Path:
    return Path(".env")


def read_env() -> Dict[str, str]:
    """Parse .env into a dict (values without quotes)."""
    env: Dict[str, str] = {}
    p = _env_path()
    if not p.exists():
        return env
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def write_env(updates: Dict[str, str]) -> None:
    """Merge updates into .env, preserving existing lines."""
    p = _env_path()
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []

    existing_keys = set()
    new_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            new_lines.append(line)
            continue
        k = stripped.split("=", 1)[0].strip()
        if k in updates:
            new_lines.append(f'{k}={updates[k]}')
            existing_keys.add(k)
        else:
            new_lines.append(line)

    for k, v in updates.items():
        if k not in existing_keys:
            new_lines.append(f'{k}={v}')

    p.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


# ── pyproject.toml superlink patcher ─────────────────────────

def update_pyproject_superlink(address: str, insecure: bool) -> None:
    """Patch the remote-federation address in pyproject.toml in-place.

    Finds the [tool.flwr.federations.remote-federation] section and
    rewrites the address and insecure lines without touching anything else.
    """
    path = Path("pyproject.toml")
    if not path.exists():
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    new_lines: List[str] = []
    in_remote = False

    for line in lines:
        stripped = line.strip()

        # Detect section entry / exit
        if stripped == "[tool.flwr.federations.remote-federation]":
            in_remote = True
        elif stripped.startswith("[") and in_remote:
            in_remote = False

        if in_remote and stripped.startswith("address"):
            new_lines.append(f'address = "{address}"')
        elif in_remote and stripped.lstrip("# ").startswith("insecure"):
            # Uncomment and set, or comment out, depending on flag
            new_lines.append("insecure = true" if insecure else "# insecure = true  # Remove comment to enable TLS bypass")
        else:
            new_lines.append(line)

    path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
