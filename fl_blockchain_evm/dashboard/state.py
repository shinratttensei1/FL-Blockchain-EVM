"""Live state tracking for FL dashboard and real-time monitoring.

This module maintains the current state of the federated learning training
process, allowing external components (like fl_dashboard_server.py) to query
the live status, current round, metrics, etc.
"""

import threading
from typing import Dict, List, Optional, Any

# Thread-safe state container
_lock = threading.Lock()
_state: Dict[str, Any] = {
    "initialized": False,
    "current_round": 0,
    "total_rounds": 0,
    "contract_address": None,
    "status": "idle",  # idle, training, evaluating, complete
    "last_update": None,
    "metrics": {},
    "chain_length": 0,
    "chain_valid": None,
    "client_data": [],
    "votes": [],
    "ipfs_enabled": False,
    "ipfs_cids": {},
    "ipfs_pin_count": 0,
}


def init(num_rounds: int, contract_address: str):
    """Initialize the live state at the start of training."""
    with _lock:
        _state["initialized"] = True
        _state["total_rounds"] = num_rounds
        _state["contract_address"] = contract_address
        _state["current_round"] = 0
        _state["status"] = "initialized"
        _state["metrics"] = {}
        _state["client_data"] = []
        _state["votes"] = []


def round_started(round_num: int):
    """Mark the start of a new training round."""
    with _lock:
        _state["current_round"] = round_num
        _state["status"] = "training"
        _state["client_data"] = []
        _state["votes"] = []


def evaluating(round_num: int):
    """Mark that the server is evaluating the global model."""
    with _lock:
        _state["current_round"] = round_num
        _state["status"] = "evaluating"


def clients_trained(
    client_data: List[Dict],
    loss_mean: float,
    loss_std: float,
    threshold: float,
    votes: List[Dict],
):
    """Update state after clients have completed training."""
    with _lock:
        _state["client_data"] = client_data
        _state["loss_mean"] = loss_mean
        _state["loss_std"] = loss_std
        _state["threshold"] = threshold
        _state["votes"] = votes


def round_complete(
    round_num: int,
    metrics: Dict[str, float],
    chain_length: int,
    chain_valid: bool,
):
    """Mark a round as complete with its metrics."""
    with _lock:
        _state["current_round"] = round_num
        _state["status"] = "round_complete"
        _state["metrics"] = metrics
        _state["chain_length"] = chain_length
        _state["chain_valid"] = chain_valid


def done(final_chain_length: int):
    """Mark training as complete."""
    with _lock:
        _state["status"] = "complete"
        _state["chain_length"] = final_chain_length


def ipfs_pinned(round_num: int, cids: Dict[str, str]):
    """Record IPFS CIDs for a completed round."""
    with _lock:
        _state["ipfs_enabled"] = True
        _state["ipfs_cids"][round_num] = cids
        _state["ipfs_pin_count"] = sum(
            len(v) for v in _state["ipfs_cids"].values()
        )


def get_state() -> Dict[str, Any]:
    """Get a copy of the current state (thread-safe)."""
    with _lock:
        return dict(_state)


def reset():
    """Reset the state (useful for testing)."""
    with _lock:
        _state.clear()
        _state.update({
            "initialized": False,
            "current_round": 0,
            "total_rounds": 0,
            "contract_address": None,
            "status": "idle",
            "last_update": None,
            "metrics": {},
            "chain_length": 0,
            "chain_valid": None,
            "client_data": [],
            "votes": [],
        })
