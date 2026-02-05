"""Priority-Based Medical Federated Averaging Strategy.

This strategy implements priority-weighted aggregation for medical IoT FL networks,
where emergency cases receive higher weight (e=2) to address the Service-Utility Gap.
Based on FL-BlockNet framework (Ben Othman Soufiane, 2025) achieving 45.443ms latency.
"""

import json
import hashlib
from datetime import datetime
from typing import List, Optional, Tuple, Union, Dict
from logging import INFO

from flwr.app import ConfigRecord, MetricRecord, RecordDict
from flwr.common.logger import log
from flwr.serverapp.strategy import FedAvg

# ANSI Color codes
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RESET = '\033[0m'


class PriorityMedicalFedAvg(FedAvg):
    """FedAvg with Priority-Weighted Aggregation for medical emergencies.

    Emergency updates receive a priority multiplier (default e=2.0) to ensure
    critical pathologies are not averaged out by routine data. This creates
    the "Emergency Acuity" mechanism for clinical accountability.

    Parameters
    ----------
    emergency_multiplier : float, optional
        Priority weight for emergency updates (default: 2.0)
    alert_log_path : str, optional
        Path to log emergency alerts (default: "emergency_alerts.json")
    **kwargs
        Additional arguments passed to FedAvg
    """

    def __init__(
        self,
        emergency_multiplier: float = 2.0,
        alert_log_path: str = "outputs/emergency_alerts.json",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.emergency_multiplier = emergency_multiplier
        self.alert_log_path = alert_log_path
        self.total_emergency_count = 0
        self.round_emergency_count = {}
        self._current_round = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[RecordDict, int]],
        failures: List[Union[Tuple[RecordDict, int], BaseException]],
    ) -> Tuple[Optional[RecordDict], Dict[str, Union[bool, bytes, float, int, str]]]:
        """Aggregate model updates with priority weighting for emergencies."""
        self._current_round = server_round
        if not results:
            return None, {}

        emergency_clients = []
        normal_clients = []

        weighted_results = []

        print(
            f"\n{YELLOW}[DEBUG] Round {server_round}: Received {len(results)} client results{RESET}")

        for i, (record_dict, num_examples) in enumerate(results):
            print(
                f"{YELLOW}[DEBUG] Result {i}: type={type(record_dict)}, keys={list(record_dict.keys()) if hasattr(record_dict, 'keys') else 'N/A'}{RESET}")

            metrics = record_dict.get("metrics", {})
            if isinstance(metrics, MetricRecord):
                print(
                    f"{YELLOW}[DEBUG] Converting MetricRecord to dict{RESET}")
                metrics = dict(metrics)

            print(
                f"{YELLOW}[DEBUG] Metrics type: {type(metrics)}, content: {metrics}{RESET}")

            is_emergency = bool(metrics.get("is_emergency", 0))
            client_id = metrics.get("client_id", "unknown")

            print(
                f"SERVER: Client {client_id} - is_emergency in metrics: {metrics.get('is_emergency')}, bool: {is_emergency}")

            if is_emergency:
                effective_weight = num_examples * self.emergency_multiplier
                emergency_clients.append({
                    "client_id": client_id,
                    "num_examples": num_examples,
                    "effective_weight": effective_weight,
                    "pathology_count": metrics.get("pathology_count", 0),
                })
                self.total_emergency_count += 1
                log(
                    INFO,
                    f"{RED}[EMERGENCY ALERT] Round {server_round}, Client {client_id}: "
                    f"{metrics.get('pathology_count', 0)} pathologies detected, "
                    f"weight boosted {num_examples} → {effective_weight}{RESET}"
                )
            else:
                effective_weight = num_examples
                normal_clients.append({
                    "client_id": client_id,
                    "num_examples": num_examples,
                    "effective_weight": effective_weight
                })

            weighted_results.append((record_dict, effective_weight))

        self.round_emergency_count[server_round] = len(emergency_clients)
        if emergency_clients:
            self._log_emergency_alert(
                server_round, emergency_clients, normal_clients)

        aggregated_record, aggregated_metrics = super().aggregate_fit(
            server_round, weighted_results, failures
        )

        aggregated_metrics["emergency_count"] = len(emergency_clients)
        aggregated_metrics["normal_count"] = len(normal_clients)
        aggregated_metrics["total_alerts"] = self.total_emergency_count
        aggregated_metrics["emergency_ratio"] = (
            len(emergency_clients) /
            (len(emergency_clients) + len(normal_clients))
            if (emergency_clients or normal_clients) else 0.0
        )

        return aggregated_record, aggregated_metrics

    def _log_emergency_alert(
        self,
        server_round: int,
        emergency_clients: List[Dict],
        normal_clients: List[Dict]
    ) -> None:
        """Log emergency alert to file for immutable audit trail.

        This creates the "Non-Repudiation" record that would be stored
        on-chain in production via EVM smart contract.

        Parameters
        ----------
        server_round : int
            Current FL round
        emergency_clients : List[Dict]
            List of emergency client metadata
        normal_clients : List[Dict]
            List of normal client metadata
        """
        alert_record = {
            "timestamp": datetime.now().isoformat(),
            "server_round": server_round,
            "alert_type": "CRITICAL_PATHOLOGY_DETECTED",
            "emergency_clients": emergency_clients,
            "normal_clients": normal_clients,
            "total_emergency": len(emergency_clients),
            "total_normal": len(normal_clients),
            "emergency_multiplier": self.emergency_multiplier,
            "blockchain_ready": {
                "contract_function": "logClinicalAlert",
                "hospital_ids": [c["client_id"] for c in emergency_clients],
                "total_pathologies": sum(c.get("pathology_count", 0) for c in emergency_clients),
            },
            "transaction_hash": hashlib.sha256(
                f"{server_round}_{datetime.now().isoformat()}_{len(emergency_clients)}".encode()
            ).hexdigest()
        }

        try:
            with open(self.alert_log_path, "a") as f:
                json.dump(alert_record, f)
                f.write("\n")
        except Exception as e:
            log(INFO, f"Warning: Could not write alert log: {e}")
