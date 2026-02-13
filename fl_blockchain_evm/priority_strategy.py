"""Priority-Based Medical Federated Averaging Strategy.

Implements unweighted aggregation so that all 10 IoT edge devices
contribute equally to the global model, preventing majority-class
clients from drowning out arrhythmia updates.
"""

from typing import List, Tuple, Union, Optional, Dict
from flwr.serverapp.strategy import FedAvg
from flwr.app import RecordDict


class MedicalFedAvg(FedAvg):
    """FedAvg with Unweighted Aggregation for medical FL.

    Forces all clients to contribute equally to the global model,
    preventing majority-class clients from drowning out arrhythmia updates.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_emergency_count = 0
        self.round_emergency_count = {}
        self._current_round = 0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[RecordDict, int]],
        failures: List[Union[Tuple[RecordDict, int], BaseException]],
    ) -> Tuple[Optional[RecordDict], Dict[str, Union[bool, bytes, float, int, str]]]:

        if not results:
            return None, {}

        # Unweighted aggregation: assign weight=1 to every client
        unweighted_results = [(record, 1) for record, _ in results]
        return super().aggregate_fit(server_round, unweighted_results, failures)
