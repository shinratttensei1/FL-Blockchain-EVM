from typing import List, Tuple, Union, Optional, Dict
from flwr.serverapp.strategy import FedAvg
from flwr.app import RecordDict


class MedicalFedAvg(FedAvg):
    """Standard FedAvg weighted by each client's number of training examples."""

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        return super().aggregate_fit(server_round, results, failures)
