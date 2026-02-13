"""BLOCK-CARE: Unweighted FedAvg — all devices contribute equally."""

from typing import List, Tuple, Union, Optional, Dict
from flwr.serverapp.strategy import FedAvg
from flwr.app import RecordDict


class MedicalFedAvg(FedAvg):
    """Forces weight=1 for every client so rare-pathology devices
    (e.g. those with more HYP/MI) get equal influence."""

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        return super().aggregate_fit(
            server_round, [(rec, 1) for rec, _ in results], failures)
