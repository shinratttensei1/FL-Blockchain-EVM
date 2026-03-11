"""Shared constants for the FL-Blockchain-EVM project.

Diagnostic superclass taxonomy, SCP code mapping, and class metadata
used across the model, data loading, training, and evaluation modules.
"""

from typing import Dict, List

# ── Diagnostic superclass SCP codes (PTB-XL) ──────────────────

DIAG_NORM = ["NORM"]
DIAG_MI = ["IMI", "ASMI", "ILMI", "AMI", "ALMI", "INJAS", "LMI",
           "INJAL", "IPLMI", "IPMI", "INJIN", "INJLA", "PMI", "INJIL"]
DIAG_STTC = ["NDT", "NST_", "DIG", "LNGQT", "ISC_", "ISCAL",
             "ISCIN", "ISCIL", "ISCAS", "ISCLA", "ANEUR", "EL", "ISCAN"]
DIAG_CD = ["LAFB", "IRBBB", "1AVB", "IVCD", "CRBBB",
           "CLBBB", "LPFB", "WPW", "ILBBB", "3AVB", "2AVB"]
DIAG_HYP = ["LVH", "LAO/LAE", "RVH", "RAO/RAE", "SEHYP"]

SC_NAMES: List[str] = ["NORM", "MI", "STTC", "CD", "HYP"]
NUM_CLASSES = 5

# SCP code → superclass index mapping
_SCP_TO_SC: Dict[str, int] = {}
for _i, _codes in enumerate([DIAG_NORM, DIAG_MI, DIAG_STTC, DIAG_CD, DIAG_HYP]):
    for _c in _codes:
        _SCP_TO_SC[_c] = _i
