"""PTB-XL 12-Lead ECG → 5 Superclasses (NORM/MI/STTC/CD/HYP).

Backward-compatible re-export layer.  The original monolithic module was
split into focused sub-modules:

    constants  – diagnostic codes, class names, SCP mapping
    model      – FocalLoss, _SEResBlock, Net
    data       – load_data, balancing (ROS+RUS), augmentation
    training   – train(), test(), evaluation helpers

All public symbols remain importable from this module.

Paper reference: "Application of FL Techniques for Arrhythmia Classification
Using 12-Lead ECG Signals", Jimenez et al., arXiv:2208.10993v3, Jan 2024.
"""

from fl_blockchain_evm.core.constants import (  # noqa: F401
    DIAG_NORM, DIAG_MI, DIAG_STTC, DIAG_CD, DIAG_HYP,
    SC_NAMES, NUM_CLASSES, _SCP_TO_SC,
)
from fl_blockchain_evm.core.model import FocalLoss, Net, _SEResBlock  # noqa: F401
from fl_blockchain_evm.core.data import (  # noqa: F401
    load_data, _balance_ros_rus, _augment,
)
from fl_blockchain_evm.core.training import (  # noqa: F401
    train, test, _class_weights, _empty,
)
