"""UCI-HAR Human Activity Recognition — 6 activity classes.

Backward-compatible re-export layer. The original monolithic module was
split into focused sub-modules:

    constants  – activity class names, NUM_CLASSES
    model      – FocalLoss, _SEResBlock, Net
    data       – load_data, balancing (ROS+RUS), augmentation
    training   – train(), test(), evaluation helpers

All public symbols remain importable from this module.
"""

from fl_blockchain_evm.core.constants import (  # noqa: F401
    ACTIVITY_NAMES, SC_NAMES, NUM_CLASSES,
)
from fl_blockchain_evm.core.model import FocalLoss, Net, _SEResBlock  # noqa: F401
from fl_blockchain_evm.core.data import (  # noqa: F401
    load_data, _balance_ros_rus, _augment,
)
from fl_blockchain_evm.core.training import (  # noqa: F401
    train, test, _class_weights, _empty,
)
