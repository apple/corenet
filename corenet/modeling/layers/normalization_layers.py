#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import math

from torch import nn

from corenet.modeling.layers.normalization import (
    NORM_LAYER_CLS,
    build_normalization_layer,
)
from corenet.utils import logger

norm_layers_tuple = tuple(NORM_LAYER_CLS)


get_normalization_layer = build_normalization_layer
