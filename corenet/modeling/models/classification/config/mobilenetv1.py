#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
from typing import Dict

from corenet.utils.math_utils import make_divisible


def get_configuration(opts) -> Dict:
    width_mult = getattr(opts, "model.classification.mobilenetv1.width_multiplier", 1.0)

    def scale_channels(in_channels):
        return make_divisible(int(math.ceil(in_channels * width_mult)), 16)

    config = {
        "conv1_out": scale_channels(32),
        "layer1": {"out_channels": scale_channels(64), "stride": 1, "repeat": 1},
        "layer2": {
            "out_channels": scale_channels(128),
            "stride": 2,
            "repeat": 1,
        },
        "layer3": {
            "out_channels": scale_channels(256),
            "stride": 2,
            "repeat": 1,
        },
        "layer4": {
            "out_channels": scale_channels(512),
            "stride": 2,
            "repeat": 5,
        },
        "layer5": {
            "out_channels": scale_channels(1024),
            "stride": 2,
            "repeat": 1,
        },
    }
    return config
