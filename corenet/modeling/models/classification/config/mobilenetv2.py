#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Dict


def get_configuration(opts) -> Dict:

    mobilenetv2_config = {
        "layer1": {
            "expansion_ratio": 1,
            "out_channels": 16,
            "num_blocks": 1,
            "stride": 1,
        },
        "layer2": {
            "expansion_ratio": 6,
            "out_channels": 24,
            "num_blocks": 2,
            "stride": 2,
        },
        "layer3": {
            "expansion_ratio": 6,
            "out_channels": 32,
            "num_blocks": 3,
            "stride": 2,
        },
        "layer4": {
            "expansion_ratio": 6,
            "out_channels": 64,
            "num_blocks": 4,
            "stride": 2,
        },
        "layer4_a": {
            "expansion_ratio": 6,
            "out_channels": 96,
            "num_blocks": 3,
            "stride": 1,
        },
        "layer5": {
            "expansion_ratio": 6,
            "out_channels": 160,
            "num_blocks": 3,
            "stride": 2,
        },
        "layer5_a": {
            "expansion_ratio": 6,
            "out_channels": 320,
            "num_blocks": 1,
            "stride": 1,
        },
    }
    return mobilenetv2_config
