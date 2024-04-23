#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Dict

from corenet.utils import logger


def get_configuration(opts) -> Dict:
    mv3_mode = getattr(opts, "model.classification.mobilenetv3.mode", "large")

    if mv3_mode is None:
        logger.error(
            "MobileNetv3 mode can't be none. Please specify --model.classification.mobilenetv3.mode"
        )

    mv3_mode = mv3_mode.lower()
    mv3_config = dict()
    if mv3_mode == "small":
        # kernel_size, expansion_factor, in_channels, use_se, use_hs, stride
        mv3_config["layer_1"] = [[3, 1, 16, True, False, 2]]

        mv3_config["layer_2"] = [[3, 4.5, 24, False, False, 2]]

        mv3_config["layer_3"] = [[3, 3.67, 24, False, False, 1]]

        mv3_config["layer_4"] = [
            [5, 4, 40, True, True, 2],
            [5, 6, 40, True, True, 1],
            [5, 6, 40, True, True, 1],
            [5, 3, 48, True, True, 1],
            [5, 3, 48, True, True, 1],
        ]

        mv3_config["layer_5"] = [
            [5, 6, 96, True, True, 2],
            [5, 6, 96, True, True, 1],
            [5, 6, 96, True, True, 1],
        ]
        mv3_config["last_channels"] = 1024
    elif mv3_mode == "large":
        mv3_config["layer_1"] = [[3, 1, 16, False, False, 1]]
        mv3_config["layer_2"] = [
            [3, 4, 24, False, False, 2],
            [3, 3, 24, False, False, 1],
        ]

        mv3_config["layer_3"] = [
            [5, 3, 40, True, False, 2],
            [5, 3, 40, True, False, 1],
            [5, 3, 40, True, False, 1],
        ]

        mv3_config["layer_4"] = [
            [3, 6, 80, False, True, 2],
            [3, 2.5, 80, False, True, 1],
            [3, 2.3, 80, False, True, 1],
            [3, 2.3, 80, False, True, 1],
            [3, 6, 112, True, True, 1],
            [3, 6, 112, True, True, 1],
        ]

        mv3_config["layer_5"] = [
            [5, 6, 160, True, True, 2],
            [5, 6, 160, True, True, 1],
            [5, 6, 160, True, True, 1],
        ]
        mv3_config["last_channels"] = 1280
    else:
        logger.error(
            "Current supported modes for MobileNetv3 are small and large. Got: {}".format(
                mv3_mode
            )
        )
    return mv3_config
