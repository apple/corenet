#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
from dataclasses import dataclass
from functools import partial
from typing import Dict

from corenet.utils import logger
from corenet.utils.math_utils import make_divisible


@dataclass
class CompoundScalingConfig:
    """This class stores the compound scaling configuration"""

    width_mult: float
    depth_mult: float
    train_resolution: int


@dataclass
class EfficientNetBlockConfig:
    """This class stores the config for each block in EfficientNet i.e. MBConv layers
    in Table 1 of `EfficientNet paper <https://arxiv.org/abs/1905.11946>`_
    Notably, this class takes width_mult and depth_mult as input too and adjusts
    layers' depth and width, as is required in different modes of EfficientNet.
    """

    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float,
        depth_mult: float,
    ):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_channels = int(make_divisible(in_channels * width_mult, 8))
        self.out_channels = int(make_divisible(out_channels * width_mult, 8))
        self.num_layers = int(math.ceil(num_layers * depth_mult))


def get_configuration(opts) -> Dict:
    network_mode = getattr(opts, "model.classification.efficientnet.mode")

    if network_mode is None:
        logger.error(
            "EfficientNet mode can't be none. Please specify --model.classification.efficientnet.mode"
        )

    network_mode = network_mode.lower()
    network_config = dict()

    # EfficientNet scales depth, width and resolution.
    # We will make use of resolution in the yaml configuration file, but leave it here for the sake of completeness

    compound_scaling_cfg = {
        "b0": CompoundScalingConfig(1.0, 1.0, 224),
        "b1": CompoundScalingConfig(1.0, 1.1, 240),
        "b2": CompoundScalingConfig(1.1, 1.2, 260),
        "b3": CompoundScalingConfig(1.2, 1.4, 300),
        "b4": CompoundScalingConfig(1.4, 1.8, 380),
        "b5": CompoundScalingConfig(1.6, 2.2, 456),
        "b6": CompoundScalingConfig(1.8, 2.6, 528),
        "b7": CompoundScalingConfig(2.0, 3.1, 600),
        "b8": CompoundScalingConfig(2.2, 3.6, 672),
    }

    if network_mode in compound_scaling_cfg:
        compound_scaling_cfg_mode_i = compound_scaling_cfg[network_mode]
        width_mult = compound_scaling_cfg_mode_i.width_mult
        depth_mult = compound_scaling_cfg_mode_i.depth_mult

        # pre-feed depth and width multipliers as they are always used and same across layers.
        block_builder = partial(
            EfficientNetBlockConfig, width_mult=width_mult, depth_mult=depth_mult
        )

        # Build the configuration at each spatial level.
        # The format of configuraiton is: (expand_ratio, kernel, stride, in_channels, out_channels, num_layers)

        # Configuration at output stride of 2
        network_config["layer_1"] = [block_builder(1, 3, 1, 32, 16, 1)]

        # Configuration at output stride of 4
        network_config["layer_2"] = [
            block_builder(6, 3, 2, 16, 24, 2),
        ]

        # Configuration at output stride of 8
        network_config["layer_3"] = [
            block_builder(6, 5, 2, 24, 40, 2),
        ]

        # Configuration at output stride of 16
        network_config["layer_4"] = [
            block_builder(6, 3, 2, 40, 80, 3),
            block_builder(6, 5, 1, 80, 112, 3),
        ]
        # Configuration at output stride of 32
        network_config["layer_5"] = [
            block_builder(6, 5, 2, 112, 192, 4),
            block_builder(6, 3, 1, 192, 320, 1),
        ]
        network_config["last_channels"] = 4 * network_config["layer_5"][-1].out_channels
    else:
        logger.error(
            "Current supported modes for EfficientNet are b[0-7]. Got: {}".format(
                network_mode
            )
        )

    # Count the total number of layers throughout all blocks.
    # This will be used for stochastic depth (if enabled)
    total_layers = 0
    for layer_name in ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5"]:
        for block_config in network_config[layer_name]:
            total_layers += block_config.num_layers
    network_config["total_layers"] = total_layers
    return network_config
