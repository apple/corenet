#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import numpy as np

from corenet.utils import logger
from corenet.utils.math_utils import make_divisible

supported_modes = [
    "x_200mf",
    "x_400mf",
    "x_600mf",
    "x_800mf",
    "x_1.6gf",
    "x_3.2gf",
    "x_4.0gf",
    "x_6.4gf",
    "x_8.0gf",
    "x_12gf",
    "x_16gf",
    "x_32gf",
    "y_200mf",
    "y_400mf",
    "y_800mf",
    "y_600mf",
    "y_1.6gf",
    "y_3.2gf",
    "y_4.0gf",
    "y_6.4gf",
    "y_8.0gf",
    "y_12gf",
    "y_16gf",
    "y_32gf",
]


@dataclass
class BlockParamsConfig:
    """
    This class stores the quantized linear block params. It is adapted from torchvision.models.regnet:
        https://github.com/pytorch/vision/blob/c06d52b1c5f6aee36802661c3ebc6347b97cc59e/torchvision/models/regnet.py#L203

    Args:
        depth: The total number of XBlocks in the network
        w_0: Initial width
        w_a: Width slope
        w_m: Width slope in the log space
        groups: The number of groups to use in the XBlock. Referred to
        se_ratio: The squeeze-excitation ratio. The number of channels in the SE module will be the
            input channels scaled by this ratio.
        bottleneck_multiplier: The number of output channels in the intermediate conv layers in bottleneck/Xblock
            block will be scaled by this value.
        quant: Block widths will be divisible by this value
        stride: The stride of the 3x3 conv of the XBlocks
    """

    def __init__(
        self,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        groups: int,
        se_ratio: float = 0.0,
        bottleneck_multiplier: float = 1.0,
        quant: int = 8,
        stride: int = 2,
    ) -> None:
        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError(
                f"Invalid RegNet settings. Need"
                f" w_a >= 0, given w_a={w_a};"
                f" w_m > 1, given w_m={w_m};"
                f" w_0 > 0 and w_0 % 8 == 0, given w_0={w_0}."
            )

        # Continuous widths for each block. Each stage has a unique block width
        block_widths_cont = np.arange(depth) * w_a + w_0  # u_j in eq. (2) of paper
        block_capacity = np.round(
            np.log(block_widths_cont / w_0) / np.log(w_m)
        )  # s_j in eq. (3) of paper

        # Quantized block widths
        block_widths_quant = (
            (np.round(np.divide(w_0 * np.power(w_m, block_capacity), quant)) * quant)
            .astype(int)
            .tolist()
        )
        num_stages = len(set(block_widths_quant))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths_quant + [0],
            [0] + block_widths_quant,
            block_widths_quant + [0],
            [0] + block_widths_quant,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths_quant, splits[:-1]) if t]
        stage_depths = (
            np.diff([d for d, t in enumerate(splits) if t]).astype(int).tolist()
        )

        strides = [stride] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        stage_groups = [groups] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, stage_groups = self._make_widths_compatible(
            stage_widths, stage_groups, bottleneck_multipliers
        )

        self.depths = stage_depths
        self.widths = stage_widths
        self.stage_groups = stage_groups
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    def _make_widths_compatible(
        self,
        stage_widths: List[int],
        stage_groups: List[int],
        bottleneck_multipliers: List[float],
    ) -> Tuple[List[int], List[int]]:
        """
        Scales widths by bottleneck multipliers and adjusts them to be compatible with
        the specified groups.
        """
        # Scale widths according to bottleneck multipliers
        widths = [
            int(width * multiplier)
            for width, multiplier in zip(stage_widths, bottleneck_multipliers)
        ]
        group_widths_min = [
            min(groups, width) for groups, width in zip(stage_groups, widths)
        ]

        # Ensure the widths are divisible by groups
        bottleneck_widths = [
            make_divisible(width, groups)
            for width, groups in zip(widths, group_widths_min)
        ]

        # Undo scaling
        stage_widths = [
            int(width / multiplier)
            for width, multiplier in zip(bottleneck_widths, bottleneck_multipliers)
        ]

        return stage_widths, group_widths_min

    def extra_repr(self) -> str:
        extra_repr_str = ""
        extra_repr_str += f"\n\tdepths={self.depths}"
        extra_repr_str += f"\n\twidths={self.widths}"
        extra_repr_str += f"\n\tstrides={self.strides}"
        extra_repr_str += f"\n\tstage_groups={self.stage_groups}"
        extra_repr_str += f"\n\tbottleneck_multipliers={self.bottleneck_multipliers}"
        extra_repr_str += f"\n\tse_ratio={self.se_ratio}"

        return extra_repr_str

    def __repr__(self) -> str:
        return "{}({}\n)".format(self.__class__.__name__, self.extra_repr())


def get_configuration(
    opts: argparse.Namespace,
) -> Dict[str, Dict[str, Union[int, float]]]:
    """Gets the RegNet model configuration for the specified RegNet mode.

    Args:
        opts: command-line arguments

    Returns:
        * A dictionary containing the configuration for each layer. Each key is of the form
            layer<i> and the corresponding value is another dictionary with the following keys:
                depth: The depth of the stage at layer<i>
                width: The width of the blocks at this stage
                groups: The convolution groups of each block at this stage
                stride: The stride of the convolutions in each block at this stage
                bottleneck_multiplier: The multiplier for the bottleneck conv in each of this stage's blocks
                se_ratio: The squeeze-excitation ratio for each block in this stage
    """
    network_mode = getattr(opts, "model.classification.regnet.mode")

    if network_mode is None:
        logger.error(
            "RegNet mode can't be none. Please specify --model.classification.regnet.mode"
        )

    network_config = dict()

    block_params_config = {
        "x_200mf": BlockParamsConfig(13, 24, 36.44, 2.49, 8),
        "x_400mf": BlockParamsConfig(22, 24, 24.48, 2.54, 16),
        "x_600mf": BlockParamsConfig(16, 48, 36.97, 2.24, 24),
        "x_800mf": BlockParamsConfig(16, 56, 35.73, 2.28, 16),
        "x_1.6gf": BlockParamsConfig(18, 80, 34.01, 2.25, 24),
        "x_3.2gf": BlockParamsConfig(25, 88, 26.31, 2.25, 48),
        "x_4.0gf": BlockParamsConfig(23, 96, 38.65, 2.43, 40),
        "x_6.4gf": BlockParamsConfig(17, 184, 60.83, 2.07, 56),
        "x_8.0gf": BlockParamsConfig(23, 80, 49.56, 2.88, 120),
        "x_12gf": BlockParamsConfig(19, 168, 73.36, 2.37, 112),
        "x_16gf": BlockParamsConfig(22, 216, 55.59, 2.1, 128),
        "x_32gf": BlockParamsConfig(23, 320, 69.86, 2.0, 168),
        "y_200mf": BlockParamsConfig(13, 24, 36.44, 2.49, 8, se_ratio=0.25),
        "y_400mf": BlockParamsConfig(16, 48, 27.89, 2.09, 8, se_ratio=0.25),
        "y_600mf": BlockParamsConfig(15, 48, 32.54, 2.32, 16, se_ratio=0.25),
        "y_800mf": BlockParamsConfig(14, 56, 38.84, 2.4, 16, se_ratio=0.25),
        "y_1.6gf": BlockParamsConfig(27, 48, 20.71, 2.65, 24, se_ratio=0.25),
        "y_3.2gf": BlockParamsConfig(21, 80, 42.63, 2.66, 24, se_ratio=0.25),
        "y_4.0gf": BlockParamsConfig(22, 96, 31.41, 2.24, 64, se_ratio=0.25),
        "y_6.4gf": BlockParamsConfig(25, 112, 33.22, 2.27, 72, se_ratio=0.25),
        "y_8.0gf": BlockParamsConfig(17, 192, 76.82, 2.19, 56, se_ratio=0.25),
        "y_12gf": BlockParamsConfig(19, 168, 73.36, 2.37, 112, se_ratio=0.25),
        "y_16gf": BlockParamsConfig(18, 200, 106.23, 2.48, 112, se_ratio=0.25),
        "y_32gf": BlockParamsConfig(20, 232, 115.89, 2.53, 232, se_ratio=0.25),
    }

    if network_mode in block_params_config:
        regnet_block_params_cfg = block_params_config[network_mode]

        stage_depths = regnet_block_params_cfg.depths
        stage_widths = regnet_block_params_cfg.widths
        stage_groups = regnet_block_params_cfg.stage_groups
        bottleneck_multipliers = regnet_block_params_cfg.bottleneck_multipliers
        strides = regnet_block_params_cfg.strides
        se_ratio = regnet_block_params_cfg.se_ratio

        for i, layer_name in enumerate([f"layer{i}" for i in range(1, 5)]):
            network_config[layer_name] = {
                "depth": stage_depths[i],
                "width": stage_widths[i],
                "groups": stage_groups[i],
                "stride": strides[i],
                "bottleneck_multiplier": bottleneck_multipliers[i],
                "se_ratio": se_ratio,
            }
    else:
        logger.error(
            f"Current supported modes for RegNet are {', '.join(supported_modes)}. Got: {network_mode}"
        )

    return network_config
