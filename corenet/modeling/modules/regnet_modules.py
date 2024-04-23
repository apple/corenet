#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import List, Optional, Tuple

from torch import Tensor, nn

from corenet.modeling.layers import ConvLayer2d, Identity, StochasticDepth
from corenet.modeling.layers.activation import build_activation_layer
from corenet.modeling.modules import BaseModule, SqueezeExcitation


class XRegNetBlock(BaseModule):
    """
    This class implements the `X` block based on the ResNet bottleneck block. See figure 4 of RegNet
    paper `RegNet model <https://arxiv.org/pdf/2003.13678.pdf>`_

    Args:
        opts: command-line arguments
        width_in: The number of input channels
        width_out: The number of output channels
        stride: Stride for convolution
        groups: Number of groups for convolution
        bottleneck_multiplier: The number of in/out channels of the intermediate
            conv layer will be scaled by this value
        se_ratio: The numer squeeze-excitation ratio. The number of channels in the SE
            module will be scaled by this value
        stochastic_depth_prob: The stochastic depth probability
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        width_in: int,
        width_out: int,
        stride: int,
        groups: int,
        bottleneck_multiplier: float,
        se_ratio: float,
        stochastic_depth_prob: float = 0.0,  # drop probability (= 1 - survival_prob)
    ) -> None:
        super().__init__()

        bottleneck_width = int(round(width_out * bottleneck_multiplier))
        bottleneck_groups = bottleneck_width // groups

        conv_1x1_1 = ConvLayer2d(
            opts=opts,
            in_channels=width_in,
            out_channels=bottleneck_width,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=True,
        )

        conv_3x3 = ConvLayer2d(
            opts=opts,
            in_channels=bottleneck_width,
            out_channels=bottleneck_width,
            kernel_size=3,
            stride=stride,
            groups=bottleneck_groups,
            use_norm=True,
            use_act=True,
        )

        se = Identity()
        if se_ratio > 0:
            squeeze_channels = int(round(se_ratio * width_in))
            se = SqueezeExcitation(
                opts,
                in_channels=bottleneck_width,
                squeeze_channels=squeeze_channels,
            )

        conv_1x1_2 = ConvLayer2d(
            opts=opts,
            in_channels=bottleneck_width,
            out_channels=width_out,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=True,
        )

        block = nn.Sequential()
        block.add_module("conv_1x1_1", module=conv_1x1_1)
        block.add_module("conv_3x3", module=conv_3x3)
        block.add_module("se", module=se)
        block.add_module("conv_1x1_2", module=conv_1x1_2)

        down_sample = Identity()
        if stride != 1 or width_out != width_in:
            down_sample = ConvLayer2d(
                opts,
                in_channels=width_in,
                out_channels=width_out,
                kernel_size=1,
                stride=stride,
                use_act=False,
            )

        act_type = getattr(opts, "model.activation.name")
        neg_slope = getattr(opts, "model.activation.neg_slope")
        inplace = getattr(opts, "model.activation.inplace")
        final_act = build_activation_layer(
            opts=opts,
            act_type=act_type,
            inplace=inplace,
            negative_slope=neg_slope,
            num_parameters=width_out,
        )

        self.stochastic_depth = StochasticDepth(p=stochastic_depth_prob, mode="row")

        self.block = block
        self.down_sample = down_sample
        self.final_act = final_act
        self.width_in = width_in
        self.width_out = width_out
        self.stride = stride
        self.groups = groups
        self.bottleneck_multiplier = bottleneck_multiplier
        self.se_ratio = se_ratio
        self.stochastic_depth_prob = stochastic_depth_prob

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for XRegNetBlock.

        Args:
            x: Batch of images

        Retruns:
            * output of XRegNetBlock including stochastic depth layer and
                residual.

        Shape:
            x: :math:`(N, C_{in}, H_{in}, W_{in})`
            Output: :math:`(N, C_{out}, H_{out}, W_{out})`
        """
        out = self.block(x)
        out = self.stochastic_depth(out)
        res = self.down_sample(x)
        out = res + out
        return self.final_act(out)

    def __repr__(self) -> str:
        return "{}(width_in={}, width_out={}, stride={}, groups={}, bottleneck_multiplier={}, se_ratio={}, stochastic_depth_prob={})".format(
            self.__class__.__name__,
            self.width_in,
            self.width_out,
            self.stride,
            self.groups,
            self.bottleneck_multiplier,
            self.se_ratio,
            self.stochastic_depth_prob,
        )


class AnyRegNetStage(BaseModule):
    """
    This class implements a 'stage' as defined in the `RegNet paper <https://arxiv.org/pdf/2003.13678.pdf>`_.
    It consists of a sequence of bottleneck blocks.

    Args:
        opts: command-line arguments
        depth: The number of XRegNetBlocks in the stage
        width_in: The number of input channels of the first block
        width_out: The number of output channels of each block
        stride: Stride for convolution of first block
        groups: Number of groups for the intermediate convolution (bottleneck) layer in each block
        bottleneck_multiplier: The number of in/out channels of the intermediate
            conv layer of each block will be scaled by this value
        se_ratio: The numer squeeze-excitation ratio. The number of channels in the SE
            module of each block will be scaled by this value
        stage_depths: A list of the number of blocks in each stage
        stage_index: The index of the current stage being constructed
        stochastic_depth_prob: The stochastic depth probability
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        depth: int,
        width_in: int,
        width_out: int,
        stride: int,
        groups: int,
        bottleneck_multiplier: float,
        se_ratio: float,
        stage_index: int,
        stochastic_depth_probs: List[float],
    ) -> None:

        super().__init__()

        stage_blocks = nn.Sequential()

        for i, sd_prob in enumerate(stochastic_depth_probs):
            block = XRegNetBlock(
                opts,
                width_in=width_in if i == 0 else width_out,
                width_out=width_out,
                stride=stride if i == 0 else 1,
                groups=groups,
                bottleneck_multiplier=bottleneck_multiplier,
                se_ratio=se_ratio,
                stochastic_depth_prob=sd_prob,
            )
            stage_blocks.add_module(f"Stage{stage_index}-Block{i}", module=block)

        self.stage = stage_blocks
        self.depth = depth
        self.width_in = width_in
        self.width_out = width_out
        self.stride = stride
        self.groups = groups
        self.bottleneck_multiplier = bottleneck_multiplier
        self.se_ratio = se_ratio
        self.stage_index = stage_index
        self.stochastic_depth_probs = stochastic_depth_probs

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through all blocks in the stage.

        Args:
            x: Batch of images.

        Returns:
            * output of passing x through all blocks in the stage.

        Shape:
            x: :math:`(N, C_{in}, H_{in}, W_{in})`
            Output: :math:`(N, C_{out}, H_{out}, W_{out})`
        """
        return self.stage(x)

    def __repr__(self) -> str:
        return "{}(depth={}, width_in={}, width_out={}, stride={}, groups={}, bottleneck_multiplier={}, se_ratio={}, stage_index={}, stochastic_depth_probs={})".format(
            self.__class__.__name__,
            self.depth,
            self.width_in,
            self.width_out,
            self.stride,
            self.groups,
            self.bottleneck_multiplier,
            self.se_ratio,
            self.stage_index,
            self.stochastic_depth_probs,
        )
