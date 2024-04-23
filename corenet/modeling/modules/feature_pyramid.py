#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from corenet.modeling.layers import ConvLayer2d, norm_layers_tuple
from corenet.modeling.misc.init_utils import (
    initialize_conv_layer,
    initialize_norm_layers,
)
from corenet.modeling.modules import BaseModule
from corenet.utils import logger


class FeaturePyramidNetwork(BaseModule):
    """
    This class implements the `Feature Pyramid Network <https://arxiv.org/abs/1612.03144>`_ module for object detection.

    Args:
        opts: command-line arguments
        in_channels (List[int]): List of channels at different output strides
        output_strides (List[int]): Feature maps from these output strides will be used in FPN
        out_channels (int): Output channels

    """

    def __init__(
        self,
        opts,
        in_channels: List[int],
        output_strides: List[str],
        out_channels: int,
        *args,
        **kwargs
    ) -> None:

        if isinstance(in_channels, int):
            in_channels = [in_channels]
        if isinstance(output_strides, int):
            output_strides = [output_strides]

        if len(in_channels) != len(output_strides):
            logger.error(
                "For {}, we need the length of input_channels to be the same as the length of output stride. "
                "Got: {} and {}".format(
                    self.__class__.__name__, len(in_channels), len(output_strides)
                )
            )
        assert len(in_channels) == len(output_strides)
        super().__init__(*args, **kwargs)

        self.proj_layers = nn.ModuleDict()
        self.nxn_convs = nn.ModuleDict()

        for os, in_channel in zip(output_strides, in_channels):
            proj_layer = ConvLayer2d(
                opts=opts,
                in_channels=in_channel,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                use_norm=True,
                use_act=False,
            )
            nxn_conv = ConvLayer2d(
                opts=opts,
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                bias=False,
                use_norm=True,
                use_act=False,
            )

            self.proj_layers.add_module(name="os_{}".format(os), module=proj_layer)
            self.nxn_convs.add_module(name="os_{}".format(os), module=nxn_conv)

        self.num_fpn_layers = len(in_channels)
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.output_strides = output_strides

        self.reset_weights()

    def reset_weights(self) -> None:
        """Resets the weights of FPN layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                initialize_conv_layer(m, init_method="xavier_uniform")
            elif isinstance(m, norm_layers_tuple):
                initialize_norm_layers(m)

    def forward(self, x: Dict[str, Tensor], *args, **kwargs) -> Dict[str, Tensor]:
        assert len(x) == self.num_fpn_layers

        # dictionary to store results for fpn
        fpn_out_dict = {"os_".format(os): None for os in self.output_strides}

        # process the last output stride
        os_key = "os_{}".format(self.output_strides[-1])
        prev_x = self.proj_layers[os_key](x[os_key])
        prev_x = self.nxn_convs[os_key](prev_x)
        fpn_out_dict[os_key] = prev_x

        remaining_output_strides = self.output_strides[:-1]

        # bottom-up processing
        for os in remaining_output_strides[::-1]:
            os_key = "os_{}".format(os)
            # 1x1 conv
            curr_x = self.proj_layers[os_key](x[os_key])
            # upsample
            prev_x = F.interpolate(prev_x, size=curr_x.shape[-2:], mode="nearest")
            # add
            prev_x = curr_x + prev_x
            prev_x = self.nxn_convs[os_key](prev_x)
            fpn_out_dict[os_key] = prev_x

        return fpn_out_dict

    def __repr__(self):
        return "{}(in_channels={}, output_strides={} out_channels={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.output_strides,
            self.out_channels,
        )
