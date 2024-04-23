#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

import torch
from torch import Tensor, nn

from corenet.modeling.layers import AdaptiveAvgPool2d, ConvLayer2d
from corenet.modeling.layers.activation import build_activation_layer
from corenet.modeling.modules import BaseModule
from corenet.utils.math_utils import make_divisible


class SqueezeExcitation(BaseModule):
    """
    This class defines the Squeeze-excitation module, in the `SENet paper <https://arxiv.org/abs/1709.01507>`_

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        squeeze_factor (Optional[int]): Reduce :math:`C` by this factor. Default: 4
        squeeze_channels (Optional[int]): This module's output channels. Overrides squeeze_factor if specified
        scale_fn_name (Optional[str]): Scaling function name. Default: sigmoid

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)`
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        squeeze_factor: Optional[int] = 4,
        squeeze_channels: Optional[int] = None,
        scale_fn_name: Optional[str] = "sigmoid",
        *args,
        **kwargs
    ) -> None:
        if squeeze_channels is None:
            squeeze_channels = max(make_divisible(in_channels // squeeze_factor, 8), 32)

        fc1 = ConvLayer2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=squeeze_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            use_norm=False,
            use_act=True,
        )
        fc2 = ConvLayer2d(
            opts=opts,
            in_channels=squeeze_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True,
            use_norm=False,
            use_act=False,
        )
        act_fn = build_activation_layer(opts, act_type=scale_fn_name, inplace=True)
        super().__init__()
        self.se_layer = nn.Sequential()
        self.se_layer.add_module(
            name="global_pool", module=AdaptiveAvgPool2d(output_size=1)
        )
        self.se_layer.add_module(name="fc1", module=fc1)
        self.se_layer.add_module(name="fc2", module=fc2)
        self.se_layer.add_module(name="scale_act", module=act_fn)

        self.in_channels = in_channels
        self.squeeze_factor = squeeze_factor
        self.scale_fn = scale_fn_name

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return x * self.se_layer(x)

    def __repr__(self) -> str:
        return "{}(in_channels={}, squeeze_factor={}, scale_fn={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.squeeze_factor,
            self.scale_fn,
        )
