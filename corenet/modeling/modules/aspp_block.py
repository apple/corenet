#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from corenet.modeling.layers import (
    AdaptiveAvgPool2d,
    BaseLayer,
    ConvLayer2d,
    Dropout2d,
    SeparableConv2d,
)
from corenet.modeling.modules import BaseModule
from corenet.utils import logger
from corenet.utils.ddp_utils import is_master


class ASPP(BaseModule):
    """
    ASPP module defined in DeepLab papers, `here <https://arxiv.org/abs/1606.00915>`_ and `here <https://arxiv.org/abs/1706.05587>`_

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`
        atrous_rates (Tuple[int]): atrous rates for different branches.
        is_sep_conv (Optional[bool]): Use separable convolution instead of standaard conv. Default: False
        dropout (Optional[float]): Apply dropout. Default is 0.0

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H, W)`
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        atrous_rates: Tuple[int],
        is_sep_conv: Optional[bool] = False,
        dropout: Optional[float] = 0.0,
        *args,
        **kwargs
    ) -> None:
        in_proj = ConvLayer2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=True,
        )
        out_proj = ConvLayer2d(
            opts=opts,
            in_channels=5 * out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=True,
        )
        aspp_layer = ASPPSeparableConv2d if is_sep_conv else ASPPConv2d

        assert len(atrous_rates) == 3

        modules = [in_proj]
        modules.extend(
            [
                aspp_layer(
                    opts=opts,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=rate,
                )
                for rate in atrous_rates
            ]
        )
        modules.append(
            ASPPPooling(opts=opts, in_channels=in_channels, out_channels=out_channels)
        )

        if not (0.0 <= dropout < 1.0):
            if is_master(opts):
                logger.warning(
                    "Dropout value in {} should be between 0 and 1. Got: {}. Setting it to 0.0".format(
                        self.__class__.__name__, dropout
                    )
                )
            dropout = 0.0

        super().__init__()
        self.convs = nn.ModuleList(modules)
        self.project = out_proj

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.atrous_rates = atrous_rates
        self.is_sep_conv_layer = is_sep_conv
        self.n_atrous_branches = len(atrous_rates)
        self.dropout_layer = Dropout2d(p=dropout)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        out = []
        for conv in self.convs:
            out.append(conv(x))
        out = torch.cat(out, dim=1)
        out = self.project(out)
        out = self.dropout_layer(out)
        return out

    def __repr__(self):
        return "{}(in_channels={}, out_channels={}, atrous_rates={}, is_aspp_sep={}, dropout={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.atrous_rates,
            self.is_sep_conv_layer,
            self.dropout_layer.p,
        )


class ASPPConv2d(ConvLayer2d):
    """
    Convolution with a dilation  for the ASPP module
    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`
        dilation (int): Dilation rate

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H, W)`
    """

    def __init__(
        self, opts, in_channels: int, out_channels: int, dilation: int, *args, **kwargs
    ) -> None:
        super().__init__(
            opts=opts,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            use_norm=True,
            use_act=True,
            dilation=dilation,
        )

    def adjust_atrous_rate(self, rate: int) -> None:
        """This function allows to adjust the dilation rate"""
        self.block.conv.dilation = rate
        # padding is the same here
        # see ConvLayer to see the method for computing padding
        self.block.conv.padding = rate


class ASPPSeparableConv2d(SeparableConv2d):
    """
    Separable Convolution with a dilation for the ASPP module
    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`
        dilation (int): Dilation rate

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H, W)`
    """

    def __init__(
        self, opts, in_channels: int, out_channels: int, dilation: int, *args, **kwargs
    ) -> None:
        super().__init__(
            opts=opts,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            use_norm=True,
            use_act=True,
        )

    def adjust_atrous_rate(self, rate: int) -> None:
        """This function allows to adjust the dilation rate"""
        self.dw_conv.block.conv.dilation = rate
        # padding is the same here
        # see ConvLayer to see the method for computing padding
        self.dw_conv.block.conv.padding = rate


class ASPPPooling(BaseLayer):
    """
    ASPP pooling layer
    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`

    Shape:
        - Input: :math:`(N, C_{in}, H, W)`
        - Output: :math:`(N, C_{out}, H, W)`
    """

    def __init__(
        self, opts, in_channels: int, out_channels: int, *args, **kwargs
    ) -> None:
        super().__init__()
        self.aspp_pool = nn.Sequential()
        self.aspp_pool.add_module(
            name="global_pool", module=AdaptiveAvgPool2d(output_size=1)
        )
        self.aspp_pool.add_module(
            name="conv_1x1",
            module=ConvLayer2d(
                opts=opts,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                use_norm=True,
                use_act=True,
            ),
        )

        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        x_size = x.shape[-2:]
        x = self.aspp_pool(x)
        x = F.interpolate(x, size=x_size, mode="bilinear", align_corners=False)
        return x

    def __repr__(self):
        return "{}(in_channels={}, out_channels={})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )
