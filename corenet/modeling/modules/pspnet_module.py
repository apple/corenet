#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from corenet.modeling.layers import AdaptiveAvgPool2d, ConvLayer2d, Dropout2d
from corenet.modeling.modules import BaseModule
from corenet.utils import logger


class PSP(BaseModule):
    """
    This class defines the Pyramid Scene Parsing module in the `PSPNet paper <https://arxiv.org/abs/1612.01105>`_

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H, W)`
        pool_sizes Optional[Tuple[int, ...]]: List or Tuple of pool sizes. Default: (1, 2, 3, 6)
        dropout (Optional[float]): Apply dropout. Default is 0.0
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        out_channels: int,
        pool_sizes: Optional[Tuple[int, ...]] = (1, 2, 3, 6),
        dropout: Optional[float] = 0.0,
        *args,
        **kwargs
    ) -> None:
        if not (0.0 <= dropout < 1.0):
            logger.error(
                "Dropout value in {} should be between 0 and 1. Got: {}".format(
                    self.__class__.__name__, dropout
                )
            )
        reduction_dim = in_channels // len(pool_sizes)
        reduction_dim = (reduction_dim // 16) * 16
        channels_after_concat = (reduction_dim * len(pool_sizes)) + in_channels

        super().__init__()
        self.psp_branches = nn.ModuleList(
            [
                self._make_psp_layer(
                    opts, o_size=ps, in_channels=in_channels, out_channels=reduction_dim
                )
                for ps in pool_sizes
            ]
        )
        self.fusion = nn.Sequential(
            ConvLayer2d(
                opts=opts,
                in_channels=channels_after_concat,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                use_norm=True,
                use_act=True,
            ),
            Dropout2d(p=dropout),
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_sizes = pool_sizes
        self.inner_channels = reduction_dim
        self.dropout = dropout

    @staticmethod
    def _make_psp_layer(
        opts, o_size: int, in_channels: int, out_channels: int
    ) -> nn.Module:
        return nn.Sequential(
            AdaptiveAvgPool2d(output_size=(o_size, o_size)),
            ConvLayer2d(
                opts,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                use_norm=True,
                use_act=True,
            ),
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        x_size = x.shape[2:]
        out = [x] + [
            F.interpolate(
                input=psp_branch(x), size=x_size, mode="bilinear", align_corners=True
            )
            for psp_branch in self.psp_branches
        ]
        out = torch.cat(out, dim=1)
        out = self.fusion(out)
        return out

    def __repr__(self):
        return "{}(in_channels={}, out_channels={}, pool_sizes={}, inner_channels={}, dropout_2d={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.pool_sizes,
            self.inner_channels,
            self.dropout,
        )
