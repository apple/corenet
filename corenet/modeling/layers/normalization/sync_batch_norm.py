#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

import torch
from torch import Tensor, nn

from corenet.modeling.layers.normalization import register_norm_fn


@register_norm_fn(name="sync_batch_norm")
class SyncBatchNorm(nn.SyncBatchNorm):
    """
    Applies a `Synchronized Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over the input tensor

    Args:
        num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C, *)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

    Shape:
        - Input: :math:`(N, C, *)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`*` is the remaining input dimensions
        - Output: same shape as the input

    """

    def __init__(
        self,
        num_features: int,
        eps: Optional[float] = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )


@register_norm_fn(name="sync_batch_norm_fp32")
class SyncBatchNormFP32(SyncBatchNorm):
    """
    Synchronized BN in FP32
    """

    def __init__(
        self,
        num_features: int,
        eps: Optional[float] = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: Optional[bool] = True,
        track_running_stats: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        in_dtype = x.dtype
        return super().forward(x.to(dtype=torch.float)).to(dtype=in_dtype)
