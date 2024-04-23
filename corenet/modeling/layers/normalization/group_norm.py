#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

from torch import Tensor, nn

from corenet.modeling.layers.normalization import register_norm_fn


@register_norm_fn(name="group_norm")
class GroupNorm(nn.GroupNorm):
    """
    Applies a `Group Normalization <https://arxiv.org/abs/1803.08494>`_ over an input tensor

    Args:
        num_groups (int): number of groups to separate the input channels into
        num_features (int): :math:`C` from an expected input of size :math:`(N, C, *)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, *)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        and :math:`*` is the remaining dimensions of the input tensor
        - Output: same shape as the input

    .. note::
        GroupNorm is the same as LayerNorm when `num_groups=1` and it is the same as InstanceNorm when
        `num_groups=C`.
    """

    def __init__(
        self,
        num_groups: int,
        num_features: int,
        eps: Optional[float] = 1e-5,
        affine: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            num_groups=num_groups, num_channels=num_features, eps=eps, affine=affine
        )
