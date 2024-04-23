#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

from torch import Tensor, nn


class Dropout(nn.Dropout):
    """
    This layer, during training, randomly zeroes some of the elements of the input tensor with probability `p`
    using samples from a Bernoulli distribution.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where :math:`N` is the batch size
        - Output: same as the input

    """

    def __init__(
        self, p: Optional[float] = 0.5, inplace: Optional[bool] = False, *args, **kwargs
    ) -> None:
        super().__init__(p=p, inplace=inplace)


class Dropout2d(nn.Dropout2d):
    """
    This layer, during training, randomly zeroes some of the elements of the 4D input tensor with probability `p`
    using samples from a Bernoulli distribution.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the input channels,
            :math:`H` is the input tensor height, and :math:`W` is the input tensor width
        - Output: same as the input

    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__(p=p, inplace=inplace)
