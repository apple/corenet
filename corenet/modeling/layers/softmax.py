#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

from torch import Tensor, nn


class Softmax(nn.Softmax):
    """
    Applies the Softmax function to an input tensor along the specified dimension

    Args:
        dim (int): Dimension along which softmax to be applied. Default: -1

    Shape:
        - Input: :math:`(*)` where :math:`*` is one or more dimensions
        - Output: same shape as the input
    """

    def __init__(self, dim: Optional[int] = -1, *args, **kwargs):
        super().__init__(dim=dim)
