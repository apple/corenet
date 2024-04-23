#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Tuple, Union

from torch import Tensor, nn


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """
    Applies a 2D adaptive average pooling over an input tensor.

    Args:
        output_size (Optional, int or Tuple[int, int]): The target output size. If a single int :math:`h` is passed,
        then a square output of size :math:`hxh` is produced. If a tuple of size :math:`hxw` is passed, then an
        output of size `hxw` is produced. Default is 1.
    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`H` is the input height, and :math:`W` is the input width
        - Output: :math:`(N, C, h, h)` or :math:`(N, C, h, w)`
    """

    def __init__(
        self, output_size: Union[int, Tuple[int, int]] = 1, *args, **kwargs
    ) -> None:
        super().__init__(output_size=output_size)
