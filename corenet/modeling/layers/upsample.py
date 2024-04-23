#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn


class UpSample(nn.Upsample):
    """
    This layer upsamples a given input tensor.

    Args:
        size (Optional[Union[int, Tuple[int, ...]]): Output spatial size. Default: None
        scale_factor (Optional[float]): Scale each spatial dimension of the input by this factor. Default: None
        mode (Optional[str]): Upsampling algorithm (``'nearest'``, ``'linear'``, ``'bilinear'``, ``'bicubic'`` and ``'trilinear'``. Default: ``'nearest'``
        align_corners (Optional[bool]): if ``True``, the corner pixels of the input and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is ``'linear'``, ``'bilinear'``, ``'bicubic'``, or ``'trilinear'``.
            Default: ``None``

    Shape:
        - Input: :math:`(N, C, W_{in})` or :math:`(N, C, H_{in}, W_{in})` or :math:`(N, C, D_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C, W_{out})` or :math:`(N, C, H_{out}, W_{out})` or :math:`(N, C, D_{out}, H_{out}, W_{out})`
    """

    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, ...]]] = None,
        scale_factor: Optional[float] = None,
        mode: Optional[str] = "nearest",
        align_corners: Optional[bool] = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners
        )
