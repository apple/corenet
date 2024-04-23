#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

from torch import Tensor, nn


class MaxPool2d(nn.MaxPool2d):
    """
    Applies a 2D max pooling over a 4D input tensor.

    Args:
        kernel_size (Optional[int]): the size of the window to take a max over
        stride (Optional[int]): The stride of the window. Default: 2
        padding (Optional[int]): Padding to be added on both sides of the tensor. Default: 1

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` where :math:`N` is the batch size, :math:`C` is the input channels,
            :math:`H_{in}` is the input height, and :math:`W_{in}` is the input width
        - Output: :math:`(N, C, H_{out}, W_{out})` where :math:`H_{out}` is the output height, and :math:`W_{in}` is
            the output width
    """

    def __init__(
        self,
        kernel_size: Optional[int] = 3,
        stride: Optional[int] = 2,
        padding: Optional[int] = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(kernel_size=kernel_size, stride=stride, padding=padding)

    def __repr__(self):
        return "{}(kernel_size={}, stride={})".format(
            self.__class__.__name__, self.kernel_size, self.stride
        )


class AvgPool2d(nn.AvgPool2d):
    """
    Applies a 2D average pooling over a 4D input tensor.

    Args:
        kernel_size (Optional[int]): the size of the window to take a max over
        stride (Optional[int]): The stride of the window. Default: 2
        padding (Optional[int]): Padding to be added on both sides of the tensor. Default: 1
        ceil_mode (Optional[bool]): When True, will use `ceil` instead of `floor` to compute the output shape. Default: False
        count_include_pad (Optional[bool]): When True, will include the zero-padding in the averaging calculation. Default: True
        divisor_override: if specified, it will be used as divisor, otherwise size of the pooling region will be used. Default: None

    Shape:
        - Input: :math:`(N, C, H_{in}, W_{in})` where :math:`N` is the batch size, :math:`C` is the input channels,
            :math:`H_{in}` is the input height, and :math:`W_{in}` is the input width
        - Output: :math:`(N, C, H_{out}, W_{out})` where :math:`H_{out}` is the output height, and :math:`W_{in}` is
            the output width
    """

    def __init__(
        self,
        kernel_size: tuple,
        stride: Optional[tuple] = None,
        padding: Optional[tuple] = (0, 0),
        ceil_mode: Optional[bool] = False,
        count_include_pad: Optional[bool] = True,
        divisor_override: Optional[bool] = None,
    ):
        super(AvgPool2d, self).__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )

    def __repr__(self):
        return "{}(upscale_factor={})".format(
            self.__class__.__name__, self.upscale_factor
        )
