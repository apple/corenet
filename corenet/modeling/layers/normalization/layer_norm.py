#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import List, Optional, Union

import torch
from torch import Size, Tensor, nn

from corenet.modeling.layers.normalization import register_norm_fn


@register_norm_fn(name="layer_norm")
class LayerNorm(nn.LayerNorm):
    r"""
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a input tensor

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): If ``True``, use learnable affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, *)` where :math:`N` is the batch size
        - Output: same shape as the input
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
        )

    def forward(self, x: Tensor) -> Tensor:
        n_dim = x.ndim
        if x.shape[1] == self.normalized_shape[0] and n_dim > 2:  # channel-first format
            s, u = torch.std_mean(x, dim=1, keepdim=True, unbiased=False)
            x = (x - u) / (s + self.eps)
            if self.weight is not None:
                # Using fused operation for performing affine transformation: x = (x * weight) + bias
                n_dim = x.ndim - 2
                new_shape = [1, self.normalized_shape[0]] + [1] * n_dim
                x = torch.addcmul(
                    input=self.bias.reshape(*[new_shape]),
                    value=1.0,
                    tensor1=x,
                    tensor2=self.weight.reshape(*[new_shape]),
                )
            return x
        elif x.shape[-1] == self.normalized_shape[0]:  # channel-last format
            return super().forward(x)
        else:
            raise NotImplementedError(
                "LayerNorm is supported for channel-first and channel-last format only"
            )


@register_norm_fn(name="layer_norm_2d")
@register_norm_fn(name="layer_norm_nchw")
class LayerNorm2D_NCHW(nn.GroupNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a 4D input tensor

    Args:
        num_features (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): If ``True``, use learnable affine parameters. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`H` is the input height, and :math:`W` is the input width
        - Output: same shape as the input
    """

    def __init__(
        self,
        num_features: int,
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(
            num_channels=num_features, eps=eps, affine=elementwise_affine, num_groups=1
        )
        self.num_channels = num_features

    def __repr__(self):
        return "{}(num_channels={}, eps={}, affine={})".format(
            self.__class__.__name__, self.num_channels, self.eps, self.affine
        )


@register_norm_fn(name="layer_norm_fp32")
class LayerNormFP32(LayerNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a input tensor with FP32 precision
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int], Size],
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            *args,
            **kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        # Convert input from dtype X to FP32 and perform normalization operation.
        # This may help with underflow/overflow issues that we typically see with normalization layers
        inp_dtype = x.dtype
        return super().forward(x.to(torch.float32)).to(inp_dtype)
