#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
from torch import Tensor, nn

from corenet.modeling.layers.normalization import register_norm_fn


@register_norm_fn(name="rms_norm")
class RMSNorm(nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-6, *unused_args, **unused_kwargs
    ) -> None:
        """
        `Root mean square (RMS) normalization layer <https://arxiv.org/abs/1910.07467>`_.

        Args:
            num_features: The dimension of the input tensor.
            eps: A small value added to the denominator during normalization for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.num_features = num_features

    def _norm(self, x: Tensor) -> Tensor:
        """Apply the RMSNorm normalization to the input tensor.

        Args:
            x: The input tensor. The shape of the input tensor is [batch size, *, num features],
                where * denotes any dimensions.

        Returns:
            The normalized tensor. The shape of the normalized tensor is the same as the input.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the RMSNorm layer.

        Args:
            x: The input tensor. The shape of the input tensor is [batch size, *, num features],
                where * denotes any dimensions.

        Returns:
            The output tensor after applying RMSNorm. The shape of the output tensor is the same
            as the input tensor.

        ...note:
            The input is first converted to full precision and then normalized using RMSNorm.
            The resulting output is then converted back to its original data type.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def extra_repr(self) -> str:
        return (
            super().extra_repr() + f"num_features={self.num_features}, eps={self.eps}"
        )
