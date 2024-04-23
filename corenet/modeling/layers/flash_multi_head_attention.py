#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

import torch
from packaging import version
from torch import Tensor
from torch.nn import functional as F

from corenet.modeling.layers.base_layer import BaseLayer
from corenet.modeling.layers.linear_layer import LinearLayer
from corenet.utils import logger


class FlashMultiHeadSelfAttention(BaseLayer):
    """Multi-head scaled dot-product attention using `flash attention <https://arxiv.org/abs/2205.14135>`_.

    This layer uses efficient scaled-dot product attention implementation to reduce memory footprint and faster training.

    Args:
        in_features: Number of features in the input.
        head_dim: Head dimension.
        attn_dropout_prob: Attention dropout probability. Defaults to 0.0.
        qkv_features: Number of features after linear projection in QKV branch in multi-head
            attention. If None, qkv_features=in_features. Defaults to None.
        bias: Use bias or not. Defaults to False.
    """

    def __init__(
        self,
        in_features: int,
        head_dim: int,
        attn_dropout_prob: float = 0.0,
        qkv_features: Optional[int] = None,
        bias: bool = False,
    ) -> None:
        sdpa_exists = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        min_pytorch_supported_version = "2.0"
        is_supported_pytorch_version_available = version.parse(
            torch.__version__
        ) >= version.parse(min_pytorch_supported_version)

        if not (sdpa_exists and is_supported_pytorch_version_available):
            logger.error(
                f"Please use PyTorch v{min_pytorch_supported_version} and above."
            )

        if qkv_features is None:
            qkv_features = in_features

        if qkv_features % head_dim != 0:
            logger.error(
                f"QKV features should be divisible by head dimension in {self.__class__.__name__}. Got: {qkv_features} qkv_features and {head_dim} head dimension."
            )
        num_attn_heads = qkv_features // head_dim
        super().__init__()

        self.qkv_proj = LinearLayer(
            in_features=in_features,
            out_features=3 * qkv_features,
            bias=bias,
        )
        self.out_proj_attn = LinearLayer(
            in_features=qkv_features, out_features=in_features, bias=bias
        )

        self.attn_dropout_prob = attn_dropout_prob
        self.num_heads = num_attn_heads
        self.head_dim = head_dim
        self.qkv_features = qkv_features

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x: Input tensor of shape [batch size, number of tokens, embedding dim]

        Returns:
            Output tensor of the same size as the input.
        """
        batch_size, num_tokens, embed_dim = x.shape
        qkv = self.qkv_proj(x)
        # [batch size, tokens, embedding dim] -> [batch size, tokens, 3, number of heads, head dim]
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        # [batch size, tokens, 3, number of heads, head dim] --> [batch size, number of heads, 3, tokens, head dim]
        qkv = qkv.transpose(1, 3).contiguous()
        # [batch size, number of heads, 3, tokens, head dim] --> [batch size, number of heads, tokens, head dim] x 3
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # efficient attention using Flash Attention CUDA kernels
        attn_output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.attn_dropout_prob if self.training else 0,
            # For self-attention, causal masking is not required.
            is_causal=False,
        )
        # [batch size, number of heads, tokens, head dim] --> [batch size, tokens, number of heads, head dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch size, tokens, number of heads, head dim] --> [batch size, tokens, number of heads * head dim]
        attn_output = attn_output.reshape(batch_size, num_tokens, self.qkv_features)
        y = self.out_proj_attn(attn_output)
        return y
