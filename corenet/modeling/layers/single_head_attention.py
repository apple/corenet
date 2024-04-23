#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from corenet.modeling.layers.base_layer import BaseLayer
from corenet.modeling.layers.dropout import Dropout
from corenet.modeling.layers.linear_layer import LinearLayer


class SingleHeadAttention(BaseLayer):
    """
    This layer applies a single-head attention as described in `DeLighT <https://arxiv.org/abs/2008.00623>`_ paper

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        attn_dropout (Optional[float]): Attention dropout. Default: 0.0
        bias (Optional[bool]): Use bias or not. Default: ``True``

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input

    """

    def __init__(
        self,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.qkv_proj = LinearLayer(
            in_features=embed_dim, out_features=3 * embed_dim, bias=bias
        )

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = LinearLayer(
            in_features=embed_dim, out_features=embed_dim, bias=bias
        )

        self.softmax = nn.Softmax(dim=-1)
        self.embed_dim = embed_dim
        self.scaling = self.embed_dim**-0.5

    def __repr__(self) -> str:
        return "{}(embed_dim={}, attn_dropout={})".format(
            self.__class__.__name__, self.embed_dim, self.attn_dropout.p
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs
    ) -> Tensor:
        # [N, P, C] --> [N, P, 3C]
        if x_kv is None:
            qkv = self.qkv_proj(x_q)
            # [N, P, 3C] --> [N, P, C] x 3
            query, key, value = torch.chunk(qkv, chunks=3, dim=-1)
        else:
            query = F.linear(
                x_q,
                weight=self.qkv_proj.weight[: self.embed_dim, ...],
                bias=self.qkv_proj.bias[: self.embed_dim],
            )

            # [N, P, C] --> [N, P, 2C]
            kv = F.linear(
                x_kv,
                weight=self.qkv_proj.weight[self.embed_dim :, ...],
                bias=self.qkv_proj.bias[self.embed_dim :],
            )
            key, value = torch.chunk(kv, chunks=2, dim=-1)

        query = query * self.scaling

        # [N, P, C] --> [N, C, P]
        key = key.transpose(-2, -1)

        # QK^T
        # [N, P, C] x [N, C, P] --> [N, P, P]
        attn = torch.matmul(query, key)

        if attn_mask is not None:
            # attn_mask shape should be the same as attn
            assert list(attn_mask.shape) == list(
                attn.shape
            ), "Shape of attention mask and attn should be the same. Got: {} and {}".format(
                attn_mask.shape, attn.shape
            )
            attn = attn + attn_mask

        if key_padding_mask is not None:
            # Do not attend to padding positions
            # key padding mask size is [N, P]
            batch_size, num_src_tokens, num_tgt_tokens = attn.shape
            assert key_padding_mask.dim() == 2 and list(key_padding_mask.shape) == [
                batch_size,
                num_tgt_tokens,
            ], "Key_padding_mask should be 2-dimension with shape [{}, {}]. Got: {}".format(
                batch_size, num_tgt_tokens, key_padding_mask.shape
            )
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).to(torch.bool),
                float("-inf"),
            )

        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, P, P] x [N, P, C] --> [N, P, C]
        out = torch.matmul(attn, value)
        out = self.out_proj(out)

        return out
