#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
from typing import Optional

import torch
from torch import Tensor, nn

from corenet.modeling.layers.base_layer import BaseLayer
from corenet.modeling.layers.dropout import Dropout


class SinusoidalPositionalEncoding(BaseLayer):
    """
    This layer adds sinusoidal positional embeddings to a 3D input tensor. The code has been adapted from
    `Pytorch tutorial <https://pytorch.org/tutorials/beginner/transformer_tutorial.html>`_

    Args:
        d_model (int): dimension of the input tensor
        dropout (Optional[float]): Dropout rate. Default: 0.0
        max_len (Optional[int]): Max. number of patches (or seq. length). Default: 5000
        channels_last (Optional[bool]): Channels dimension is the last in the input tensor

    Shape:
        - Input: :math:`(N, C, P)` or :math:`(N, P, C)` where :math:`N` is the batch size, :math:`C` is the embedding dimension,
            :math:`P` is the number of patches
        - Output: same shape as the input

    """

    def __init__(
        self,
        d_model: int,
        dropout: Optional[float] = 0.0,
        max_len: Optional[int] = 5000,
        channels_last: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:

        position_last = not channels_last

        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        # add dummy batch dimension
        pos_encoding = pos_encoding.unsqueeze(0)  # [1 x C x P_max)

        patch_dim = -2  # patch dimension is second last (N, P, C)
        if position_last:
            pos_encoding = pos_encoding.transpose(
                1, 2
            )  # patch dimension is last (N, C, P)
            patch_dim = -1  # patch dimension is last (N, C, P)

        super().__init__()

        self.dropout = Dropout(p=dropout)
        self.patch_dim = patch_dim
        self.register_buffer("pe", pos_encoding)

    def forward_patch_last(
        self, x, indices: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        # seq_length should be the last dim
        if indices is None:
            x = x + self.pe[..., : x.shape[-1]]
        else:
            ndim = x.ndim
            repeat_size = [x.shape[0]] + [-1] * (ndim - 1)

            pe = self.pe.expand(repeat_size)
            selected_pe = torch.gather(pe, index=indices, dim=-1)
            x = x + selected_pe
        return self.dropout(x)

    def forward_others(
        self, x, indices: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        # seq_length should be the second last dim
        if indices is None:
            x = x + self.pe[..., : x.shape[-2], :]
        else:
            ndim = x.ndim
            repeat_size = [x.shape[0]] + [-1] * (ndim - 1)

            pe = self.pe.expand(repeat_size)
            selected_pe = torch.gather(pe, index=indices, dim=-2)
            x = x + selected_pe
        return self.dropout(x)

    def forward(self, x, indices: Optional[Tensor] = None, *args, **kwargs) -> Tensor:
        if self.patch_dim == -1:
            return self.forward_patch_last(x, indices=indices)
        else:
            return self.forward_others(x, indices=indices)

    def __repr__(self):
        return "{}(dropout={})".format(self.__class__.__name__, self.dropout.p)


class LearnablePositionEncoding(BaseLayer):
    """
    This layer adds learnable positional embeddings to a 3D input tensor.

    Args:
        embed_dim (int): dimension of the input tensor
        num_embeddings (int): number of input embeddings. This is similar to vocab size in NLP.
        dropout (Optional[float]): Dropout rate. Default: 0.0
        channels_last (Optional[bool]): Channels dimension is the last in the input tensor

    Shape:
        - Input: :math:`(N, *, C, P)` or :math:`(N, *, P, C)` where :math:`N` is the batch size, :math:`C` is the embedding dimension,
            :math:`P` is the number of patches
        - Output: same shape as the input

    """

    def __init__(
        self,
        embed_dim: int,
        num_embeddings: int,
        dropout: Optional[float] = 0.0,
        channels_last: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.pos_emb = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embed_dim
        )
        self.channel_last = channels_last
        self.dropout = Dropout(p=dropout)

    def forward(self, x, *args, **kwargs) -> Tensor:
        num_embeddings = x.shape[-2] if self.channel_last else x.shape[-1]
        posistions = torch.arange(num_embeddings, dtype=torch.int64, device=x.device)
        position_emb = self.pos_emb(posistions)
        position_emb = position_emb.expand_as(x)
        x = x + position_emb
        return self.dropout(x)

    def __repr__(self):
        return "{}(embed_dim={}, vocab_size={}, dropout={})".format(
            self.__class__.__name__,
            self.pos_emb.embedding_dim,
            self.pos_emb.num_embeddings,
            self.dropout.p,
        )
