#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from corenet.modeling.layers import BaseLayer


class PositionalEmbedding(BaseLayer):
    def __init__(
        self,
        opts,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        is_learnable: Optional[bool] = False,
        sequence_first: Optional[bool] = False,
        interpolation_mode: Optional[str] = "bilinear",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        module = (
            LearnablePositionalEmbedding
            if is_learnable
            else SinusoidalPositionalEmbedding
        )
        self.pos_embed = module(
            opts,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            sequence_first=sequence_first,
            interpolation_mode=interpolation_mode,
            *args,
            **kwargs
        )

    def forward(self, seq_len: int, *args, **kwargs) -> Tensor:
        return self.pos_embed(seq_len, *args, **kwargs)

    def __repr__(self):
        return self.pos_embed.__repr__()


class LearnablePositionalEmbedding(nn.Module):
    """Learnable Positional embedding"""

    def __init__(
        self,
        opts,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        sequence_first: Optional[bool] = False,
        interpolation_mode: Optional[str] = "bilinear",
        *args,
        **kwargs
    ):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.empty(1, 1, num_embeddings, embedding_dim))
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.sequence_first = sequence_first
        self.interpolation_mode = interpolation_mode

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, mean=0, std=self.embedding_dim**-0.5)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.pos_embed[:, :, self.padding_idx, ...] = 0.0

    def forward(self, seq_len: int, *args, **kwargs) -> Tensor:
        # scale pos embedding
        pos_embed = self.pos_embed
        if self.padding_idx is not None:
            with torch.no_grad():
                pos_embed[:, :, self.padding_idx, ...] = 0.0

        if seq_len != self.num_embeddings:
            pos_embed = F.interpolate(
                pos_embed,
                size=(seq_len, self.embedding_dim),
                mode=self.interpolation_mode,
            )

        # add dummy batch dimension
        if self.sequence_first:
            # Input is of the form [Seq_len, Batch, Embedding_dim]
            return pos_embed.reshape(seq_len, 1, self.embedding_dim)
        else:
            # Input is of the form [Batch, Seq_len, Embedding_dim]
            return pos_embed.reshape(1, seq_len, self.embedding_dim)

    def __repr__(self):
        return "{}(num_embeddings={}, embedding_dim={}, padding_idx={}, sequence_first={})".format(
            self.__class__.__name__,
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
            self.sequence_first,
        )


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(
        self,
        opts,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        sequence_first: Optional[bool] = False,
        interpolation_mode: Optional[str] = "bilinear",
        *args,
        **kwargs
    ):
        super().__init__()
        self.padding_idx = padding_idx
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.sequence_first = sequence_first
        self.interpolation_mode = interpolation_mode
        self.register_buffer("pos_embed", self.get_weights())

    def get_weights(self) -> Tensor:
        """Build sinusoidal embeddings. Adapted from Fairseq."""
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(self.num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).reshape(
            self.num_embeddings, -1
        )
        if self.embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(self.num_embeddings, 1)], dim=1)

        # set embeddings corresponding to padding index to 0
        if self.padding_idx is not None:
            emb[self.padding_idx, :] = 0
        return emb.unsqueeze(0).unsqueeze(0)

    def forward(self, seq_len: int, *args, **kwargs) -> Tensor:
        # scale pos embedding
        pos_embed = self.pos_embed

        if seq_len != self.num_embeddings:
            pos_embed = F.interpolate(
                pos_embed,
                size=(seq_len, self.embedding_dim),
                mode=self.interpolation_mode,
            )

        if self.sequence_first:
            # Input is of the form [Seq_len, Batch, Embedding_dim]
            return pos_embed.reshape(seq_len, 1, self.embedding_dim)
        else:
            # Input is of the form [Batch, Seq_len, Embedding_dim]
            return pos_embed.reshape(1, seq_len, self.embedding_dim)

    def __repr__(self):
        return "{}(num_embeddings={}, embedding_dim={}, padding_idx={}, sequence_first={})".format(
            self.__class__.__name__,
            self.num_embeddings,
            self.embedding_dim,
            self.padding_idx,
            self.sequence_first,
        )
