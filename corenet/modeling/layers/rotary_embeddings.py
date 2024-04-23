#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Tuple

import torch
from torch import Tensor


def _negate_half(x: Tensor) -> Tensor:
    """
    Computes the negative half of the input tensor along the last dimension.

    Args:
        x: Input tensor.

    Returns:
        Tensor with the negative second half preceding the first half along the last dimension.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(x: Tensor, pos_sin: Tensor, pos_cos: Tensor) -> Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x: Input tensor.
        pos_sin: Sine positional embeddings. The shape of 'pos_sin' embeddings is
                [1, 1, number of key tokens, model dimension].
        pos_cos: Cosine positional embeddings. The shape of 'pos_cos' embeddings is
                [1, 1, number of key tokens, model dimension].

    Returns:
        Tensor with rotary positional embeddings applied.
    """
    return (x * pos_cos) + (_negate_half(x) * pos_sin)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings (aka RoPE) from `RoFormer <https://arxiv.org/abs/2104.09864>`_.

    RoPE encodes the position information of tokens using a rotation matrix, and is able to capture
    explicit relative positional dependencies.

    Args:
        model_dim: The dimensionality of the model's hidden state.
        max_seq_length: Maximum sequence length.
        freq_constant: A constant used for computing frequencies.
    """

    def __init__(
        self, model_dim: int, max_seq_length: int, freq_constant: int = 10000
    ) -> None:
        inv_freq = 1.0 / (
            freq_constant
            ** (torch.arange(0, model_dim, 2, dtype=torch.float32) / model_dim)
        )
        super().__init__()

        self.model_dim = model_dim
        self.freq_constant = freq_constant
        self.max_seq_length = max_seq_length

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = max_seq_length
        self._compute_sin_cos_embeddings(max_seq_length)

    def extra_repr(self) -> str:
        return f"\tmodel_dim={self.model_dim}, max_seq_length={self.max_seq_length}, freq_constant={self.freq_constant}"

    def _compute_sin_cos_embeddings(
        self,
        key_len: int,
        key_device: torch.device = torch.device("cpu"),
        key_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Compute sine and cos embeddings.

        Args:
            key_len: Number of tokens in the key embeddings in the transformer model.
            device: Device where the key embeddings are stored.
            key_dtype: Data type of the key embeddings.

        Returns:
            None

        ...note:
            We recalculate the sine and cosine embeddings if any of the following conditions are met:
                1. The number of tokens in key embeddings are greater than the cached sequence length.
                2. Sine and cosine caches are empty.
                3. The device and data type of sine and cosine embeddings does not match with the key embeddings.
        """
        if (
            key_len > self._cached_seq_length
            or self._cached_cos is None
            or (self._cached_cos is not None and self._cached_cos.device != key_device)
            or (self._cached_cos is not None and self._cached_cos.dtype != key_dtype)
            or self._cached_sin is None
            or (self._cached_sin is not None and self._cached_sin.device != key_device)
            or (self._cached_sin is not None and self._cached_sin.dtype != key_dtype)
        ):
            self._cached_seq_length = max(key_len, self._cached_seq_length)

            # The shape of 'pos_index' is [number of key tokens]
            pos_index = torch.arange(
                self._cached_seq_length,
                dtype=torch.float32,
                device=self.inv_freq.device,
            )
            # The shape of 'pos_index_theta' is [number of key tokens, model dimension]
            pos_index_theta = torch.einsum("i,j->ij", pos_index, self.inv_freq)
            # The shape of 'emb' is [number of key tokens, model dimension]
            emb = torch.cat((pos_index_theta, pos_index_theta), dim=-1)

            # the shape of cos and sin embeddings is [number of key tokens, model_dim]
            cos_emb = emb.cos().to(dtype=key_dtype, device=key_device)
            sin_emb = emb.sin().to(dtype=key_dtype, device=key_device)

            # the shape of cached cos and sin embeddings is [1, 1, number of key tokens, model_dim]
            self._cached_cos = cos_emb[None, None, :, :]
            self._cached_sin = sin_emb[None, None, :, :]

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward function of RoPE embeddings.

        Args:
            query: Query embeddings in the transformer model. The shape of query embeddings is
                [Batch, number of query heads, number of query tokens, model dimension].
            key: Key embeddings in the transformer model. The shape of key embeddings is
                [Batch, number of key heads, number of key tokens, model dimension].

        Returns:
            A tuple containing the query and key embeddings with positional information. The shape of the returned query
            and key embeddings is the same as the input query and key embeddings respectively.

        ...note:
            The RoPE embedding computation is done in full-precision. After the computation, input query and key tensors
            are casted to original input datatype.
        """
        dim = key.shape[-1]
        key_len = key.shape[2]
        query_len = query.shape[2]

        assert dim == self.model_dim
        assert key.device == query.device
        assert key.dtype == query.dtype

        # In the context of self-attention, the lengths of keys and queries are equal.
        # However, in generation tasks, such as predicting the next token in a sequence, the lengths of keys and queries
        # can differ. For instance, when employing key-value (KV) caching for sequence prediction, the keys
        # represent embeddings of previous tokens and the current token, while the query corresponds
        # to the embedding of the current token only.
        assert (
            key_len >= query_len
        ), "Number of keys has to be greater than or equal to number of queries."

        query_float = query.float()
        key_float = key.float()

        self._compute_sin_cos_embeddings(
            key_len, key_device=key_float.device, key_dtype=key_float.dtype
        )
        query_float = _apply_rotary_pos_emb(
            x=query_float,
            pos_sin=self._cached_sin[..., key_len - query_len : key_len, :],
            pos_cos=self._cached_cos[..., key_len - query_len : key_len, :],
        )
        key_float = _apply_rotary_pos_emb(
            x=key_float,
            pos_sin=self._cached_sin[..., :key_len, :],
            pos_cos=self._cached_cos[..., :key_len, :],
        )

        return query_float.type_as(query), key_float.type_as(key)
