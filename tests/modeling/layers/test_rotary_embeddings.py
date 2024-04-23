#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import pytest
import torch

from corenet.modeling.layers import RotaryEmbedding


@pytest.mark.parametrize(
    "model_dim,n_queries,n_keys,n_groups",
    [
        (18, 5, 5, 1),
        (18, 5, 6, 4),
    ],
)
def test_rotary_embedding(
    model_dim: int, n_queries: int, n_keys: int, n_groups: int
) -> None:
    """Test for RoPE embeddings."""
    rope_embedding = RotaryEmbedding(
        model_dim=model_dim,
        # setting max_seq_length to the same as number of queries.
        # When n_keys > n_queries, then cos and sine embeddings are re-computed.
        max_seq_length=n_queries,
    )

    batch_size = 2
    n_query_heads = 16
    # When n_groups != 1, RoPE with GQA is tested
    n_key_heads = n_query_heads // n_groups

    query_tensor = torch.randn(
        size=(batch_size, n_query_heads, n_queries, model_dim),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    key_tensor = torch.randn(
        size=(batch_size, n_key_heads, n_keys, model_dim),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    query_tensor_with_rope, key_tensor_with_rope = rope_embedding(
        query_tensor, key_tensor
    )
    assert rope_embedding._cached_seq_length == n_keys
    assert query_tensor.shape == query_tensor_with_rope.shape
    assert key_tensor.shape == key_tensor_with_rope.shape

    assert torch.isnan(query_tensor_with_rope).to(torch.bool).sum() == 0
    assert torch.isnan(key_tensor_with_rope).to(torch.bool).sum() == 0
