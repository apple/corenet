#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import math

import pytest
import torch

from corenet.modeling.layers import token_merging


def ensure_equal_in_range(t: torch.Tensor, start: int, end: int) -> None:
    """
    Make sure that elements in the tensor @t are equal between @start and @end,
    but not beyond.

    Args:
        t: The tensor to check.
        start: The start of the equality check.
        end: The end of the equality check.
    """
    prototype = t[start]
    assert torch.all(
        (prototype - t[start:end]).abs() < 1e-3
    ), f"Expected equal values from index {start} to {end}, got tensor {t}."
    assert torch.all(
        prototype != t[end:]
    ), f"Expected values from index {end} onwards not to equal values at {start}, got tensor {t}."


def test_token_merging() -> None:
    # Set the batch_size B, sequence length N, and number of channels C.
    B, N, C = 2, 65, 8
    x = torch.ones(B, N, C)
    key_padding_mask = torch.zeros([B, N])
    # Mask values at and beyond 33.
    key_padding_mask[0, 33:] = float("-inf")
    x[0, 33:] = 10e6

    t = token_merging.TokenMerging(C)
    y, key_padding_mask = t(x, key_padding_mask)
    # Because the large @x values were masked (and then zerod by the
    # TokenMerging operation), the largest value should be far lower than 10e6.
    assert y.max() < 5

    assert y.shape[1] == 33
    assert key_padding_mask.shape == (2, 33)

    # Before downsampling, there were 33 unmasked tokens. Now, there are 17.
    # The first 16 should be identical, but the 17th will be different since
    # it wasn't merged with another token (since the sequence length wasn't
    # divisible by the window size).
    ensure_equal_in_range(y[0], 0, 16)
    ensure_equal_in_range(y[0], 16, 17)
    ensure_equal_in_range(y[0], 17, 33)

    # For the second sample, all samples are identical except the last one.
    ensure_equal_in_range(y[1], 0, 32)

    # The first 17 mask positions should be unaltered.
    assert torch.all(key_padding_mask[0, :17] == 0)
    assert torch.all(key_padding_mask[0, 18:] == float("-inf"))

    # All the mask positions should still be 0.
    assert torch.all(key_padding_mask[1] == 0)


@pytest.mark.parametrize("dim,window", [(8, 2), (8, 3), (16, 4)])
def test_token_merging_shapes(dim: int, window: int) -> None:
    # Set the batch_size B and sequence length N.
    B, N = 2, 65
    x = torch.ones(B, N, dim)
    key_padding_mask = torch.zeros([B, N])
    # Mask values at and beyond 33.
    key_padding_mask[0, 33:] = float("-inf")
    x[0, 33:] = 10e6

    t = token_merging.TokenMerging(dim, window)
    y, key_padding_mask = t(x, key_padding_mask)

    assert y.shape == (B, math.ceil(N / window), dim)
