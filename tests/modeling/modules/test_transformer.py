#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

import torch

from corenet.modeling.modules import TransformerEncoder


def get_opts() -> argparse.Namespace:
    opts = argparse.Namespace()
    setattr(opts, "model.normalization.groups", None)
    setattr(opts, "model.normalization.momentum", 0.9)
    setattr(opts, "model.activation.name", "relu")
    setattr(opts, "model.activation.inplace", False)
    setattr(opts, "model.activation.neg_slope", False)
    return opts


def ensure_equal_in_range(t: torch.Tensor, start: int, end: int) -> None:
    """
    Ensure values of @t are equal from @start to @end, but not after @end.

    The tensor can have any number of dimensions greater than 0. The first
    dimension is the dimension indexed by @start and @end.

    Args:
        t: The tensor to check.
        start: The start index.
        end: The end index.
    """
    prototype = t[start]
    assert torch.all((prototype - t[start:end]).abs() < 1e-3)
    assert torch.all(prototype != t[end:])


def test_masked_attention() -> None:
    opts = get_opts()

    B, N, C = 2, 64 + 2, 8
    t = TransformerEncoder(opts, embed_dim=C, ffn_latent_dim=4 * C)
    prototype = torch.randn([C])
    x = torch.ones([B, N, C])
    x[:, :] = prototype

    key_padding_mask = torch.zeros([B, N])
    key_padding_mask[0, 63:] = float("-inf")
    # Mask the @x values at the masked positions.
    x[0, 63:] = 0

    y = t(x, key_padding_mask=key_padding_mask)

    prototype = y[0, 0]
    assert torch.all(prototype == y[0, :63])
    assert torch.all(prototype != y[0, 63:])

    prototype = y[1, 0]
    assert torch.all(prototype == y[1, :])
