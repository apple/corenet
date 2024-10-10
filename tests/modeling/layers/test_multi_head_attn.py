#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import sys

import numpy as np
import pytest
import torch

sys.path.append("..")

from corenet.modeling.layers.multi_head_attention import MultiHeadAttention


def build_attention_mask(context_length: int, batch_size: int, use_pytorch_mha):
    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    mask = torch.empty(context_length, context_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)  # zero out the lower diagonal
    if not use_pytorch_mha:
        mask = mask.unsqueeze(0)  # add dummy batch dimension
        mask = mask.expand(batch_size, -1, -1)
    return mask


@pytest.mark.parametrize(
    "output_dim, batch_size, bias, use_attn_mask",
    [
        (32, 1, True, True),
        (16, 2, False, False),
    ],
)
def test_multihead_self_attn(
    output_dim: int, batch_size: int, bias: bool, use_attn_mask: bool
):
    seq_len = 5
    embed_dim = 8
    mha = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=2,
        attn_dropout=0.0,
        bias=bias,
        output_dim=output_dim,
        coreml_compatible=False,
    )
    mha.eval()

    qkv = torch.randn(size=(seq_len, batch_size, embed_dim))

    attn_mask = None
    if use_attn_mask:
        attn_mask = build_attention_mask(
            context_length=seq_len, batch_size=batch_size, use_pytorch_mha=True
        )

    # Pytorch MHA accepts sequence first
    out_pytorch = mha(x_q=qkv, use_pytorch_mha=True, attn_mask=attn_mask)

    # default works with batch-first
    qkv = qkv.transpose(0, 1)
    attn_mask = None
    if use_attn_mask:
        attn_mask = build_attention_mask(
            context_length=seq_len, batch_size=batch_size, use_pytorch_mha=False
        )
    out_default = mha(x_q=qkv, use_pytorch_mha=False, attn_mask=attn_mask)
    out_default = out_default.transpose(0, 1)

    torch.testing.assert_close(
        actual=out_default, expected=out_pytorch, atol=1e-3, rtol=1e-3
    )

    if hasattr(mha, "forward_tracing") and attn_mask is None:
        # check coreml compatible version
        out_tracing = mha.forward_tracing(x_q=qkv).transpose(0, 1)
        torch.testing.assert_close(
            actual=out_default, expected=out_tracing, atol=1e-3, rtol=1e-3
        )


@pytest.mark.parametrize(
    "output_dim, key_len, batch_size, bias",
    [
        (32, 15, 1, True),
        (16, 20, 2, False),
    ],
)
def test_multihead_cross_attn(
    output_dim: int, key_len: int, batch_size: int, bias: bool
):
    seq_len = 20
    embed_dim = 32
    mha = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=8,
        attn_dropout=0.0,
        bias=bias,
        output_dim=output_dim,
        coreml_compatible=False,
    )
    mha.eval()

    query = torch.randn(size=(seq_len, batch_size, embed_dim))
    key = torch.randn(size=(key_len, batch_size, embed_dim))

    # Pytorch MHA accepts sequence first
    out_pytorch = mha(x_q=query, x_kv=key, use_pytorch_mha=True)

    # default works with batch-first
    query = query.transpose(0, 1)
    key = key.transpose(0, 1)
    out_default = mha(x_q=query, x_kv=key)
    out_default = out_default.transpose(0, 1)

    torch.testing.assert_close(
        actual=out_default, expected=out_pytorch, atol=1e-3, rtol=1e-3
    )

    if hasattr(mha, "forward_tracing"):
        # check coreml compatible version
        out_tracing = mha.forward_tracing(x_q=query, x_kv=key).transpose(0, 1)
        torch.testing.assert_close(
            actual=out_default, expected=out_tracing, atol=1e-3, rtol=1e-3
        )
