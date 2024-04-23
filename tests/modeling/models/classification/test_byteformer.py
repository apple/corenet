#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

import torch

from corenet.modeling.models.classification import byteformer


def get_opts() -> argparse.Namespace:
    opts = argparse.Namespace()
    setattr(opts, "model.classification.byteformer.conv_kernel_size", 16)
    setattr(opts, "model.classification.byteformer.dropout", 0.0)
    setattr(opts, "model.classification.byteformer.norm_layer", "layer_norm")
    setattr(opts, "model.classification.byteformer.dummy_input_token_length", 1)
    setattr(opts, "model.classification.byteformer.max_num_tokens", 1000)
    setattr(opts, "model.classification.byteformer.sinusoidal_pos_emb", False)
    setattr(opts, "model.classification.byteformer.vocab_size", 257)
    setattr(opts, "model.classification.byteformer.mode", "tiny")
    setattr(opts, "model.classification.byteformer.input_dims", 2)
    setattr(opts, "model.classification.byteformer.pos_embed_type", "learnable")
    setattr(opts, "model.classification.byteformer.window_sizes", [128] * 12)
    setattr(opts, "model.classification.byteformer.window_shifts", [0, 64] * 6)
    setattr(
        opts,
        "model.classification.byteformer.downsample",
        [True, True] + ([False, True] * 4) + [False, False],
    )
    setattr(opts, "model.classification.byteformer.stochastic_dropout", 0)
    setattr(opts, "model.classification.n_classes", 1000)
    setattr(opts, "model.normalization.groups", None)
    setattr(opts, "model.normalization.momentum", 0.9)
    setattr(opts, "model.activation.name", "relu")
    setattr(opts, "model.activation.inplace", False)
    setattr(opts, "model.activation.neg_slope", False)
    return opts


def test_token_reduction_net() -> None:
    opts = get_opts()
    model = byteformer.ByteFormer(opts)

    B, N, C = 1, 256, 192
    x_values = torch.ones([B, N, C])
    mask = torch.ones([B, N])
    mask[0, 129:] = 0
    x_values[:, 129:] = 0
    y, y_mask = model.apply_token_reduction_net(x_values, mask)
    assert y.shape == (1, 31, 192)
    assert y_mask.shape == (1, 31)
    assert (y_mask > 0).float().sum() == 17


def test_model_forward_pass() -> None:
    opts = get_opts()
    model = byteformer.ByteFormer(opts)

    B, N = 1, 256
    x_values = torch.empty([B, N], dtype=torch.int)
    x_values[:, :129] = torch.randint(0, 256, size=(1, 129))
    x_values[:, 129:] = -1

    y = model(x_values)
    assert y.shape == (1, 1000)
