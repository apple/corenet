#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from collections import OrderedDict
from typing import List

import pytest
import torch
import torch.nn as nn

from corenet.modeling import get_model
from corenet.modeling.misc.common import freeze_modules_based_on_opts, get_tensor_sizes
from tests.configs import get_config
from tests.test_utils import unset_pretrained_models_from_opts


def test_freeze_modules_based_on_opts() -> None:
    model = nn.Sequential(
        OrderedDict(
            [
                ("conv1", nn.Conv2d(1, 20, 5)),
                ("relu1", nn.ReLU()),
                ("conv2", nn.Conv2d(20, 64, 5)),
                ("relu2", nn.ReLU()),
            ]
        )
    )
    opts = argparse.Namespace(**{"model.freeze_modules": "conv1"})
    freeze_modules_based_on_opts(opts, model)

    model.train()
    assert model.conv1.training == False
    assert model.conv2.training == True
    assert model.relu1.training == True


@pytest.mark.parametrize(
    ("config_file", "expected_trainable_params"),
    [
        (
            "tests/misc/dummy_clip_config.yaml",
            [
                "logit_scale",
                "image_encoder.classifier.proj",
                "text_encoder.projection_layer",
                "text_encoder.embedding_layer.weight",
                "text_encoder.positional_embedding.pos_embed.pos_embed",
                "text_encoder.transformer.0.pre_norm_mha.0.weight",
                "text_encoder.transformer.0.pre_norm_mha.0.bias",
                "text_encoder.transformer.0.pre_norm_mha.1.qkv_proj.weight",
                "text_encoder.transformer.0.pre_norm_mha.1.qkv_proj.bias",
                "text_encoder.transformer.0.pre_norm_mha.1.out_proj.weight",
                "text_encoder.transformer.0.pre_norm_mha.1.out_proj.bias",
                "text_encoder.transformer.0.pre_norm_ffn.0.weight",
                "text_encoder.transformer.0.pre_norm_ffn.0.bias",
                "text_encoder.transformer.0.pre_norm_ffn.1.weight",
                "text_encoder.transformer.0.pre_norm_ffn.1.bias",
                "text_encoder.transformer.0.pre_norm_ffn.4.weight",
                "text_encoder.transformer.0.pre_norm_ffn.4.bias",
                "text_encoder.final_layer_norm.weight",
                "text_encoder.final_layer_norm.bias",
            ],
        ),
        (
            "tests/misc/dummy_linear_probe_config.yaml",
            ["classifier.weight", "classifier.bias"],
        ),
    ],
)
def test_freeze_modules_based_on_opts_with_match_named_params(
    config_file: str, expected_trainable_params: List[str]
) -> None:
    """
    Test to check whether parameters are frozen correctly or not for models with complex structures (e.g., CLIP).
    """
    print(config_file)
    opts = get_config(config_file=config_file)

    # removing pretrained models (if any) to reduce test time as well as access issues.
    unset_pretrained_models_from_opts(opts)

    model = get_model(opts)
    model.train()

    total_model_parmams = sum([p.numel() for p in model.parameters()])
    model_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    assert model_trainable_params < total_model_parmams

    trainable_param_names = [
        p_name for p_name, p in model.named_parameters() if p.requires_grad
    ]
    assert trainable_param_names == expected_trainable_params


def test_get_tensor_sizes() -> None:
    in_width = 224
    in_height = 224
    in_channels = 3
    in_batch_size = 1
    img = torch.randn(size=(in_batch_size, in_channels, in_height, in_width))

    # test for Tensor
    size_info = get_tensor_sizes(img)
    assert size_info == [in_batch_size, in_channels, in_height, in_width]

    # test for empty dict
    data_dict = {}
    size_info = get_tensor_sizes(data_dict)
    assert size_info == []

    # test for dict with single key
    data_dict = {"image": img}
    size_info = get_tensor_sizes(data_dict)
    assert size_info == [
        str(f"image: [{in_batch_size}, {in_channels}, {in_height}, {in_width}]")
    ]

    # test for dict with two keys
    data_dict = {"image_1": img, "image_2": img}
    size_info = get_tensor_sizes(data_dict)
    assert size_info == [
        str(f"image_1: [{in_batch_size}, {in_channels}, {in_height}, {in_width}]"),
        str(f"image_2: [{in_batch_size}, {in_channels}, {in_height}, {in_width}]"),
    ]

    # test for nested dict
    data_dict = {"image_1": img, "image_2": {"image": img}}
    size_info = get_tensor_sizes(data_dict)
    assert size_info == [
        str(f"image_1: [{in_batch_size}, {in_channels}, {in_height}, {in_width}]"),
        str(
            f"image_2: ['image: [{in_batch_size}, {in_channels}, {in_height}, {in_width}]']"
        ),
    ]

    # test for nested dict with non-tensor
    data_dict = {"image": img, "random_key": "data"}
    size_info = get_tensor_sizes(data_dict)
    assert size_info == [
        str(f"image: [{in_batch_size}, {in_channels}, {in_height}, {in_width}]")
    ]
