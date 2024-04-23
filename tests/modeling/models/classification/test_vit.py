#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import pytest
import torch

from corenet.modeling.models import get_model
from tests.configs import get_config


@pytest.mark.parametrize(
    "cls_token,return_image_embeddings,use_range_augment,use_flash_attn",
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, True, False, False),
        (True, True, True, False),
        (True, True, True, True),
    ],
)
def test_vit(cls_token, return_image_embeddings, use_range_augment, use_flash_attn):
    batch_size = 2
    image_channels = 3
    image_height = 32
    image_width = 32
    model_dim = 128
    n_classes = 3
    # input image is 32x32, so the number of patches along width and height with a stride of 16 will be 2 and 2 respectively.
    n_patches_h = n_patches_w = 2

    opts = get_config(
        config_file="tests/modeling/models/classification/config/vit_config.yaml"
    )

    setattr(opts, "model.classification.vit.no_cls_token", cls_token)
    setattr(opts, "model.classification.n_classes", n_classes)
    setattr(opts, "model.classification.vit.use_flash_attention", use_flash_attn)

    if use_range_augment:
        setattr(opts, "model.learn_augmentation.mode", "distribution")
        setattr(opts, "model.learn_augmentation.brightness", True)

    model = get_model(opts)

    input = torch.randn(
        (batch_size, image_channels, image_height, image_width), dtype=torch.float
    )
    out = model(input, return_image_embeddings)

    # Four cases
    # 1. Return image embeddings along with range augment
    # 2. Return image embeddings but do not use range augment
    # 3. Use range augment but do not return image embeddings
    # 4. Neither use range augment nor return image embeddings
    if return_image_embeddings and use_range_augment:
        expected_keys_and_size = {
            "logits": [batch_size, n_classes],
            "augmented_tensor": [batch_size, image_channels, image_height, image_width],
            "image_embeddings": [batch_size, model_dim, n_patches_h, n_patches_w],
        }
    elif return_image_embeddings and not use_range_augment:
        expected_keys_and_size = {
            "logits": [batch_size, n_classes],
            "augmented_tensor": None,
            "image_embeddings": [batch_size, model_dim, n_patches_h, n_patches_w],
        }
    elif not return_image_embeddings and use_range_augment:
        expected_keys_and_size = {
            "logits": [batch_size, n_classes],
            "augmented_tensor": [batch_size, image_channels, image_height, image_width],
        }
    else:
        expected_keys_and_size = {}

    if expected_keys_and_size:
        # check all output keys are present
        assert len(out.keys() & expected_keys_and_size.keys()) == len(
            expected_keys_and_size
        )

        # check the size of output is as expected
        for key_, size_ in expected_keys_and_size.items():
            if size_ is None:
                continue
            assert list(out[key_].size()) == size_
            assert torch.all(torch.isfinite(out[key_]))
    else:
        assert out.ndim == 2
        assert list(out.size()) == [batch_size, n_classes]
        assert torch.all(torch.isfinite(out))
