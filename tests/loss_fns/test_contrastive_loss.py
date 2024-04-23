#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse

import pytest
import torch

from corenet.loss_fn.multi_modal_img_text.contrastive_loss_clip import (
    ContrastiveLossClip,
)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("projection_dim", [256, 512])
def test_contrastive_loss_in_out(batch_size: int, projection_dim: int) -> None:
    # These tests check the input and output formats are correct or not.
    parser = argparse.ArgumentParser()
    parser = ContrastiveLossClip.add_arguments(parser)

    opts = parser.parse_args([])
    criteria = ContrastiveLossClip(opts)

    image_features = torch.randn(size=(batch_size, projection_dim))
    text_features = torch.randn(size=(batch_size, projection_dim))

    input_sample = None
    targets = None

    prediction = {"image": image_features, "text": text_features}

    loss_output = criteria(input_sample, prediction, targets)
    expected_output_keys = {"total_loss", "image_loss", "text_loss", "logit_scale"}
    assert expected_output_keys.issubset(loss_output.keys())

    for loss_name, loss_val in loss_output.items():
        if loss_name == "logit_scale" and isinstance(loss_val, (float, int)):
            loss_val = torch.tensor(loss_val)
        assert isinstance(
            loss_val, torch.Tensor
        ), "Loss should be an instance of torch.Tensor"
        assert loss_val.dim() == 0, "Loss value should be a scalar"
