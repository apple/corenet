#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse

import pytest
import torch

from corenet.loss_fn.language_modeling.cross_entropy import CrossEntropyLM


@pytest.mark.parametrize(
    "label_smoothing, ignore_index,vocab_size,z_loss",
    [
        (0.0, -1, 5, False),
        (0.0, 3, 5, False),
        (0.0, -1, 5, True),
        (0.0, 2, 5, True),
        (0.1, 2, 5, True),
    ],
)
def test_cross_entropy_lm_in_out(
    label_smoothing: float, ignore_index: int, vocab_size: int, z_loss: bool
) -> None:
    """Test for CrossEntropyLM loss function.

    ...note:
        This test checks if the input and output formats are correct or not.
    """
    batch_size = 2
    seq_length = 5

    # build configuration
    parser = argparse.ArgumentParser()
    parser = CrossEntropyLM.add_arguments(parser)

    opts = parser.parse_args([])

    setattr(opts, "loss.language_modeling.cross_entropy.ignore_index", ignore_index)
    setattr(
        opts, "loss.language_modeling.cross_entropy.label_smoothing", label_smoothing
    )
    setattr(opts, "loss.language_modeling.cross_entropy.use_z_loss", z_loss)

    criteria = CrossEntropyLM(opts)

    # Two prediction cases:
    # Case 1: Prediction is a tensor
    # Case 2: Prediction is a dictionary, with logits as a mandatory key
    pred_case_1 = torch.randn(size=(batch_size, seq_length, vocab_size))
    pred_case_2 = {"logits": pred_case_1}

    target = torch.randint(
        low=0,
        high=vocab_size,
        size=(
            batch_size,
            seq_length,
        ),
    )
    # randomly set indices in target tensor to ignore_index
    random_indices = (torch.rand_like(target.float()) > 0.5) * 1.0
    random_indices = random_indices.to(dtype=torch.int)
    target[random_indices == 0] = ignore_index

    # CE loss function for LM does not care about input samples, so setting input_samples to None
    input_samples = None
    for pred in [pred_case_1, pred_case_2]:
        loss = criteria(input_samples, pred, target)
        if isinstance(loss, torch.Tensor):
            assert loss.dim() == 0, "Loss value should be a scalar"
        elif isinstance(loss, dict):
            # when z-loss is enabled, we return output as a dict (pred can be dict or tensor).
            assert z_loss
            expected_loss_keys = {"total_loss", "ce_loss", "z_loss"}
            assert any(set(loss.keys()) & expected_loss_keys)
            for loss_key, loss_val in loss.items():
                assert isinstance(
                    loss_val, torch.Tensor
                ), f"Loss should be an instance of torch.Tensor. Got {type(loss_val)} for key {loss_key}."
                assert loss_val.dim() == 0, "Loss value should be a scalar."
        else:
            raise NotImplementedError("Loss should be either a dictionary or a tensor.")
