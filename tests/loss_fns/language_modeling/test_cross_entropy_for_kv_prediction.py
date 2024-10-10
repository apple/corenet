#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse

import pytest
import torch

from corenet.loss_fn.language_modeling.cross_entropy_for_kv_prediction import (
    CrossEntropyForKVPrediction,
)


@pytest.mark.parametrize(
    "label_smoothing,ignore_index,vocab_size,z_loss,auxiliary_loss,kv_loss",
    [
        (0.0, 3, 5, False, 1, 2),
        (0.1, 2, 5, True, 1, 1),
    ],
)
def test_cross_entropy_lm_in_out(
    label_smoothing: float,
    ignore_index: int,
    vocab_size: int,
    z_loss: bool,
    auxiliary_loss: float,
    kv_loss: float,
) -> None:
    """Test for CrossEntropyLM loss function.

    ...note:
        This test checks if the input and output formats are correct or not.
    """
    batch_size = 2
    seq_length = 5
    key_heads = 2
    head_dim = 3
    num_layers = 2

    # build configuration
    parser = argparse.ArgumentParser()
    parser = CrossEntropyForKVPrediction.add_arguments(parser)

    opts = parser.parse_args([])

    setattr(
        opts,
        "loss.language_modeling.cross_entropy_for_kv_prediction.ignore_index",
        ignore_index,
    )
    setattr(
        opts,
        "loss.language_modeling.cross_entropy_for_kv_prediction.label_smoothing",
        label_smoothing,
    )
    setattr(
        opts,
        "loss.language_modeling.cross_entropy_for_kv_prediction.use_z_loss",
        z_loss,
    )
    setattr(
        opts, "loss.language_modeling.cross_entropy_for_kv_prediction.z_loss_eps", 0.01
    )
    setattr(
        opts,
        "loss.language_modeling.cross_entropy_for_kv_prediction.auxiliary_loss",
        auxiliary_loss,
    )
    setattr(opts, "loss.language_modeling.cross_entropy_for_kv_prediction.base_loss", 1)
    setattr(
        opts, "loss.language_modeling.cross_entropy_for_kv_prediction.kv_loss", kv_loss
    )

    criteria = CrossEntropyForKVPrediction(opts)

    # Two prediction cases:
    # Case 1: Prediction is a tensor
    # Case 2: Prediction is a dictionary, with logits as a mandatory key
    x = torch.randn(size=(batch_size, seq_length, vocab_size))
    model_outputs = {
        "logits": x,
        "auxiliary_logits": x,
        "past_keys": torch.randn(
            [num_layers, batch_size, key_heads, seq_length, head_dim]
        ),
        "past_values": torch.randn(
            [num_layers, batch_size, key_heads, seq_length, head_dim]
        ),
        "base_past_keys": torch.randn(
            [num_layers, batch_size, key_heads, seq_length, head_dim]
        ),
        "base_past_values": torch.randn(
            [num_layers, batch_size, key_heads, seq_length, head_dim]
        ),
    }

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
    for pred in [model_outputs]:
        loss = criteria(input_samples, pred, target)
        # when z-loss is enabled, we return output as a dict (pred can be dict or tensor).
        required_keys = {
            "auxiliary_loss",
            "base_loss",
            "k_loss/0",
            "k_loss/1",
            "k_loss/average",
            "k_loss/total",
            "total_loss",
            "v_loss/0",
            "v_loss/1",
            "v_loss/average",
            "v_loss/total",
        }
        if z_loss:
            required_keys.update({"z_loss_auxiliary", "z_loss_base"})
        assert set(loss.keys()) == (required_keys)
        for loss_key, loss_val in loss.items():
            assert isinstance(
                loss_val, torch.Tensor
            ), f"Loss should be an instance of torch.Tensor. Got {type(loss_val)} for key {loss_key}."
            assert loss_val.dim() == 0, "Loss value should be a scalar."
