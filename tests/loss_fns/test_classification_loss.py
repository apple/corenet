#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse
from typing import Mapping

import pytest
import torch

from corenet.loss_fn.classification.binary_cross_entropy import BinaryCrossEntropy
from corenet.loss_fn.classification.cross_entropy import CrossEntropy


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1])
@pytest.mark.parametrize("ignore_index", [-1, 2])
@pytest.mark.parametrize("class_weights", [True, False])
@pytest.mark.parametrize("num_classes", [2, 5, 10])
def test_cross_entropy_in_out(
    batch_size: int,
    label_smoothing: float,
    ignore_index: int,
    class_weights: bool,
    num_classes: int,
) -> None:
    # These tests check the input and output formats are correct or not.

    # build configuration
    parser = argparse.ArgumentParser()
    parser = CrossEntropy.add_arguments(parser)

    opts = parser.parse_args([])

    setattr(opts, "loss.classification.cross_entropy.label_smoothing", label_smoothing)
    setattr(opts, "loss.classification.cross_entropy.ignore_index", ignore_index)
    setattr(opts, "loss.classification.cross_entropy.class_weights", class_weights)

    criteria = CrossEntropy(opts)

    # Two prediction cases:
    # Case 1: Prediction is a tensor
    # Case 2: Prediction is a dictionary, with logits as a mandatory key
    pred_case_1 = torch.randn(size=(batch_size, num_classes))
    pred_case_2 = {"logits": torch.randn(size=(batch_size, num_classes))}

    target = torch.randint(low=0, high=num_classes, size=(batch_size,))
    # this loss function does not care about it, so we can have any input
    input_sample = torch.randint(low=0, high=1, size=(1,))
    for pred in [pred_case_1, pred_case_2]:
        loss = criteria(input_sample, pred, target)
        assert isinstance(
            loss, torch.Tensor
        ), "Loss should be an instance of torch.Tensor"
        assert loss.dim() == 0, "Loss value should be a scalar"


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("reduction", ["sum", "mean", "none", "batch_mean"])
def test_binary_cross_entropy_in_out(batch_size: int, reduction: str) -> None:
    # These tests check the input and output formats are correct or not.

    # build configuration
    parser = argparse.ArgumentParser()
    parser = BinaryCrossEntropy.add_arguments(parser)

    opts = parser.parse_args([])

    setattr(opts, "loss.classification.binary_cross_entropy.reduction", reduction)

    criteria = BinaryCrossEntropy(opts)
    n_classes = 10

    # Two prediction cases:
    # Case 1: Prediction is a tensor
    # Case 2: Prediction is a dictionary, with logits as a mandatory key
    pred_case_1 = torch.randn(size=(batch_size, n_classes))
    pred_case_2 = {"logits": torch.randn(size=(batch_size, n_classes))}

    # two target cases:
    # Case 1: Target is a tensor containing soft-labels or probabilities
    # Case 2: Target is a tensor containing hard labels.
    target_case_1 = torch.randn(size=(batch_size, n_classes))
    target_case_2 = torch.randint(low=0, high=n_classes, size=(batch_size,))

    # this loss function does not care about it, so we can have any input
    input_sample = torch.randint(low=0, high=1, size=(1,))

    for pred in [pred_case_1, pred_case_2]:
        for target in [target_case_1, target_case_2]:
            loss = criteria(input_sample, pred, target)
            if isinstance(pred, Mapping):
                pred = pred["logits"]
            assert isinstance(
                loss, torch.Tensor
            ), "Loss should be an instance of torch.Tensor"

            if reduction in ["mean", "batch_mean", "sum"]:
                assert loss.dim() == 0, "Loss value should be a scalar"
            elif reduction == "none":
                assert (
                    loss.dim() == pred.dim()
                ), "Loss value should have the same shape as prediction"
