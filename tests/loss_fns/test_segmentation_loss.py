#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse

import pytest
import torch

from corenet.loss_fn.segmentation.cross_entropy import SegCrossEntropy


@pytest.mark.parametrize(
    "batch_size, label_smoothing, ignore_index, class_weights, num_classes, aux_weight",
    [
        (1, 0, -1, True, 2, 0),
        (2, 0.1, 255, False, 5, 0.4),
    ],
)
def test_seg_cross_entropy_in_out(
    batch_size: int,
    label_smoothing: float,
    ignore_index: int,
    class_weights: bool,
    num_classes: int,
    aux_weight: float,
) -> None:
    # These tests check the input and output formats are correct or not.

    # build configuration
    parser = argparse.ArgumentParser()
    parser = SegCrossEntropy.add_arguments(parser)

    opts = parser.parse_args([])

    setattr(opts, "loss.segmentation.cross_entropy.label_smoothing", label_smoothing)
    setattr(opts, "loss.segmentation.cross_entropy.ignore_index", ignore_index)
    setattr(opts, "loss.segmentation.cross_entropy.class_weights", class_weights)
    setattr(opts, "loss.segmentation.cross_entropy.aux_weight", aux_weight)

    criteria = SegCrossEntropy(opts)
    height = 10
    width = 10

    # Four prediction cases:
    # Case 1: Prediction is a tensor
    # Case 2: Prediction is a dictionary, with segmentation_output as a mandatory key
    # Case 3: Prediction is a Tuple[tensor, tensor]
    # Case 4: Prediction is a dictionary, with segmentation_output as a mandatory key and is a Tuple[Tensor, Tensor]
    pred_case_1 = torch.randn(size=(batch_size, num_classes, height, width))
    pred_case_2 = {"segmentation_output": pred_case_1}
    pred_case_3 = (
        torch.randn(size=(batch_size, num_classes, height, width)),
        torch.randn(size=(batch_size, num_classes, height, width)),
    )
    pred_case_4 = {"segmentation_output": pred_case_3}

    target = torch.randint(low=0, high=num_classes, size=(batch_size, height, width))
    # randomly set indices in target tensor to ignore_index
    random_indices = (torch.rand_like(target.float()) > 0.5) * 1.0
    random_indices = random_indices.to(dtype=torch.int)
    target[random_indices == 0] = ignore_index

    # this loss function does not care about it, so we can have any input
    input_sample = torch.randint(low=0, high=1, size=(1,))
    for pred in [pred_case_1, pred_case_2, pred_case_3, pred_case_4]:
        loss = criteria(input_sample, pred, target)
        assert isinstance(loss, dict), "loss should be an instance of dict"
        assert "total_loss" in loss, "total_loss is a mandatory key in loss"
        if len(loss) == 3:
            # when we compute aux loss, in that case, seg_loss and aux_loss are also returned
            assert {"total_loss", "seg_loss", "aux_loss"}.issubset(loss.keys())
        for loss_key, loss_val in loss.items():
            assert isinstance(
                loss_val, torch.Tensor
            ), "Loss should be an instance of torch.Tensor"
            assert loss_val.dim() == 0, "Loss value should be a scalar"
