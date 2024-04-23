#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse
import random

import pytest
import torch

from corenet.loss_fn.detection.mask_rcnn_loss import MaskRCNNLoss
from corenet.loss_fn.detection.ssd_multibox_loss import SSDLoss


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("neg_pos_ratio", [1, 3])
@pytest.mark.parametrize("label_smooth", [0.0, 0.1])
def test_ssd_loss_in_out(
    batch_size: int, neg_pos_ratio: int, label_smooth: int
) -> None:
    # These tests check the input and output formats are correct or not.

    # build configuration
    parser = argparse.ArgumentParser()
    parser = SSDLoss.add_arguments(parser)

    opts = parser.parse_args([])

    setattr(opts, "loss.detection.ssd_multibox_loss.neg_pos_ratio", neg_pos_ratio)
    setattr(opts, "loss.detection.ssd_multibox_loss.label_smoothing", label_smooth)

    criteria = SSDLoss(opts)
    num_anchors = 20
    num_classes = 2

    # prediction is a dictionary with scores and boxes as a key
    predictions = {
        "scores": torch.rand(size=(batch_size, num_anchors, num_classes)),
        "boxes": torch.rand(size=(batch_size, num_anchors, 4)),
    }

    # target is a dictionary with box_labels and box_coordinates as a key
    targets = {
        "box_labels": torch.randint(
            low=0, high=num_classes, size=(batch_size, num_anchors)
        ),
        "box_coordinates": torch.rand(size=(batch_size, num_anchors, 4)),
    }
    # this loss function does not care about it, so we can have any input
    input_sample = torch.randint(low=0, high=1, size=(1,))

    loss = criteria(input_sample, predictions, targets)
    assert {"total_loss", "reg_loss", "cls_loss"}.issubset(loss.keys())
    for loss_name, loss_val in loss.items():
        assert isinstance(
            loss_val, torch.Tensor
        ), "Loss should be an instance of torch.Tensor"
        assert loss_val.dim() == 0, "Loss value should be a scalar"


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("classifier_weight", [0, 0.5, 1.0])
@pytest.mark.parametrize("box_reg_weight", [0, 0.5, 1.0])
@pytest.mark.parametrize("mask_weight", [0, 0.5, 1.0])
@pytest.mark.parametrize("objectness_weight", [0, 0.5, 1.0])
@pytest.mark.parametrize("rpn_box_reg", [0, 0.5, 1.0])
def test_maskrcnn_loss_in_out(
    batch_size: int,
    classifier_weight: float,
    box_reg_weight: float,
    mask_weight: float,
    objectness_weight: float,
    rpn_box_reg: float,
) -> None:
    # These tests check the input and output formats are correct or not.

    # build configuration
    parser = argparse.ArgumentParser()
    parser = MaskRCNNLoss.add_arguments(parser)

    opts = parser.parse_args([])

    setattr(opts, "loss.detection.mask_rcnn_loss.classifier_weight", classifier_weight)
    setattr(opts, "loss.detection.mask_rcnn_loss.box_reg_weight", box_reg_weight)
    setattr(opts, "loss.detection.mask_rcnn_loss.mask_weight", mask_weight)
    setattr(opts, "loss.detection.mask_rcnn_loss.objectness_weight", objectness_weight)
    setattr(opts, "loss.detection.mask_rcnn_loss.rpn_box_reg", rpn_box_reg)

    criteria = MaskRCNNLoss(opts)

    prediction_losses = {}
    for loss_key in [
        "loss_classifier",
        "loss_box_reg",
        "loss_mask",
        "loss_objectness",
        "loss_rpn_box_reg",
    ]:
        prediction_losses.update({loss_key: torch.tensor(random.random())})

    # this loss function does not care about it, so we can have any input
    input_sample = torch.randint(low=0, high=1, size=(1,))

    loss = criteria(input_sample, prediction_losses)
    assert {
        "loss_classifier",
        "loss_box_reg",
        "loss_mask",
        "loss_objectness",
        "loss_rpn_box_reg",
        "total_loss",
    }.issubset(loss.keys())
    for loss_name, loss_val in loss.items():
        assert isinstance(
            loss_val, torch.Tensor
        ), "Loss should be an instance of torch.Tensor"
        assert loss_val.dim() == 0, "Loss value should be a scalar"
