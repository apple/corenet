#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import copy
import sys

sys.path.append("../..")

from corenet.loss_fn.composite_loss import CompositeLoss


def test_composite_loss() -> None:
    # This function tests the composite loss function
    # We create a list of three loss functions and then check the composition
    # of one, two, and all loss functions
    composite_losses = [
        {
            "loss_category": "classification",
            "loss_weight": 1.0,
            "classification": {
                "name": "cross_entropy",
                "cross_entropy": {
                    "label_smoothing": 0.1,
                    "ignore_index": -1,
                    "class_weights": False,
                },
            },
        },
        {
            "loss_category": "neural_augmentation",
            "loss_weight": 1.0,
            "neural_augmentation": {
                "perceptual_metric": "psnr",
                "target_value": [40, 20],
                "curriculum_method": "cosine",
                "alpha": 1.0,
            },
        },
        {
            "loss_category": "segmentation",
            "loss_weight": 1.0,
            "segmentation": {
                "name": "cross_entropy",
                "cross_entropy": {
                    "label_smoothing": 0.1,
                    "ignore_index": -1,
                    "class_weights": False,
                    "aux_weight": 0.0,
                },
            },
        },
    ]

    for i in range(1, 4):
        # test for 1, 2, and 3 loss functions
        parser = argparse.ArgumentParser()
        opts = parser.parse_args([])
        setattr(opts, "loss.category", "composite_loss")
        setattr(opts, "scheduler.max_epochs", 100)
        setattr(opts, "loss.composite_loss", copy.deepcopy(composite_losses[:i]))

        compsite_loss_fn = CompositeLoss(opts)
        assert len(compsite_loss_fn.loss_fns) == i
        assert len(compsite_loss_fn.loss_weights) == i
