#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import warnings
from typing import Optional, Set

import pytest
import torch
from torch import Tensor

from corenet.metrics import multiclass_classification_pr


def get_expected_keys(include_curve: bool) -> Set[str]:
    expected_keys = {
        "micro",
        "macro",
        "weighted",
        "AP",
        "ODS-F1",
        "Recall@P=50",
    }
    if include_curve:
        expected_keys |= {
            "precisions",
            "recalls",
            "thresholds",
        }
    return expected_keys


@pytest.mark.parametrize(
    "pred,target,include_curve,suppress_warnings",
    [
        ("pred", "target", False, False),
        ("pred", "target", False, True),
        ("pred", None, True, False),
        (None, "target", False, False),
        (None, None, True, False),
    ],
)
def test_metric(
    pred: Optional[str],
    target: Optional[str],
    include_curve: bool,
    suppress_warnings: bool,
) -> None:
    opts = argparse.Namespace()
    setattr(
        opts, "stats.metrics.multiclass_classification_pr.include_curve", include_curve
    )
    setattr(
        opts,
        "stats.metrics.multiclass_classification_pr.suppress_warnings",
        suppress_warnings,
    )
    metric = multiclass_classification_pr.MulticlassClassificationPR(
        opts, pred=pred, target=target
    )

    num_batches = 2
    batch_size = 6
    num_classes = 3

    for i in range(num_batches):
        predictions = torch.randn([batch_size, num_classes]).softmax(dim=-1)

        # @targets can be [batch_size, ...] or [batch_size, num_classes, ...].
        # Test both.
        if i % 2 == 0:
            # Class label targets.
            targets = torch.randint(0, num_classes, [batch_size])
        else:
            # Binary targets.
            targets = torch.randint(0, 1, [batch_size, num_classes])

        if pred is not None:
            predictions = {pred: predictions}
        if target is not None:
            targets = {target: targets}

        metric.update(predictions, targets)

    results = metric.compute()

    expected_keys = get_expected_keys(include_curve)
    assert set(results.keys()) == expected_keys

    for key in ["micro", "macro", "weighted"]:
        assert 0 <= results[key] <= 1

    if include_curve:
        for class_idx in range(num_classes):
            # Recalls and thresholds should be sorted.
            assert results["recalls"][class_idx] == list(
                reversed(sorted(results["recalls"][class_idx]))
            ), f"Recalls aren't sorted for class {class_idx} of {num_classes}."
            assert results["thresholds"][class_idx] == sorted(
                results["thresholds"][class_idx]
            ), f"Thresholds aren't sorted for class {class_idx} of {num_classes}."

            # All values should be in [0, 1].
            for key in ["precisions", "recalls", "thresholds"]:
                assert all(
                    0 <= elem <= 1 for elem in results[key][class_idx]
                ), f"Not all precisions/recalls/thresholds are in [0, 1] for class {class_idx} of {num_classes}."


@pytest.mark.parametrize(
    "include_curve,suppress_warnings", [(False, False), (True, False), (True, True)]
)
def test_flatten_metric(include_curve: bool, suppress_warnings: bool) -> None:
    opts = argparse.Namespace()
    setattr(
        opts, "stats.metrics.multiclass_classification_pr.include_curve", include_curve
    )
    setattr(
        opts,
        "stats.metrics.multiclass_classification_pr.suppress_warnings",
        suppress_warnings,
    )
    metric = multiclass_classification_pr.MulticlassClassificationPR(opts)

    num_batches = 2
    batch_size = 6
    num_classes = 3

    for i in range(num_batches):
        predictions = torch.randn([batch_size, num_classes]).softmax(dim=-1)
        targets = torch.randint(0, num_classes, [batch_size])
        metric.update(predictions, targets)

    results = metric.compute()
    my_name = "my_name"
    flattened_results = metric.flatten_metric(results, my_name)

    expected_keys = get_expected_keys(include_curve)
    assert set(flattened_results.keys()) == set(f"{my_name}/{k}" for k in expected_keys)
