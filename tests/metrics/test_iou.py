#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Callable

import numpy as np
import torch

from corenet.metrics.stats import Statistics
from tests.metrics.base import transform_args


def test_gather_iou_metrics(transform_args: Callable):
    # [Batch, num_classes, height, width]
    # in this example, [1, 2, 2, 3]
    prediction = torch.tensor(
        [
            [
                [[0.2, 0.8, 0.2], [0.9, 0.2, 0.1]],
                [[0.8, 0.2, 0.8], [0.1, 0.8, 0.9]],  # spatial dms
            ]  # classes
        ]  # batch
    )

    target = torch.tensor([[[0, 0, 0], [0, 1, 1]]])

    metric_names, stats_args = transform_args(["iou"], prediction, target)

    expected_inter = np.array([2.0, 2.0])
    expected_union = np.array([4.0, 4.0])

    expected_iou = np.mean(expected_inter / expected_union) * 100

    stats = Statistics(opts=None, metric_names=metric_names)
    stats.update(*stats_args)

    np.testing.assert_equal(
        actual=stats.avg_statistics(metric_names[0]), desired=expected_iou
    )
