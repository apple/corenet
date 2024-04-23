#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Callable

import numpy as np

from corenet.metrics.stats import Statistics
from tests.metrics.base import sample_classification_outputs, transform_args


def test_gather_top_k_metrics(transform_args: Callable):
    metric_names, stats_args = transform_args(
        ["top1", "top5"], *sample_classification_outputs()
    )

    stats = Statistics(opts=None, metric_names=metric_names)
    stats.update(*stats_args)
    top1_acc = round(stats.avg_statistics(metric_names[0]), 2)
    top5_acc = round(stats.avg_statistics(metric_names[1]), 2)

    np.testing.assert_almost_equal(top1_acc, 33.33, decimal=2)
    np.testing.assert_almost_equal(top5_acc, 100.00, decimal=2)
