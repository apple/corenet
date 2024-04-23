#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Callable

import numpy as np

from corenet.metrics.stats import Statistics
from tests.configs import default_training_opts
from tests.metrics.base import sample_classification_outputs, transform_args


def test_probability_histogram(transform_args: Callable):
    metric_names, stats_args = transform_args(
        ["prob_hist"], *sample_classification_outputs()
    )

    stats = Statistics(opts=default_training_opts(), metric_names=metric_names)
    stats.update(*stats_args)

    # max values -> 0.91, 0.81, 0.51
    max_conf_hist = stats.avg_statistics(metric_names[0], "max")
    np.testing.assert_almost_equal(
        max_conf_hist,
        [0, 0, 0, 0, 0, 0.33, 0, 0, 0.33, 0.33],
        decimal=2,
    )

    # target values -> 0.05, 0.16, 0.51
    target_conf_hist = stats.avg_statistics(metric_names[0], "target")
    np.testing.assert_almost_equal(
        target_conf_hist,
        [0.33, 0.33, 0, 0, 0, 0.33, 0, 0, 0, 0],
        decimal=2,
    )
