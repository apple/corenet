#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
from typing import Callable

import numpy as np
import torch

from corenet.metrics.stats import Statistics
from tests.metrics.base import transform_args


def test_gather_psnr_metrics(transform_args: Callable):
    # Test for case 1
    inp_tensor = torch.randn((3, 2), dtype=torch.float)
    target_tensor = inp_tensor

    # Ideally, the PSNR should be infinite when input and target are the same, because error between
    # signal and noise is 0. However, we add a small eps value (error of 1e-10) in the computation
    # for numerical stability. Therefore, PSNR will not be infinite.
    expected_psnr = 10.0 * math.log10(255.0**2 / 1e-10)

    metric_names, stats_args = transform_args(["psnr"], inp_tensor, target_tensor)

    stats = Statistics(opts=None, metric_names=metric_names)
    stats.update(*stats_args)

    np.testing.assert_almost_equal(
        stats.avg_statistics(metric_names[0]), expected_psnr, decimal=2
    )
