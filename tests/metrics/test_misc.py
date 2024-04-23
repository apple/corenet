#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import numpy as np
import torch

from corenet.metrics.stats import Statistics


def test_gather_loss():
    # loss could be a Tensor or Dictionary
    loss = torch.tensor([3.2], dtype=torch.float)

    stats = Statistics(opts=None, metric_names=["loss"])
    stats.update({}, {}, {"loss": loss})
    np.testing.assert_almost_equal(stats.avg_statistics("loss"), 3.2)

    loss_dict = {
        "aux_loss": torch.tensor([4.2]),
        "total_loss": torch.tensor(
            [5.2]
        ),  # total loss key is mandatory for a loss in dict format
    }
    stats = Statistics(opts=None, metric_names=["loss"])
    stats.update({}, {}, {"loss": loss_dict})
    np.testing.assert_almost_equal(
        stats.avg_statistics("loss", "aux_loss"), 4.2, decimal=2
    )
    np.testing.assert_almost_equal(
        stats.avg_statistics("loss", "total_loss"), 5.2, decimal=2
    )

    # Empty extras
    stats = Statistics(opts=None, metric_names=["loss"])
    stats.update({}, {}, {})
    stats.update({}, {}, {"loss": None})

    metric = stats.avg_statistics("loss")
    np.testing.assert_almost_equal(metric, 0)


def test_gather_grad_norm():
    # Grad norm could be a Tensor or Dictionary
    grad_norm = torch.tensor([3.2], dtype=torch.float)

    stats = Statistics(opts=None, metric_names=["grad_norm"])
    stats.update({}, {}, {"grad_norm": grad_norm})
    np.testing.assert_almost_equal(stats.avg_statistics("grad_norm"), 3.2)

    grad_norm_dict = {
        "dummy_norm_a": torch.tensor([4.2]),
        "dummy_norm_b": torch.tensor([5.2]),
    }
    stats = Statistics(opts=None, metric_names=["grad_norm"])
    stats.update({}, {}, {"grad_norm": grad_norm_dict})

    _ = stats.avg_statistics("grad_norm")

    np.testing.assert_almost_equal(
        stats.avg_statistics("grad_norm", "dummy_norm_a"), 4.2, decimal=2
    )
    np.testing.assert_almost_equal(
        stats.avg_statistics("grad_norm", "dummy_norm_b"), 5.2, decimal=2
    )

    # Empty extras
    stats = Statistics(opts=None, metric_names=["grad_norm"])
    stats.update({}, {}, {})
    stats.update({}, {}, {"grad_norm": None})

    metric = stats.avg_statistics("grad_norm")
    np.testing.assert_almost_equal(metric, 0)
