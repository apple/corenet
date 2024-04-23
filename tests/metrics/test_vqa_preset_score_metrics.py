#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Test for metrics/vqa_score.py."""

import numpy as np
import torch

from corenet.metrics.stats import Statistics


def test_vqa_preset_score() -> None:
    predictions = {
        "logits": torch.tensor(
            [
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=torch.float,
        )
    }
    targets = torch.tensor(
        [
            [0, 0, 1],
            [0, 1, 0],
            [0, 0.5, 0.5],
        ],
        dtype=torch.float,
    )

    stats = Statistics(opts=None, metric_names=["vqa_preset_score"])
    stats.update(predictions, targets)
    score = round(stats.avg_statistics("vqa_preset_score", "bbox"), 2)

    np.testing.assert_almost_equal(score, 50.0)
