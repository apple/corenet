#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import pytest
import torch

from corenet.metrics.stats import Statistics
from tests.configs import default_training_opts


@pytest.mark.parametrize(
    "batch_size, num_captions, hidden_dim, text_dim",
    [
        (1, 1, 8, 2),
        (2, 5, 4, 3),
    ],
)
def test_image_text_retrieval(
    batch_size: int, num_captions: int, hidden_dim: int, text_dim: int
) -> None:
    stats = Statistics(
        opts=default_training_opts(), metric_names=["image_text_retrieval"]
    )
    for _ in range(3):
        image_emb = torch.randn(batch_size, hidden_dim)
        text_emb = torch.randn(batch_size, num_captions, hidden_dim)
        if text_dim == 2:
            text_emb = text_emb.reshape(-1, hidden_dim)
        stats.update({"image": image_emb, "text": text_emb}, {}, {})

    metrics = stats._compute_avg_statistics_all()
    img_text_metrics = metrics["image_text_retrieval"]

    parent_keys = ["text2image", "image2text"]
    child_keys = ["recall@1", "recall@5", "recall@10", "mean_rank", "median_rank"]
    for parent_key in parent_keys:
        assert parent_key in img_text_metrics
        for child_key in child_keys:
            assert child_key in img_text_metrics[parent_key]
