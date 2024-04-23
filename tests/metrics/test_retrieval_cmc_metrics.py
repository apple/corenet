#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

import numpy as np
import pytest
import torch

from corenet.metrics.retrieval_cmc import (
    RetrievalCMC,
    cmc_calculation,
    cosine_distance_matrix,
    l2_distance_matrix,
    mean_ap,
)


def test_cosine_distance_matrix() -> None:
    # Sanity checks using random matrices
    n = 10
    m = 15
    d = 8
    x = torch.randn(n, d)
    y = torch.randn(m, d)
    dist_matrix = cosine_distance_matrix(x, y).numpy()
    assert dist_matrix.shape[0] == n
    assert dist_matrix.shape[1] == m
    assert np.all(0 <= dist_matrix)
    assert np.all(dist_matrix <= 2)

    # Numerical value check
    x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    y = torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    dist_matrix = cosine_distance_matrix(x, y).numpy()
    assert dist_matrix.shape[0] == 2
    assert dist_matrix.shape[1] == 2
    expected_similarity_dist = np.array([[0.0, 1.0], [2.0, 1.0]])
    np.testing.assert_almost_equal(dist_matrix, expected_similarity_dist)


def test_l2_distance_matrix() -> None:
    # Sanity checks using random matrices
    n = 3
    m = 7
    d = 4
    x = torch.randn(n, d)
    y = torch.randn(m, d)
    dist_matrix = l2_distance_matrix(x, y).numpy()
    assert dist_matrix.shape[0] == n
    assert dist_matrix.shape[1] == m
    assert np.all(0 <= dist_matrix)

    # Numerical value check
    x = torch.tensor(
        [
            [
                3.0,
                4.0,
                0.0,
            ]
        ]
    )
    y = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    dist_matrix = l2_distance_matrix(x, y).numpy()
    assert dist_matrix.shape[0] == 1
    assert dist_matrix.shape[1] == 2
    expected_similarity_dist = np.array([[4.0, 5.0]])
    np.testing.assert_almost_equal(dist_matrix, expected_similarity_dist)


def test_cmc_calculation() -> None:
    # Make sure input arrays are not modified
    distance_matrix = torch.zeros(10, 10)
    query_ids = torch.randint(0, 5, (10,))
    _, _ = cmc_calculation(distance_matrix, query_ids, 5)
    np.testing.assert_equal(distance_matrix.numpy(), 0)

    # Perfect match case
    distance_matrix = torch.tensor(
        [
            [0.0, 1.0, 0.5, 2.0],  # id = 100
            [1.0, 0.0, 2.0, 0.3],  # id = 101
            [0.5, 2.0, 0.0, 3.0],  # id = 100
            [2.0, 0.3, 3.0, 0.0],  # id = 101
        ]
    )
    query_ids = torch.tensor([100, 101, 100, 101])
    top1, top3 = cmc_calculation(distance_matrix, query_ids, 3)
    assert top1 == top3 == pytest.approx(1.0, 0.001)

    # Another case
    distance_matrix = torch.tensor(
        [
            [0.0, 1.0, 0.5, 1.75],  # id = 100
            [1.0, 0.0, 1.5, 0.75],  # id = 100
            [0.5, 1.5, 0.0, 2.75],  # id = 100
            [1.75, 0.75, 2.75, 0.0],  # id = 101
        ]
    )
    query_ids = torch.tensor([100, 100, 100, 101])
    top1, top2 = cmc_calculation(distance_matrix, query_ids, 2)
    assert top1 == pytest.approx(0.5, 0.001)
    assert top2 == pytest.approx(0.75, 0.001)


def test_mean_ap() -> None:
    # Numerical test
    distance_matrix = torch.tensor(
        [
            [0.0, 1.0, 0.5, 1.75],  # id = 100, PRs = [(1/1,1/2), (2/2, 2/2)], ap = 1.0
            [
                1.0,
                0.0,
                1.5,
                0.75,
            ],  # id = 100, PRs = [(0/1,0/2), (1/2, 1/2), (2/3, 2/2)], ap = 1/3+1/4 = 7/12
            [0.5, 1.5, 0.0, 2.75],  # id = 100, PRs = [(1/1,1/2), (2/2, 2/2)], ap = 1.0
            [
                1.75,
                0.75,
                2.75,
                0.0,
            ],  # id = 101, This query does not have any match -> excluded from mAP calculation.
        ]
    )
    query_ids = torch.tensor([100, 100, 100, 101])
    meanap = mean_ap(distance_matrix, query_ids)
    np.testing.assert_almost_equal(meanap, (1.0 + 7.0 / 12.0 + 1.0) / 3.0)


def test_retrieval_cmc() -> None:
    opts = argparse.Namespace()
    setattr(opts, "stats.metrics.retrieval_cmc.distance_metric", "l2")
    setattr(opts, "stats.metrics.retrieval_cmc.subset_fraction", 1.0)
    setattr(opts, "stats.metrics.retrieval_cmc.k", 2)
    cmc_eval = RetrievalCMC(
        # device=torch.device("cpu"),
        opts=opts,
        is_distributed=False,
        compute_map=True,
    )

    embeddings = torch.tensor(
        [
            [[0.0, 0.0], [0.0, 1.0]],
            [[0.0, -0.5], [0.0, 1.75]],
        ]
    )
    labels = torch.tensor([[100, 100], [100, 101]])

    cmc_eval.reset()
    for embedding, label in zip(embeddings, labels):
        cmc_eval.update(embedding, label)

    # outputs are reported in percentages
    cmc_metrics = cmc_eval.compute()

    np.testing.assert_almost_equal(cmc_metrics["top1"], 50)
    np.testing.assert_almost_equal(cmc_metrics["top2"], 75)
    np.testing.assert_almost_equal(
        cmc_metrics["mAP"], 100 * (1.0 + 7.0 / 12.0 + 1.0) / 3.0
    )
