#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import copy
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.nn import functional as F

from corenet.metrics import METRICS_REGISTRY
from corenet.metrics.metric_base import EpochMetric
from corenet.utils import logger
from corenet.utils.ddp_utils import is_master
from corenet.utils.registry import Registry

DISTANCE_REGISTRY = Registry("distance_metrics")


@DISTANCE_REGISTRY.register("cosine")
def cosine_distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Get pair-wise cosine distances.

    Args:
        x: A feature tensor with shape (n, d).
        y: A feature tensor with shape (m, d).

    Returns: Distance tensor between features x and y with shape (n, m).

    """
    assert len(x.shape) == len(y.shape) == 2
    assert x.shape[1] == y.shape[1]

    cosine_sim = F.cosine_similarity(x.unsqueeze(-1), y.T.unsqueeze(0), dim=1)
    assert cosine_sim.shape[0] == x.shape[0]
    assert cosine_sim.shape[1] == y.shape[0]
    assert len(cosine_sim.shape) == 2

    return 1 - cosine_sim


@DISTANCE_REGISTRY.register("l2")
def l2_distance_matrix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Get pair-wise l2 distances.

    Args:
        x: A torch feature tensor with shape (n, d).
        y: A torch feature tensor with shape (m, d).

    Returns: Distance tensor between features x and y with shape (n, m).

    """
    assert len(x.shape) == len(y.shape) == 2
    assert x.shape[1] == y.shape[1]

    return torch.cdist(x, y, p=2)


@METRICS_REGISTRY.register(name="retrieval_cmc")
class RetrievalCMC(EpochMetric):
    """
    Compute CMC-top-k and mAP metrics in retrieval setup.
    """

    def __init__(
        self,
        opts: argparse.Namespace = None,
        is_distributed: bool = False,
        pred: str = "embedding",
        target: str = None,
        compute_map: bool = True,
    ) -> None:
        super().__init__(opts, is_distributed, pred, target)
        distance_metric = getattr(opts, "stats.metrics.retrieval_cmc.distance_metric")
        self.k = getattr(opts, "stats.metrics.retrieval_cmc.k")
        self.subset_fraction = float(
            getattr(opts, "stats.metrics.retrieval_cmc.subset_fraction")
        )

        self.compute_map = compute_map

        self.get_distance_matrix = DISTANCE_REGISTRY[distance_metric]
        self.embedding = []
        self.label = []
        self.is_master = is_master(opts)
        if self.subset_fraction > 1.0:
            logger.error(
                "Subset fraction should be a positive number smaller than 1.0."
                f" Got {self.subset_fraction}"
            )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add metric specific arguments"""
        if cls == RetrievalCMC:
            parser.add_argument(
                "--stats.metrics.retrieval-cmc.subset-fraction",
                type=float,
                default=1.0,
                help="Use fraction of gallery set for CMC calculation when set."
                " Defaults to 1.0",
            )
            parser.add_argument(
                "--stats.metrics.retrieval-cmc.k",
                type=int,
                default=5,
                help="CMC top-k: percentage of query images with at least one same-class"
                " gallery image in their k-NN. Defaults to 5.",
            )
            parser.add_argument(
                "--stats.metrics.retrieval-cmc.distance-metric",
                type=str,
                default="l2",
                choices=["l2", "cosine"],
                help="Distance to use for nearest-neighbor calculation."
                " Defaults to l2",
            )

        return parser

    def compute_with_aggregates(
        self, embedding: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, float]:
        """Compute retrieval metrics over full epoch.

        Args:
            embedding: tensor of m embeddings with shape (m, d), where d is embedding dimension.
            labels: tensor of m labels.

        Returns: A dictionary of `top1`, `top-{k}` and `mAP`.

        """
        # (Possibly) use a smaller subset
        if self.subset_fraction < 1.0:
            gallery_size = embedding.shape[0]
            n_subset = int(self.subset_fraction * gallery_size)
            mask = torch.randperm(embedding.shape[0])[:n_subset]
            embedding = embedding[mask]
            labels = labels[mask]

        # Same embeddings are used for both gallery and query
        distance_matrix = self.get_distance_matrix(embedding, embedding)

        if self.is_master:
            logger.log(
                f"Begin CMC calculation on embeddings with shape = {embedding.shape}."
            )

        top1, topk = cmc_calculation(
            distance_matrix=distance_matrix,
            query_ids=labels,
            k=self.k,
        )
        top1 = float(top1)
        topk = float(topk)

        if self.compute_map:
            retrieval_map = mean_ap(distance_matrix=distance_matrix, labels=labels)
        else:
            retrieval_map = 0

        # Convert to percent and return
        return {
            "top1": 100 * top1,
            f"top{self.k}": 100 * topk,
            "mAP": 100 * retrieval_map,
        }


def cmc_calculation(
    distance_matrix: torch.Tensor,
    query_ids: torch.Tensor,
    k: int = 5,
) -> Tuple[float, float]:
    """Compute Cumulative Matching Characteristics metric.

    Args:
        distance_matrix: pairwise distance matrix between embeddings of gallery and query sets
        query_ids: labels for the query data (assuming the same as gallery)
        k: parameter for top k retrieval

    Returns: cmc-top1, cmc-top5

    """
    distance_matrix = copy.deepcopy(distance_matrix)
    query_ids = copy.deepcopy(query_ids)

    distance_matrix.fill_diagonal_(float("inf"))
    _, indices = torch.sort(distance_matrix)
    labels = query_ids.unsqueeze(dim=0).repeat(query_ids.shape[0], 1)
    sorted_labels = torch.gather(labels, 1, indices)
    top_1 = (sorted_labels[:, 0] == query_ids).sum() / query_ids.shape[0]
    top_k = (sorted_labels[:, :k] == query_ids.unsqueeze(1)).sum(dim=1).clamp(
        max=1
    ).sum() / query_ids.shape[0]

    return top_1, top_k


def mean_ap(
    distance_matrix: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute Mean Average Precision.

    Args:
        distance_matrix: pairwise distance matrix between embeddings of gallery and query sets, shape = (m,m)
        labels: labels for the query data (assuming the same as gallery), shape = (m,)

    Returns: mean average precision (float)

    """
    m, n = distance_matrix.shape
    assert m == n

    # Sort and find correct matches
    distance_matrix, gallery_matched_indices = torch.sort(distance_matrix, dim=1)
    truth_mask = labels[gallery_matched_indices] == labels[:, None]

    distance_matrix = distance_matrix.cpu().numpy()
    gallery_matched_indices = gallery_matched_indices.cpu().numpy()
    truth_mask = truth_mask.cpu().numpy()

    # Compute average precision for each query
    average_precisions = list()
    for query_index in range(n):
        valid_sorted_match_indices = (
            gallery_matched_indices[query_index, :] != query_index
        )
        y_true = truth_mask[query_index, valid_sorted_match_indices]
        y_score = -distance_matrix[query_index][valid_sorted_match_indices]
        if not np.any(y_true):
            continue  # if a query does not have any match, we exclude it from mAP calculation.
        average_precisions.append(average_precision_score(y_true, y_score))
    return np.mean(average_precisions)
