#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from numbers import Number
from typing import Dict, Union

import numpy as np
from torch import Tensor
from torch.nn import functional as F

from corenet.metrics import METRICS_REGISTRY
from corenet.metrics.metric_base import EpochMetric
from corenet.utils import logger


@METRICS_REGISTRY.register("prob_hist")
class ProbabilityHistogramMetric(EpochMetric):
    def __init__(
        self,
        opts: argparse.Namespace = None,
        is_distributed: bool = False,
        pred: str = None,
        target: str = None,
    ):
        super().__init__(opts, is_distributed, pred, target)
        self.num_bins = getattr(self.opts, "stats.metrics.prob_hist.num_bins")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add metric specific arguments"""
        if cls == ProbabilityHistogramMetric:
            parser.add_argument(
                "--stats.metrics.prob-hist.num-bins", type=int, default=10
            )
        return parser

    def compute_with_aggregates(
        self, y_pred: Tensor, y_true: Tensor
    ) -> Union[Number, Dict[str, Number]]:
        y_pred = F.softmax(y_pred, dim=-1).numpy()
        y_true = y_true.numpy()

        max_confs = y_pred.max(axis=-1)
        max_hist = np.histogram(max_confs, bins=self.num_bins, range=[0, 1])[0]
        max_hist = max_hist / max_hist.sum()

        target_confs = np.take_along_axis(y_pred, y_true.reshape(-1, 1), 1)
        target_hist = np.histogram(target_confs, bins=self.num_bins, range=[0, 1])[0]
        target_hist = target_hist / target_hist.sum()

        return {
            "max": max_hist.tolist(),
            "target": target_hist.tolist(),
        }
