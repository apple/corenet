#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import traceback
from numbers import Number
from typing import Dict, Union

import numpy as np
from sklearn.metrics import average_precision_score
from torch import Tensor
from torch.nn import functional as F

from corenet.metrics import METRICS_REGISTRY
from corenet.metrics.metric_base import EpochMetric
from corenet.utils import logger


@METRICS_REGISTRY.register("average_precision")
class AveragePrecisionMetric(EpochMetric):
    def compute_with_aggregates(
        self, y_pred: Tensor, y_true: Tensor
    ) -> Union[Number, Dict[str, Number]]:
        y_pred, y_true = self.get_aggregates()

        y_pred = F.softmax(y_pred, dim=-1).numpy().astype(np.float32)
        y_true = y_true.numpy().astype(np.float32)

        # Clip predictions to reduce chance of getting INF
        y_pred = y_pred.clip(0, 1)

        if y_pred.ndim == 1 or y_pred.ndim == 2 and y_pred.shape[1] == 1:
            pass  # TODO?
        elif y_pred.ndim == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
        else:
            logger.warning(
                "Expected only two classes, got prediction Tensor of shape {}".format(
                    y_pred.shape
                )
            )

        try:
            ap = 100 * average_precision_score(y_true, y_pred, average=None)
        except ValueError as e:
            logger.warning("Could not compute Average Precision: {}".format(str(e)))
            traceback.print_exc()
            ap = 0  # we don't want the job to fail over a metric computation issue

        return ap
