#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from numbers import Number
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor

from corenet.metrics import METRICS_REGISTRY
from corenet.metrics.metric_base import BaseMetric
from corenet.utils.tensor_utils import reduce_tensor_sum


# TODO: tests
@METRICS_REGISTRY.register("confusion_matrix")
class ConfusionMatrix(BaseMetric):
    """
    Computes the confusion matrix and is based on `FCN <https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/score.py>`_
    """

    def reset(self):
        self.confusion_mat = None
        self.prediction_key = "logits"

    def update(
        self,
        prediction: Union[Tensor, Dict],
        target: Union[Tensor, Dict],
        extras: Dict[str, Any] = {},
        batch_size: Optional[int] = 1,
    ):
        if isinstance(prediction, dict) and self.prediction_key in prediction:
            prediction = prediction[self.prediction_key]

        if isinstance(prediction, dict) or isinstance(prediction, dict):
            raise NotImplementedError(
                "ConfusionMatrix does not currently support Dict predictions or targets"
            )

        n_classes = prediction.shape[1]

        if self.confusion_mat is None:
            self.confusion_mat = torch.zeros(
                (n_classes, n_classes), dtype=torch.int64, device=target.device
            )

        with torch.no_grad():
            prediction = prediction.argmax(1).flatten()
            target = target.flatten()
            k = (target >= 0) & (target < n_classes)
            inds = n_classes * target[k].to(torch.int64) + prediction[k]
            cnts = torch.bincount(inds, minlength=n_classes**2).reshape(
                n_classes, n_classes
            )
            if self.is_distributed:
                cnts = reduce_tensor_sum(cnts)
            self.confusion_mat += cnts

    def compute(self) -> Union[Number, Dict[str, Union[Number, List[Number]]]]:
        if self.confusion_mat is None:
            print("Confusion matrix is None. Check code")
            return None

        h = self.confusion_mat.float()

        metrics: Dict[str, Tensor] = {}
        metrics["accuracy_global"] = torch.diag(h).sum() / h.sum()
        diag_h = torch.diag(h)
        metrics["class_accuracy"] = diag_h / h.sum(1)
        metrics["mean_class_accuracy"] = metrics["class_accuracy"].mean()
        metrics["iou"] = diag_h / (h.sum(1) + h.sum(0) - diag_h)
        metrics["mean_iou"] = metrics["iou"].mean()
        metrics["confusion"] = self.confusion_mat

        # Making sure all values are converted to Python values
        metrics = {k: v.detach().cpu().tolist() for k, v in metrics.items()}
        return metrics
