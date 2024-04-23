#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Preset score metric for VQA evaluation."""

from typing import Any, Dict, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from corenet.metrics import METRICS_REGISTRY
from corenet.metrics.metric_base import AverageMetric


def vqa_preset_score_metric(output: Tensor, target: Tensor) -> list:
    """Sum the VQA preset scores for tasks with more than one ground-truth target.

    Score metric is an alternative to the strict 0-1 accuracy metric in multi-class
    classification that assigns non-zero score to partially correct answers. For
    example, in the Visual-Question-Answering task the question "How many cats are in
    this picture?" can have a correct answer '3' with score 1.0 and a partially correct
    answer '2' with score 0.5. All other predicted answers get score 0.0 .

    Arguments:
        output: Prediction probabilities to be scored (batch_size, n_clasess).
        target: Preset ground-truth score for predicted label per input
          (batch_size, n_classes).

    Returns:
        A tensor of a single float as the average of score on the batch.
    """
    batch_size = target.shape[0]
    output = F.softmax(output, dim=-1)
    _, pred = output.max(1, keepdim=True)  # (B, C)
    score = torch.gather(target, 1, pred).sum()
    score = score.mul_(100.0 / batch_size)
    return score


@METRICS_REGISTRY.register(name="vqa_preset_score")
class VQAPresetScoreMetric(AverageMetric):
    """Metric class for VQA preset score."""

    def gather_metrics(
        self,
        prediction: Union[Tensor, Dict],
        target: Union[Tensor, Dict],
        extras: Dict[str, Any],
    ) -> Tensor:
        """Gather VQA score metric for given set of predictions and targets.

        This function accepts two combinations between prediction and target types:
        1. (Tensor, Tensor)
        2. (Dict, Tensor)

        Args:
            prediction: a tensor or a dictionary with `logits` as the key.
            target: a tensor of the same shape as prediction.
            extras: a dictionary with extra arguments. Not used in this class.

        Returns:
            A tensor of a single float as the average of score on the batch.
        """
        if isinstance(prediction, Tensor) and isinstance(target, Tensor):
            score = vqa_preset_score_metric(prediction, target)
            return score
        elif isinstance(prediction, Dict) and isinstance(target, Tensor):
            pred_v = prediction["logits"]
            if isinstance(pred_v, Tensor) and pred_v.dim() == 2 and target.dim() == 2:
                score = vqa_preset_score_metric(pred_v, target)
            return score
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__} only takes (Tensor, Tensor) or"
                " (Dict, Tensor) as prediction and target."
            )
