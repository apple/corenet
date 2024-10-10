#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch import Tensor
from torch.nn import functional as F

from corenet.metrics import METRICS_REGISTRY
from corenet.metrics.metric_base import BaseMetric
from corenet.utils import logger, tensor_utils
from corenet.utils.file_logger import FileLogger


def get_recall_at_precision(
    precisions: np.ndarray,
    recalls: np.ndarray,
    precision_value: float,
    suppress_warnings: bool = False,
) -> float:
    """
    Compute the recall at the given @precision_value.

    Args:
      precisions: An array of shape [num_elements] with precision values.
      recalls: An array of shape [num_elements] with precision values.
      precision_value: The precision at which to obtain the recall.
      suppress_warnings: Suppress warnings.

    Returns: The recall at @precision_value.
    """
    sort_indices = np.argsort(precisions)
    sorted_precisions = precisions[sort_indices]
    sorted_recalls = recalls[sort_indices]

    index = np.searchsorted(sorted_precisions, precision_value)

    if not suppress_warnings and not np.isclose(
        sorted_precisions[index], precision_value, rtol=0.01, atol=0.01
    ):
        # The difference between the requested and true precisions are higher than expected.
        logger.warning(
            f"Found recall at precision {sorted_precisions[index]} "
            f"when recall at precision {precision_value} was requested."
        )

    return float(sorted_recalls[index])


def compute_oi_f1(predictions: Tensor, targets: Tensor) -> Tuple[float, float]:
    """
    Compute the "Optimal Instance" F1 score.

    The "official" computation corresponds to:
        1. Compute the threshold that maximizes individual input's F1 scores
            separately.
        2. Using those thresholds, count the true positives, false positives,
            and false negatives.
        3. Use these values to compute the F1 score.

    This function also computes a simple averaging of F1 scores individually
    calculated from each input's optimal thresholds.

    Args:
        predictions: A tensor of shape [batch_size, num_classes, predictions_per_sample]
            containing predictions.
        targets: A tensor of shape [batch_size, num_classes, predictions_per_sample]
            containing targets.

    Returns: A tuple containing the "official" F1 and the simple average of individual
        F1 scores, as described above.
    """
    if not predictions.ndim == 3:
        raise ValueError(f"Invalid shape {predictions.shape}")
    if not targets.ndim == 3:
        raise ValueError(f"Invalid shape {targets.shape}")

    batch_size = predictions.shape[0]
    num_classes = predictions.shape[1]

    official_ois_f1_scores = []
    avg_of_best_f1_scores = []
    for class_id in range(num_classes):
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        best_f1_scores = []

        for idx in range(batch_size):
            # 1. Find the threshold that maximizes F1 score.
            prediction = predictions[idx, class_id]
            target = targets[idx, class_id]
            precisions, recalls, thresholds = precision_recall_curve(target, prediction)
            f1_scores = (
                2
                * precisions
                * recalls
                / (precisions + recalls + ((precisions + recalls) == 0))
            )
            max_idx = np.argmax(f1_scores)
            threshold = thresholds[max_idx]
            best_f1_scores.append(f1_scores[max_idx])

            # 2. Use the threshold to update counts.
            true_positives += ((prediction >= threshold) * (target == 1)).sum().item()
            false_positives += ((prediction >= threshold) * (target == 0)).sum().item()
            false_negatives += ((prediction < threshold) * (target == 1)).sum().item()

        # Store the F1 score for this class.
        precision = true_positives / (
            true_positives + false_positives + ((true_positives + false_positives) == 0)
        )
        recall = true_positives / (
            true_positives + false_negatives + ((true_positives + false_negatives) == 0)
        )
        official_ois_f1_scores.append(
            float(
                (2 * precision * recall)
                / (precision + recall + ((precision + recall) == 0))
            )
        )
        avg_of_best_f1_scores.append(sum(best_f1_scores) / len(best_f1_scores))
    return official_ois_f1_scores, avg_of_best_f1_scores


@METRICS_REGISTRY.register(name="multiclass_classification_pr")
class MulticlassClassificationPR(BaseMetric):
    """
    Computes multiclass precision/recall metrics.

    Example .yaml configuration to use this metric (assuming your
    model outputs a dict with key "logits"):

    stats:
        val: ["multiclass_classification_pr(pred=logits)"]
        checkpoint_metric: "multiclass_classification_pr(pred=logits).macro"
        checkpoint_metric_max: true
        metrics:
            multiclass_classification_pr:
                include_curve: false
    """

    def __init__(
        self,
        opts: Optional[argparse.Namespace] = None,
        is_distributed: bool = False,
        pred: str = None,
        target: str = None,
    ) -> None:
        self.all_predictions: List[torch.Tensor] = []
        self.all_targets: List[torch.Tensor] = []
        self.include_curve = getattr(
            opts, "stats.metrics.multiclass_classification_pr.include_curve"
        )
        self.include_classwise_ap = getattr(
            opts, "stats.metrics.multiclass_classification_pr.include_classwise_ap"
        )
        self.suppress_warnings = getattr(
            opts, "stats.metrics.multiclass_classification_pr.suppress_warnings"
        )
        super().__init__(opts, is_distributed, pred, target)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add metric specific arguments.

        Args:
            parser: The parser to which to add the arguments.

        Returns:
            The parser.
        """
        if cls == MulticlassClassificationPR:
            parser.add_argument(
                "--stats.metrics.multiclass-classification-pr.include-curve",
                action="store_true",
                help="If set, PR curves will be stored.",
            )
            parser.add_argument(
                "--stats.metrics.multiclass-classification-pr.include-classwise-ap",
                action="store_true",
                help="If set, AP will be plotted for each class.",
            )
            parser.add_argument(
                "--stats.metrics.multiclass-classification-pr.suppress-warnings",
                action="store_true",
                help="If set, warnings will be suppressed. This is useful to reduce the logs size during training.",
            )
        return parser

    def reset(self) -> None:
        """
        Resets all aggregated data.
        Called at the start of every epoch.
        """
        self.all_predictions.clear()
        self.all_targets.clear()

    def update(
        self,
        prediction: Union[Tensor, Dict],
        target: Union[Tensor, Dict],
        extras: Dict[str, Any] = {},
        batch_size: Optional[int] = 1,
    ) -> None:
        """
        Processes a new batch of predictions and targets for computing the metric.

        Args:
            predictions: model outputs for the current batch. They must be a
                tensor of shape [batch_size, num_classes, ...], or a dictionary
                with key self.pred_key containing such a tensor.
            target: labels for the current batch. They may be a tensor of shape
                [batch_size, ...], or a dictionary with key self.target_key containing
                such a tensor. If so, the entries are assumed to be class indices.
                The target may also be a tensor of shape [batch_size, num_classes, ...],
                or a dictionary with key self.target_key containing such a tensor. If so,
                the entries are assumed to be binary class labels.
            extras: unused.
            batch_size: unused.
        """
        if isinstance(prediction, dict):
            if self.pred_key in prediction:
                prediction = prediction[self.pred_key]
            else:
                raise KeyError(
                    f"Missing prediction key '{self.pred_key}. Existing keys: {prediction.keys()}'"
                )

        if isinstance(target, dict):
            if self.target_key in target:
                target = target[self.target_key]
            else:
                raise KeyError(
                    f"Missing target key '{self.target_key}. Existing keys: {target.keys()}'"
                )

        if (prediction.ndim - target.ndim) not in (0, 1):
            raise ValueError(
                f"Invalid dimensions prediction.shape={prediction.shape}, target.shape={target.shape}"
            )

        if target.ndim < prediction.ndim:
            # The target doesn't have a num_classes dimension because it has
            # class labels. Expand it.
            num_classes = prediction.shape[1]
            target = F.one_hot(target, num_classes=num_classes)
            # Change from [batch_size, ..., num_classes] to [batch_size, num_classes, ...].
            new_order = (
                0,
                target.ndim - 1,
            ) + tuple(range(1, target.ndim - 1))
            target = target.permute(*new_order)

        # Now, @target and @prediction are both in [batch_size, num_classes, ...] order.
        assert target.shape == prediction.shape

        if prediction.dim() > 2:
            prediction = prediction.reshape(
                prediction.shape[0], prediction.shape[1], -1
            )
            target = target.reshape(target.shape[0], target.shape[1], -1)

        with torch.no_grad():
            if self.is_distributed:
                all_predictions = tensor_utils.all_gather_list(
                    prediction.detach().cpu().contiguous()
                )
                all_targets = tensor_utils.all_gather_list(
                    target.detach().cpu().contiguous()
                )
                all_predictions = torch.cat(
                    [p.detach().cpu() for p in all_predictions], dim=0
                )
                all_targets = torch.cat([t.detach().cpu() for t in all_targets], dim=0)
            else:
                all_predictions = prediction.detach().cpu()
                all_targets = target.detach().cpu()
            self.all_predictions.append(all_predictions)
            self.all_targets.append(all_targets)

    def compute(self) -> Dict[str, Union[Number, List[List[Number]]]]:
        """
        Compute the multiclass classification Precision-Recall metrics.

        See https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
        for details.

        Returns:
            A dictionary containing:
            {
                "micro": The "micro"-averaged precision, as defined by SKLearn.
                    This corresponds to treating each element in the multiclass
                    prediction separately.
                "macro": The "macro"-averaged precision, as defined by SKLearn.
                    This corresponds to calculating precision-recall metrics
                    separately for each label, then computing an unweighted mean.
                "weighted": The "weighted"-averaged precision, as defined by SKLearn.
                    This corresponds to calculating precision-recall metrics
                    separately for each label, then computing a weighted mean.
                "precisions": A list of lists, where element [i][j] is the
                    j'th precision value for the i'th class.
                "recalls": A list of lists, where element [i][j] is
                    the j'th precision value for the i'th class.
                "thresholds": A list of lists, where element [i][j] is
                    the j'th threshold value for the i'th class.
            }
        """
        if self.include_curve:
            metrics = {
                "precisions": [],
                "recalls": [],
                "thresholds": [],
                "ODS-F1": [],
                "AP": [],
                "Recall@P=50": [],
            }
        else:
            metrics = {
                "ODS-F1": [],
                "AP": [],
                "Recall@P=50": [],
            }

        predictions = (
            torch.cat(self.all_predictions, dim=0).float().numpy()
        )  # [batch_size, num_classes, ...]
        num_classes = predictions.shape[1]

        targets = (
            torch.cat(self.all_targets, dim=0).float().numpy()
        )  # [batch_size, num_classes, ...]

        if predictions.ndim == 3:
            assert targets.ndim == 3
            # @predictions and @targets have shape [batch_size, num_classes, predictions_per_element]. Compute
            # the optimal instance score (OIS-F), which is the only metric that needs the predictions_per_element
            # dimension. Then, reshape to [batch_size * predictions_per_element, num_classes].
            official, avg = compute_oi_f1(predictions, targets)
            metrics["OIS-F1-official"] = official
            metrics["OIS-F1-avg"] = avg
            predictions = predictions.transpose(0, 2, 1).reshape(-1, num_classes)
            targets = targets.transpose(0, 2, 1).reshape(-1, num_classes)

        for class_id in range(num_classes):
            (
                precisions,
                recalls,
                thresholds,
            ) = precision_recall_curve(targets[:, class_id], predictions[:, class_id])

            f1_scores = (2 * precisions * recalls) / (
                precisions + recalls + ((precisions + recalls) == 0)
            )
            metrics["ODS-F1"].append(f1_scores.max().item())

            if self.include_curve:
                metrics["precisions"].append(precisions.tolist())
                metrics["recalls"].append(recalls.tolist())
                metrics["thresholds"].append(thresholds.tolist())

            metrics["AP"].append(
                float(
                    average_precision_score(
                        targets[:, class_id], predictions[:, class_id]
                    )
                )
            )
            metrics["Recall@P=50"].append(
                get_recall_at_precision(
                    precisions, recalls, 0.5, suppress_warnings=self.suppress_warnings
                )
            )

        for average in ["micro", "macro", "weighted"]:
            metrics[average] = float(
                average_precision_score(targets, predictions, average=average)
            )

        if self.include_classwise_ap:
            for i, v in enumerate(metrics["AP"]):
                metrics[f"AP-class{i}"] = v

        return metrics

    def is_epoch_summary_enabled_for_metric(
        self, metric_name: str, log_writer: Any
    ) -> bool:
        """
        Determines whether to log a metric with the given @metric_name when the
        given @log_writer is invoked.

        This is mainly used to prevent logs from becoming too large. For
        example, we might not want to display every value in a PR curve, even
        though we want to calculate and store the curve.

        Args:
            metric_name: The name of the metric.
            log_writer: An object that can be used as a log writer (for example,
                a TensorBoardLogger).

        Returns:
            True if the name of the metric should be logged. False otherwise.
        """
        if isinstance(log_writer, FileLogger):
            # For FileLoggers, we log everything, including the rather large
            # precisions/thresholds/recalls keys.
            return True
        else:
            # For other loggers, we avoid the precisions/thresholds/recalls
            # keys.
            return not any(
                (
                    "precisions" in metric_name.lower(),
                    "thresholds" in metric_name.lower(),
                    "recalls" in metric_name.lower(),
                )
            )

    def flatten_metric(
        self, values: Union[Number, List, Dict[str, Any]], metric_name: str
    ) -> Dict[str, Union[Number, List, Dict[str, Any]]]:
        """
        Flatten the given metric @values, prepending @metric_name to the
        resulting dictionary's keys.

        Unlike the base class's method, we do not recursively flatten. This is
        because we have lists of PR curve values, and we don't want to generate
        an enormous number of keys to avoid inefficient storage.

        Args:
            values: The values, as output by @self.compute.
            metric_name: The metric name key prefix.

        Returns:
            A version of @values that has been flattened, with key names
                starting with @metric_name.
        """
        return {f"{metric_name}/{k}": v for k, v in values.items()}

    def summary_string(self, name: str, sep: str, values: Dict[str, Any]) -> str:
        """
        Get a string representation of the given metric values, suitable for
        printing to the terminal.

        We avoid printing precision/thresholds/recalls from PR curve
        computation, to avoid excessively long logs.

        Args:
            name: The name of the metric.
            sep: The separator used in the printout.
            values: The metric values, as output by @self.compute.

        Returns:
            A string representation of the metric.
        """
        filtered_keys = {"precisions", "thresholds", "recalls"}
        values = {k: v for k, v in values.items() if k not in filtered_keys}
        return super().summary_string(name, sep, values)
