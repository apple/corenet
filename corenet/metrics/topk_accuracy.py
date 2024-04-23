#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Any, Dict, Optional, Union

from torch import Tensor

from corenet.metrics import METRICS_REGISTRY
from corenet.metrics.metric_base import AverageMetric
from corenet.utils import logger


def top_k_accuracy(
    output: Tensor, target: Tensor, top_k: Optional[tuple] = (1,)
) -> list:
    maximum_k = max(top_k)
    batch_size = target.shape[0]

    _, pred = output.topk(maximum_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    results = []
    for k in top_k:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc_k = correct_k.mul_(100.0 / batch_size)
        results.append(acc_k)
    return results


class TopKMetric(AverageMetric):
    K = 1

    def gather_metrics(
        self,
        prediction: Union[Tensor, Dict],
        target: Union[Tensor, Dict],
        extras: Dict[str, Any],
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        This function gather top-k metrics from different processes and converts to float.
        """
        # We have four combinations between prediction and target types:
        # 1. (Tensor, Tensor)
        # 2. (Dict, Tensor)
        # 3. (Dict, Dict)
        # 4. (Tensor, Dict) --> This combination is rare

        if isinstance(prediction, Tensor) and isinstance(target, Tensor):
            (top_k_acc,) = top_k_accuracy(prediction, target, top_k=(self.K,))
            return top_k_acc
        elif isinstance(prediction, Dict) and isinstance(target, Tensor):
            top_k_dict = {}
            for pred_k, pred_v in prediction.items():
                if (
                    isinstance(pred_v, Tensor)
                    and pred_v.dim() == 2
                    and target.dim() == 1
                ):  # Output tensor should be of size [batch_size, num_classes] and target should be of shape [batch_size]
                    (top_1_acc,) = top_k_accuracy(pred_v, target, top_k=(self.K,))
                    top_k_dict[pred_k] = top_1_acc
            return top_k_dict
        elif isinstance(prediction, Dict) and isinstance(target, Dict):
            # prediction and target dictionaries should have intersecting keys
            prediction_keys = prediction.keys()
            target_keys = target.keys()

            intersection_keys = list(set(prediction_keys).intersection(target_keys))
            if len(intersection_keys) == 0:
                logger.error(
                    "The keys in prediction and target are different. "
                    " Got: Prediction keys={} and Target keys={}".format(
                        prediction_keys, target_keys
                    )
                )

            top_k_dict = {}
            for pred_k in intersection_keys:
                pred_v = prediction[pred_k]
                target_v = target[pred_k]
                if (
                    isinstance(pred_v, Tensor)
                    and isinstance(target_v, Tensor)
                    and pred_v.dim() == 2
                    and target_v.dim() == 1
                ):  # Output tensor should be of size [batch_size, num_classes] and target should be of shape [batch_size]
                    (top_1_acc,) = top_k_accuracy(pred_v, target_v, top_k=(self.K,))
                    top_k_dict[pred_k] = top_1_acc
            return top_k_dict
        elif isinstance(prediction, Tensor) and isinstance(target, Dict):
            # rare but possible
            top_k_dict = {}
            for target_k, target_v in target.items():
                if (
                    isinstance(target_v, Tensor)
                    and prediction.dim() == 2
                    and target_v.dim() == 1
                ):  # Output tensor should be of size [batch_size, num_classes] and target should be of shape [batch_size]
                    (top_1_acc,) = top_k_accuracy(prediction, target_v, top_k=(self.K,))
                    top_k_dict[target_k] = top_1_acc
            return top_k_dict
        else:
            logger.error("Metric monitor supports Tensor or Dict of Tensors")


@METRICS_REGISTRY.register(name="top1")
class Top1Metric(TopKMetric):
    K = 1


@METRICS_REGISTRY.register(name="top5")
class Top5Metric(TopKMetric):
    K = 5
