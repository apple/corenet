#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from numbers import Number
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from corenet.metrics import METRICS_REGISTRY
from corenet.metrics.metric_base import AverageMetric
from corenet.utils import logger


def compute_miou_batch(
    prediction: Union[Tuple[Tensor, Tensor], Tensor],
    target: Tensor,
    epsilon: Optional[float] = 1e-7,
):
    if isinstance(prediction, Tuple) and len(prediction) == 2:
        mask = prediction[0]
        assert isinstance(mask, Tensor)
    elif isinstance(prediction, Tensor):
        mask = prediction
        assert isinstance(mask, Tensor)
    else:
        raise NotImplementedError(
            "For computing loss for segmentation task, we need prediction to be an instance of Tuple or Tensor"
        )

    num_classes = mask.shape[1]
    pred_mask = torch.max(mask, dim=1)[1]
    assert (
        pred_mask.dim() == 3
    ), "Predicted mask tensor should be 3-dimensional (B x H x W)"

    pred_mask = pred_mask.byte()
    target = target.byte()

    # shift by 1 so that 255 is 0
    pred_mask += 1
    target += 1

    pred_mask = pred_mask * (target > 0)
    inter = pred_mask * (pred_mask == target)
    area_inter = torch.histc(inter.float(), bins=num_classes, min=1, max=num_classes)
    area_pred = torch.histc(pred_mask.float(), bins=num_classes, min=1, max=num_classes)
    area_mask = torch.histc(target.float(), bins=num_classes, min=1, max=num_classes)
    area_union = area_pred + area_mask - area_inter + epsilon
    return area_inter, area_union


@METRICS_REGISTRY.register(name="iou")
class IOUMetric(AverageMetric):
    def gather_metrics(
        self,
        prediction: Union[Tensor, Dict],
        target: Union[Tensor, Dict],
        extras: Dict[str, Any],
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        This function gathers intersection and union metrics from different processes and converts to float.
        """
        if isinstance(prediction, Tensor) and isinstance(target, Tensor):
            inter, union = compute_miou_batch(prediction=prediction, target=target)
            return {"inter": inter, "union": union}
        # elif isinstance(prediction, Dict):
        #    logger.error("IOU metrics are not supported for a dictionary of predictions")
        # We will revisit it later, as per the use case.

        # inter_dict = {}
        # union_dict = {}
        # for k, v in prediction.items():
        #     inter, union = compute_miou_batch(prediction=v, target=target)
        #     inter = tensor_to_python_float(inter, is_distributed=is_distributed)
        #     union = tensor_to_python_float(union, is_distributed=is_distributed)
        #     inter_dict[k] = inter
        #     union_dict[k] = union
        # return inter_dict, union_dict
        else:
            logger.error("Metric monitor supports Tensor only for IoU")

    def compute(self) -> Union[Number, Dict[str, Number]]:
        averaged = super().compute()
        iou = averaged["inter"] / averaged["union"]

        if isinstance(iou, Tensor):
            iou = iou.cpu().numpy()

        # Converting iou from [0, 1] to [0, 100]
        # other metrics are by default in [0, 100 range]
        avg_iou = np.mean(iou) * 100.0

        return avg_iou
