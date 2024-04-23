#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

from corenet.metrics import METRICS_REGISTRY
from corenet.metrics.metric_base import AverageMetric
from corenet.utils import logger


def compute_psnr(
    prediction: Tensor, target: Tensor, no_uint8_conversion: Optional[bool] = False
) -> Tensor:

    if not no_uint8_conversion:
        prediction = prediction.mul(255.0).to(torch.uint8)
        target = target.mul(255.0).to(torch.uint8)
        MAX_I = 255**2
    else:
        MAX_I = 1

    error = torch.pow(prediction - target, 2).float()
    mse = torch.mean(error) + 1e-10
    psnr = 10.0 * torch.log10(MAX_I / mse)
    return psnr


@METRICS_REGISTRY.register(name="psnr")
class PSNRMetric(AverageMetric):
    def gather_metrics(
        self,
        prediction: Union[Tensor, Dict],
        target: Union[Tensor, Dict],
        extras: Dict[str, Any],
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        This function gathers psnr scores from different processes and converts to float.
        """
        # We have four combinations between prediction and target types:
        # 1. (Tensor, Tensor)
        # 2. (Dict, Tensor)
        # 3. (Dict, Dict)
        # 4. (Tensor, Dict) --> This combination is rare

        if isinstance(prediction, Tensor) and isinstance(target, Tensor):
            if prediction.numel() != target.numel():
                logger.error(
                    "Prediction and target have different number of elements."
                    "Got: Prediction={} and target={}".format(
                        prediction.shape, target.shape
                    )
                )
            psnr = compute_psnr(prediction=prediction, target=target)
            return psnr
        elif isinstance(prediction, Dict) and isinstance(target, Tensor):
            psnr_dict = {}
            for pred_k, pred_v in prediction.items():
                # only compute PSNR where prediction size and target sizes are the same
                if isinstance(pred_v, Tensor) and (pred_v.numel() == target.numel()):
                    psnr = compute_psnr(prediction=pred_v, target=target)
                    psnr_dict[pred_k] = psnr
            return psnr_dict
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

            psnr_dict = {}
            for pred_k in intersection_keys:
                pred_v = prediction[pred_k]
                target_v = target[pred_k]
                # only compute PSNR where prediction size and target sizes are the same
                if (
                    isinstance(pred_v, Tensor)
                    and isinstance(target_v, Tensor)
                    and (pred_v.numel() == target_v.numel())
                ):
                    psnr = compute_psnr(prediction=pred_v, target=target_v)
                    psnr_dict[pred_k] = psnr
            return psnr_dict
        elif isinstance(prediction, Tensor) and isinstance(target, Dict):
            psnr_dict = {}
            for target_k, target_v in target.items():
                # only compute PSNR where prediction size and target sizes are the same
                if isinstance(target_v, Tensor) and (
                    prediction.numel() == target_v.numel()
                ):
                    psnr = compute_psnr(prediction=prediction, target=target_v)
                    psnr_dict[target_k] = psnr
            return psnr_dict
        else:
            logger.error("Metric monitor supports Tensor or Dict of Tensors")
