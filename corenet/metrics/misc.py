#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from numbers import Number
from typing import Any, Dict, Union

import torch
from torch import Tensor

from corenet.metrics import METRICS_REGISTRY
from corenet.metrics.metric_base import AverageMetric
from corenet.utils import logger


@METRICS_REGISTRY.register(name="loss")
class LossMetric(AverageMetric):
    def gather_metrics(
        self,
        prediction: Union[Tensor, Dict],
        target: Union[Tensor, Dict],
        extras: Dict[str, Any],
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        This function gather losses from different processes and converts to float.
        """
        if extras is None:
            extras = {}

        loss = extras.get("loss", None)

        if loss is None:
            loss = 0.0

        if isinstance(loss, Tensor):
            return loss
        elif isinstance(loss, Number):
            return torch.tensor(loss, device=self.device)
        elif isinstance(loss, Dict):
            loss.pop(None, None)

            for k, v in loss.items():
                if isinstance(v, Number):
                    loss[k] = torch.tensor(loss, device=self.device)
                elif not isinstance(v, Tensor):
                    logger.error(
                        "Loss metric supports Number, Tensor, or Dict of Tensors."
                        f" Got {v} with {type(v)} type under key {k}."
                    )

            return loss
        else:
            logger.error(
                "Loss metric supports Number, Tensor, or Dict of Tensors."
                f" Got {loss} with {type(loss)} type."
            )


@METRICS_REGISTRY.register(name="grad_norm")
class GradNormMetric(AverageMetric):
    def gather_metrics(
        self,
        prediction: Union[Tensor, Dict],
        target: Union[Tensor, Dict],
        extras: Dict[str, Any],
    ) -> Union[Tensor, Dict[str, Tensor]]:
        if extras is None:
            extras = {}

        grad_norm = extras.get("grad_norm", None)

        if grad_norm is None:
            grad_norm = 0.0

        if isinstance(grad_norm, Tensor):
            return grad_norm
        elif isinstance(grad_norm, Number):
            return torch.tensor(grad_norm, device=self.device)
        elif isinstance(grad_norm, Dict):
            grad_norm.pop(None, None)

            for k, v in grad_norm.items():
                if isinstance(v, Number):
                    grad_norm[k] = torch.tensor(grad_norm, device=self.device)
                elif isinstance(v, str):
                    del grad_norm[k]
                elif not isinstance(v, Tensor):
                    logger.error(
                        "Grad-norm metric supports Number, Tensor, or Dict of Tensors."
                        f" Got {v} with {type(v)} type under key {k}."
                    )

            return grad_norm
        else:
            logger.error(
                "Grad-norm metric supports Number, Tensor, or Dict of Tensors."
                f" Got {grad_norm} with {type(grad_norm)} type."
            )
