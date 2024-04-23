#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import abc
import argparse
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from corenet.utils import logger
from corenet.utils.object_utils import flatten_to_dict
from corenet.utils.tensor_utils import (
    all_gather_list,
    reduce_tensor_sum,
    tensor_to_python_float,
)


class BaseMetric(abc.ABC):
    def __init__(
        self,
        opts: Optional[argparse.Namespace] = None,
        is_distributed: bool = False,
        pred: str = None,
        target: str = None,
    ):
        self.opts = opts
        # We need the default value of device for tests.
        self.device = getattr(opts, "dev.device", "cpu")
        self.is_distributed = is_distributed
        self.pred_key = pred
        self.target_key = target
        self.reset()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add metric specific arguments"""
        return parser

    @abc.abstractmethod
    def reset(self) -> None:
        """
        Resets all aggregated data.
        Called at the start of every epoch.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def update(
        self,
        prediction: Union[Tensor, Dict],
        target: Union[Tensor, Dict],
        extras: Dict[str, Any],
        batch_size: Optional[int] = 1,
    ) -> None:
        """
        Processes a new batch of predictions and targets for computing the metric.

        Args:
            predictions: model outputs for the current batch
            target: labels for the current batch
            extras: dict containing extra information.
                During training this includes "loss" and "grad_norm" keys.
                During validaiton only includes "loss".
            batch_size: optionally used to correctly compute the averages when
                the batch size varies across batches.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute(
        self,
    ) -> Union[Number, List, Dict[str, Any]]:
        """
        Computes the metrics with the existing data.

        It gets called at every log iteration as well as the end of each epoch,
        e.g. train, val, valEMA.
        Logging happens at iteration 1 and every `common.log_freq` thereafter.

        Note: for computationally heavy metrics, you may want to increase `common.log_freq`.

        Returns:
            Depending on the metric, can return a scalar metric or a dictionary of metrics.
            Lists (or dicts of lists) are also generally accepted but not encouraged.
        """
        raise NotImplementedError

    def preprocess_predictions(
        self, prediction: Union[Tensor, Dict]
    ) -> Union[Tensor, Dict]:
        if isinstance(prediction, dict) and self.pred_key in prediction:
            prediction = prediction[self.pred_key]

        return prediction

    def preprocess_targets(self, target: Union[Tensor, Dict]) -> Union[Tensor, Dict]:
        if isinstance(target, dict) and self.target_key in target:
            target = target[self.target_key]

        return target

    def summary_string(self, name: str, sep: str, values: Dict[str, Any]) -> str:
        """
        Get a string representation of the given metric values, suitable for
        printing to the terminal.

        Note that we might not print everything inside @values, e.g. if it would
        create too large of an output that would make logs too verbose.

        Args:
            name: The name of the metric.
            sep: The separator used in the printout.
            values: The metric values, as output by @self.compute.

        Returns:
            A string representation of the metric.
        """
        return f"{name:<}{sep}{values}"

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
        return True

    def flatten_metric(
        self, values: Union[Number, List, Dict[str, Any]], metric_name: str
    ) -> Dict[str, Union[Number, List, Dict[str, Any]]]:
        """
        Flatten the given metric @values, prepending @metric_name to the
        resulting dictionary's keys.

        Args:
            values: The values, as output by @self.compute.
            metric_name: The metric name key prefix.

        Returns:
            A version of @values that has been flattened, with key names
                starting with @metric_name.
        """
        return flatten_to_dict(values, metric_name)


class AverageMetric(BaseMetric):
    def reset(self):
        self.count = 0
        self.value = None

    @abc.abstractmethod
    def gather_metrics(
        self,
        prediction: Union[Tensor, Dict],
        target: Union[Tensor, Dict],
        extras: Dict[str, Any],
    ) -> Union[Tensor, Dict[str, Tensor]]:
        raise NotImplementedError(
            "gather_metrics needs to be implemented for subclasses of AverageMetric"
        )

    def _aggregate_ddp_sum(
        self, value: Union[Tensor, Number]
    ) -> Union[float, List[float]]:
        """
        Given a value, sums it up across distributed workers (if distributed) and
        returns the value as a float (if scalar) or a Numpy array (otherwise).
        """
        with torch.no_grad():
            if not isinstance(value, Tensor):
                value = torch.tensor(value)
            value = value.to(device=self.device)

            value = tensor_to_python_float(
                value,
                is_distributed=self.is_distributed,
                reduce_op="sum",
            )
        return value

    def update(
        self,
        prediction: Union[Tensor, Dict],
        target: Union[Tensor, Dict],
        extras: Optional[Dict[str, Any]] = {},
        batch_size: Optional[int] = 1,
    ) -> None:
        prediction = self.preprocess_predictions(prediction)
        target = self.preprocess_targets(target)

        metric = self.gather_metrics(prediction, target, extras)

        if isinstance(metric, Dict):
            # The values should be summed over all existing workers
            metric = {
                k: self._aggregate_ddp_sum(v * batch_size) for k, v in metric.items()
            }
            if self.value is None:
                self.value = metric
            else:
                for k, v in metric.items():
                    self.value[k] += v

        elif isinstance(metric, Tensor):
            if self.value is None:
                self.value = 0

            # The value should be summed over all existing workers
            self.value += self._aggregate_ddp_sum(metric * batch_size)
        else:
            raise ValueError(
                "gather_metrics should return a Tensor or a Dict containing Tensors. Got {}: {}".format(
                    metric.__class__, metric
                )
            )

        # The count should be summed over all existing workers
        self.count += self._aggregate_ddp_sum(batch_size)

    def compute(self) -> Union[Number, List, Dict[str, Any]]:
        if self.value is None:
            return {}
        elif isinstance(self.value, Number):
            return self.value / self.count
        elif isinstance(self.value, Dict):
            avg_dict = {k: v / self.count for k, v in self.value.items()}
            return avg_dict


class EpochMetric(BaseMetric):
    def __init__(
        self,
        opts: Optional[argparse.Namespace] = None,
        is_distributed: bool = False,
        pred: str = None,
        target: str = None,
        force_cpu: bool = True,
    ):
        super().__init__(opts, is_distributed, pred, target)
        self.force_cpu = force_cpu

    def reset(self):
        self._predictions: List[Tensor] = []
        self._targets: List[Tensor] = []

    def update(
        self,
        prediction: Union[Tensor, Dict],
        target: Union[Tensor, Dict],
        extras: Dict[str, Any] = None,
        batch_size: Optional[int] = 1,
    ) -> None:
        prediction = self.preprocess_predictions(prediction)
        target = self.preprocess_targets(target)

        if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
            logger.error(
                "EpochMetric only works on Tensor, got {} and {}.".format(
                    type(prediction), type(target)
                )
                + " Please set pred_key or target_key by setting the proper metric name:"
                + " `stats.val: ['metric_name(pred=key1, target=key2)']`"
            )
            return

        if self.is_distributed:
            prediction = all_gather_list(prediction)
            target = all_gather_list(target)
        else:
            prediction = [prediction]
            target = [target]

        # Detach the variables: we don't need to backprop in metrics
        prediction = [x.detach() for x in prediction]
        target = [x.detach() for x in target]
        # By default we move things to CPU so as to not put extra burden on GPU memory
        # but we allow child-classes/instances to keep the data on GPU for efficiency.
        if self.force_cpu:
            prediction = [x.cpu() for x in prediction]
            target = [x.cpu() for x in target]

        self._predictions.extend(prediction)
        self._targets.extend(target)

    def get_aggregates(self) -> Tuple[Tensor, Tensor]:
        """Aggregates predictions and targets.

        This function gets called every time `self.compute` is called, which is at every
        log iteration as well as the end of each epoch, e.g. train, val, valEMA.
        Logging happens at iteration 1 and every `common.log_freq` thereafter.

        Note: for computationally heavy metrics, you may want to increase `common.log_freq`.
        """
        self._predictions = [torch.cat(self._predictions, dim=0)]
        self._targets = [torch.cat(self._targets, dim=0)]

        return self._predictions[0], self._targets[0]

    def compute_with_aggregates(self, predictions: Tensor, targets: Tensor):
        """
        Computes the metrics given aggregated predictions and targets.

        It gets called by `self.compute`. This happens at every
        log iteration as well as the end of each epoch, e.g. train, val, valEMA.
        Logging happens at iteration 1 and every `common.log_freq` thereafter.

        Note: for computationally heavy metrics, you may want to increase `common.log_freq`.
        """
        raise NotImplementedError

    def compute(self) -> Union[Number, List, Dict[str, Any]]:
        predictions, targets = self.get_aggregates()
        return self.compute_with_aggregates(predictions, targets)
