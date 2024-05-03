#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import sys
import time
import traceback
from numbers import Number
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch import Tensor

from corenet.metrics import METRICS_REGISTRY
from corenet.metrics.metric_base import BaseMetric
from corenet.utils import logger
from corenet.utils.object_utils import apply_recursively


class Statistics(object):
    def __init__(
        self,
        opts: argparse.Namespace,
        metric_names: list = ["loss"],
        is_master_node: Optional[bool] = False,
        is_distributed: Optional[bool] = False,
        log_writers: Optional[List] = [],
    ) -> None:
        if len(metric_names) == 0:
            logger.error("Metric names list cannot be empty")

        # key is the metric name and value is the value
        self.metric_dict: Dict[str, BaseMetric] = {}
        for m_name in metric_names:
            if m_name in METRICS_REGISTRY:
                self.metric_dict[m_name] = METRICS_REGISTRY[m_name](
                    opts=opts, is_distributed=is_distributed
                )
            else:
                if is_master_node:
                    logger.log(
                        "{} statistics not supported. Supported: {}".format(
                            m_name, METRICS_REGISTRY.keys()
                        )
                    )

        self.round_places = 4
        self.is_master_node = is_master_node
        self.log_writers = log_writers

        self.batch_time = 0
        self.batch_counter = 0

    def update(
        self,
        pred_label: Union[Tensor, Dict],
        target_label: Union[Tensor, Dict],
        extras: Optional[Dict[str, Any]] = None,
        batch_time: Optional[float] = 0.0,
        batch_size: Optional[int] = 1,
    ) -> None:
        """
        Updates all the metrics after a batch.

        :param pred_label: predictions coming from a model (must be a Tensor or a Dict of Tensors)
        :param target_label: GT labels (Tensor or a Dict of Tensors)
        :param extras: Optional Dict containing extra info, usually Loss and GradNorm
                       e.g. {"loss": loss_value, "grad_norm": gradient_norm}
        :param batch_time: Optional time it took to run through the batch
        :param n: batch size (to be used in averaging the numbers correctly)
        """
        for metric_name, metric in self.metric_dict.items():
            try:
                metric.update(
                    prediction=pred_label,
                    target=target_label,
                    extras=extras,
                    batch_size=batch_size,
                )
            except Exception as e:
                traceback.print_exc()
                logger.error(
                    "Caught an error while updating metric {}: {}".format(
                        metric_name, e
                    )
                )

        self.batch_time += batch_time
        self.batch_counter += 1

    def _avg_statistics_all(self, sep=": ", metrics=None) -> List[str]:
        """
        This function computes average statistics of all metrics and returns them as a list of strings.

        Examples:
         loss: 12.9152
         loss: {'total_loss': 12.9152, 'reg_loss': 2.8199, 'cls_loss': 10.0953}
        """
        if metrics is None:
            metrics = self._compute_avg_statistics_all()

        return [
            self.metric_dict[name].summary_string(name, sep, avg)
            for name, avg in metrics.items()
            if isinstance(avg, Number) or avg
        ]

    def _compute_avg_statistics_all(self) -> Dict[str, Union[float, Dict]]:
        metric_stats = {}
        for metric_name, metric in self.metric_dict.items():
            value = metric.compute()
            metric_stats[metric_name] = apply_recursively(
                value, lambda x: round(x * 1.0, self.round_places)
            )

        return metric_stats

    # TODO: change name: avg is presumptuous
    def compute_avg_statistics(
        self, metric_name: str, sub_metric_name: Optional[str] = None, *args, **kwargs
    ) -> float:
        """
        This function computes the average statistics of a given metric.

        .. note::
        The statistics are stored in form of a dictionary and each key-value pair can be of string and number
        OR string and dictionary of string and number.

        Examples:
             {'loss': 10.0, 'top-1': 50.0}
             {'loss': {'total_loss': 10.0, 'cls_loss': 2.0, 'reg_loss': 8.0}, 'mAP': 5.0}

        """
        if metric_name in self.metric_dict:
            computed_metric = self.metric_dict[metric_name].compute()
            computed_metric = apply_recursively(
                computed_metric, lambda x: round(x * 1.0, self.round_places)
            )

            if isinstance(computed_metric, Dict):
                if sub_metric_name is not None:
                    if sub_metric_name in computed_metric:
                        return computed_metric[sub_metric_name]
                    else:
                        logger.error(
                            "{} not present in the dictionary. Available keys are: {}".format(
                                sub_metric_name, list(computed_metric.keys())
                            )
                        )
                else:
                    return None

            elif isinstance(computed_metric, Number):
                return computed_metric
            else:
                return None

        return None

    def iter_summary(
        self,
        epoch: int,
        n_processed_samples: int,
        total_samples: int,
        elapsed_time: float,
        learning_rate: float or list,
    ) -> None:
        if self.is_master_node:
            metric_stats = self._avg_statistics_all()
            el_time_str = "Elapsed time: {:5.2f}".format(time.time() - elapsed_time)
            if isinstance(learning_rate, float):
                lr_str = "LR: {:1.6f}".format(learning_rate)
            else:
                learning_rate = [round(lr, 6) for lr in learning_rate]
                lr_str = "LR: {}".format(learning_rate)
            epoch_str = "Epoch: {:3d} [{:8d}/{:8d}]".format(
                epoch, n_processed_samples, total_samples
            )
            batch_str = "Avg. batch load time: {:1.3f}".format(
                self.batch_time / self.batch_counter
            )

            stats_summary = [epoch_str]
            stats_summary.extend(metric_stats)
            stats_summary.append(lr_str)
            stats_summary.append(batch_str)
            stats_summary.append(el_time_str)

            summary_str = ", ".join(stats_summary)
            logger.log(summary_str)
            sys.stdout.flush()

    def epoch_summary(self, epoch: int, stage: Optional[str] = "Training") -> None:
        if self.is_master_node:
            metrics = self._compute_avg_statistics_all()
            metric_stats = self._avg_statistics_all(sep="=", metrics=metrics)
            metric_stats_str = " || ".join(metric_stats)
            logger.log("*** {} summary for epoch {}".format(stage.title(), epoch))
            print("\t {}".format(metric_stats_str))
            sys.stdout.flush()

            # TODO: this step is only here for backward-compatibility. We can remove it as well
            shortened_stage_map = {
                "training": "Train",
                "validation": "Val",
                "evaluation": "Eval",
                "validation (EMA)": "Val_EMA",
            }
            s_stage = shortened_stage_map.get(stage, stage)

            for metric_name, metric in self.metric_dict.items():
                values = metrics[metric_name]
                for log_writer in self.log_writers:
                    for scalar_name, scalar_value in metric.flatten_metric(
                        values, metric_name
                    ).items():
                        if metric.is_epoch_summary_enabled_for_metric(
                            scalar_name, log_writer
                        ):
                            log_writer.add_scalar(
                                "{}/{}".format(s_stage, scalar_name.title()),
                                scalar_value,
                                epoch,
                            )
