#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import time

import torch

from corenet.constants import DEFAULT_LOG_FREQ, SUPPORTED_VIDEO_CLIP_VOTING_FN
from corenet.data.loader.dataloader import CoreNetDataLoader
from corenet.engine.utils import autocast_fn, get_batch_size, get_log_writers
from corenet.loss_fn.base_criteria import BaseCriteria
from corenet.metrics.stats import Statistics
from corenet.modeling.models.base_model import BaseAnyNNModel
from corenet.options.parse_args import parse_validation_metric_names
from corenet.utils import logger
from corenet.utils.common_utils import move_to_device
from corenet.utils.ddp_utils import is_master


class Evaluator:
    def __init__(
        self,
        opts: argparse.Namespace,
        model: BaseAnyNNModel,
        test_loader: CoreNetDataLoader,
        criteria: BaseCriteria,
    ) -> None:
        """
        Evaluates the criteria on a validation or test dateset using a model.

        Args:
            opts: The command-line arguments as a namespace.
            model: The neural network model to be trained.
            test_loader: The data loader for the test dataset.
            criteria: The loss function used for training.
        """
        self.opts = opts

        self.model = model
        self.criteria = criteria

        self.test_loader = test_loader

        self.device = getattr(opts, "dev.device", torch.device("cpu"))
        self.use_distributed = getattr(self.opts, "ddp.use_distributed", False)
        self.is_master_node = is_master(opts)
        self.stage_name = getattr(opts, "common.eval_stage_name", "evaluation")

        self.mixed_precision_training = getattr(opts, "common.mixed_precision", False)
        self.mixed_precision_dtype = getattr(
            opts, "common.mixed_precision_dtype", "float16"
        )

        (
            self.metric_names,
            self.ckpt_metric,
            self.ckpt_submetric,
        ) = parse_validation_metric_names(self.opts)

        self.log_writers = get_log_writers(self.opts, save_location=None)

    def eval_fn(self) -> None:
        model = self.model
        criteria = self.criteria
        log_freq = getattr(self.opts, "common.log_freq", DEFAULT_LOG_FREQ)

        evaluation_stats = Statistics(
            opts=self.opts,
            metric_names=self.metric_names,
            is_master_node=self.is_master_node,
            is_distributed=self.use_distributed,
            log_writers=self.log_writers,
        )

        model.eval()
        criteria.eval()

        with torch.no_grad():
            epoch_start_time = time.time()
            total_samples = len(self.test_loader)
            processed_samples = 0

            for batch_id, batch in enumerate(self.test_loader):
                batch = move_to_device(opts=self.opts, x=batch, device=self.device)

                samples, targets = batch["samples"], batch["targets"]

                batch_size = get_batch_size(samples)

                with autocast_fn(
                    enabled=self.mixed_precision_training,
                    amp_precision=self.mixed_precision_dtype,
                ):
                    # prediction
                    pred_label = model(samples)
                    loss_dict_or_tensor = criteria(
                        input_sample=samples, prediction=pred_label, target=targets
                    )

                processed_samples += batch_size

                evaluation_stats.update(
                    pred_label=pred_label,
                    target_label=targets,
                    extras={"loss": loss_dict_or_tensor},
                    batch_time=0.0,
                    batch_size=batch_size,
                )

                if batch_id % log_freq == 0 and self.is_master_node:
                    evaluation_stats.iter_summary(
                        epoch=0,
                        n_processed_samples=processed_samples,
                        total_samples=total_samples,
                        elapsed_time=epoch_start_time,
                        learning_rate=0.0,
                    )

        evaluation_stats.epoch_summary(epoch=0, stage=self.stage_name)

    def run(self) -> None:
        eval_start_time = time.time()
        self.eval_fn()
        eval_end_time = time.time() - eval_start_time
        logger.log("Evaluation took {} seconds".format(eval_end_time))
