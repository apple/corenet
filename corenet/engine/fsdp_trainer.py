#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import gc
import time
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from corenet.constants import if_test_env
from corenet.data.loader.dataloader import CoreNetDataLoader
from corenet.data.transforms.image_torch import apply_mixing_transforms
from corenet.engine.default_trainer import DefaultTrainer
from corenet.engine.utils import get_batch_size
from corenet.loss_fn.base_criteria import BaseCriteria
from corenet.metrics.stats import Statistics
from corenet.modeling.misc.averaging_utils import EMA
from corenet.modeling.models.base_model import BaseAnyNNModel
from corenet.modeling.models.fsdp_wrapper import FullyShardedDataParallelWrapper
from corenet.optims.base_optim import BaseOptim
from corenet.optims.scheduler.base_scheduler import BaseLRScheduler
from corenet.utils import logger
from corenet.utils.common_utils import move_to_device


class FSDPTrainer(DefaultTrainer):
    """
    This class defines the training and validation code for training models with FullyShardedDataParallel.

    Args:
        opts: The command-line arguments as a namespace.
        model: The neural network model to be trained.
        validation_loader: The data loader for the validation dataset.
        training_loader: The data loader for the training dataset.
        criteria: The loss function used for training.
        optimizer: The optimizer used for training.
        scheduler: The learning rate scheduler used for training.
        start_epoch: The epoch to start training from.
        start_iteration: The iteration to start training from.
        best_metric: The best validation metric value achieved so far.
        model_ema: An optional instance of EMA model.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        model: BaseAnyNNModel,
        validation_loader: CoreNetDataLoader,
        training_loader: CoreNetDataLoader,
        criteria: BaseCriteria,
        optimizer: BaseOptim,
        scheduler: BaseLRScheduler,
        start_epoch: int = 0,
        start_iteration: int = 0,
        best_metric: float = 0.0,
        model_ema: Optional[EMA] = None,
    ) -> None:
        if getattr(opts, "common.accum_freq") > 1:
            logger.error("Gradient accumumlation is not supported with FSDP")

        if getattr(opts, "ema.enable"):
            logger.error("EMA is not yet supported with FSDP")

        if not isinstance(model, FullyShardedDataParallelWrapper):
            logger.error(
                f"{self.__class__.__name__} expects model to be an instance of FullyShardedDataParallelWrapper"
            )

        super().__init__(
            opts=opts,
            model=model,
            validation_loader=validation_loader,
            training_loader=training_loader,
            criteria=criteria,
            optimizer=optimizer,
            scheduler=scheduler,
            # FSDP does not use autocast for mixed-precision training and handles it internally.
            # So, gradient scalar is not required.
            gradient_scaler=None,
            start_epoch=start_epoch,
            start_iteration=start_iteration,
            best_metric=best_metric,
            model_ema=model_ema,
        )

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train model for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Average value for loss and checkpoint metric.
        """
        time.sleep(
            if_test_env(0.5, otherwise=2)
        )  # To prevent possible deadlock during epoch transition

        if self.is_master_node:
            logger.double_dash_line()
            logger.info(f"Training epoch {epoch}")

        train_stats = Statistics(
            opts=self.opts,
            metric_names=self.train_metric_names,
            is_master_node=self.is_master_node,
            is_distributed=self.use_distributed,
            log_writers=self.log_writers,
        )

        # set the training mode for model and criteria
        self._set_training_mode()

        max_norm = getattr(self.opts, "common.grad_clip")

        # set the gradient to zero or None
        self._zero_grad()

        epoch_start_time = time.time()
        batch_load_start = time.time()
        grad_norm = torch.tensor([0.0], dtype=torch.float, device=self.device)
        for batch_id, batch in enumerate(self.train_loader):
            if self.train_iterations > self.max_iterations:
                self.max_iterations_reached = True
                break

            # move to device
            batch = move_to_device(opts=self.opts, x=batch, device=self.device)
            # apply mix-up transforms if any
            batch = apply_mixing_transforms(opts=self.opts, data=batch)

            batch_load_toc = time.time() - batch_load_start

            samples, targets = batch["samples"], batch["targets"]

            batch_size = get_batch_size(samples)

            # update the learning rate
            self.optimizer = self.scheduler.update_lr(
                optimizer=self.optimizer, epoch=epoch, curr_iter=self.train_iterations
            )

            # prediction
            pred_label = self.model(samples)
            # compute loss
            loss_dict_or_tensor: Union[Dict, Tensor] = self.criteria(
                input_sample=samples,
                prediction=pred_label,
                target=targets,
                epoch=epoch,
                iterations=self.train_iterations,
            )

            if isinstance(loss_dict_or_tensor, Dict):
                if "total_loss" not in loss_dict_or_tensor.keys():
                    logger.error(
                        "'total_loss' key is required for loss functions that return outputs as dictionary."
                    )
                loss = loss_dict_or_tensor["total_loss"]
            elif isinstance(loss_dict_or_tensor, Tensor):
                loss = loss_dict_or_tensor
            else:
                logger.error(
                    f"Loss value should be an instance of Tensor or Dict. Got: {type(loss_dict_or_tensor)}"
                )

            if not isinstance(loss, torch.Tensor):
                logger.error(
                    f"Loss value should be an instance of Tensor. Got: {type(loss)}"
                )

            if torch.isnan(loss):
                logger.error("Nan encountered in the loss.")

            loss.backward()
            if max_norm is not None:
                self.model.clip_grad_norm_(max_norm=max_norm)

            if "grad_norm" in self.train_metric_names:
                # compute grad_norm for logging purposes.
                # We can't use the output of clip_grad_norm_ because it returns the total norm before clipping
                grad_norm = self.compute_grad_norm()

            self.optimizer.step()

            self._zero_grad()
            self.train_iterations += 1

            train_stats.update(
                pred_label=pred_label,
                target_label=targets,
                extras={"loss": loss_dict_or_tensor, "grad_norm": grad_norm},
                batch_time=batch_load_toc,
                batch_size=batch_size,
            )

            self._save_and_evaluate_interval_checkpoint(
                epoch=epoch, iterations=self.train_iterations, loss=loss
            )

            if batch_id % self.log_freq == 0 and self.is_master_node:
                lr = self.scheduler.retrieve_lr(self.optimizer)
                train_stats.iter_summary(
                    epoch=epoch,
                    n_processed_samples=self.train_iterations,
                    total_samples=self.max_iterations,
                    learning_rate=lr,
                    elapsed_time=epoch_start_time,
                )

            batch_load_start = time.time()

        avg_loss = train_stats.avg_statistics(
            metric_name="loss", sub_metric_name="total_loss"
        )
        train_stats.epoch_summary(epoch=epoch, stage="training")
        avg_ckpt_metric = train_stats.avg_statistics(
            metric_name=self.ckpt_metric, sub_metric_name=self.ckpt_submetric
        )

        # Python may not clean up the variables, so calling gc explicitly.
        gc.collect()
        return avg_loss, avg_ckpt_metric
