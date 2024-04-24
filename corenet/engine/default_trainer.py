#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import copy
import gc
import shutil
import time
import traceback
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.cuda.amp import GradScaler
from torch.nn import functional as F

from corenet.constants import DEFAULT_EPOCHS, DEFAULT_ITERATIONS, if_test_env
from corenet.data.loader.dataloader import CoreNetDataLoader
from corenet.data.sampler.base_sampler import BaseSampler, BaseSamplerDDP
from corenet.data.transforms.image_torch import apply_mixing_transforms
from corenet.engine.utils import (
    autocast_fn,
    get_batch_size,
    get_log_writers,
    log_metrics,
)
from corenet.loss_fn import BaseCriteria
from corenet.metrics.stats import Statistics
from corenet.modeling.misc.averaging_utils import EMA
from corenet.modeling.models.base_model import BaseAnyNNModel
from corenet.optims import BaseOptim
from corenet.optims.scheduler import BaseLRScheduler
from corenet.options.parse_args import parse_validation_metric_names
from corenet.utils import logger
from corenet.utils.checkpoint_utils import save_checkpoint
from corenet.utils.common_utils import move_to_device
from corenet.utils.ddp_utils import dist_barrier, is_master
from corenet.utils.tensor_utils import reduce_tensor_sum


class DefaultTrainer(object):
    """
    Default training and validation engine.

    Args:
        opts: The command-line arguments as a namespace.
        model: The neural network model to be trained.
        validation_loader: The data loader for the validation dataset.
        training_loader: The data loader for the training dataset.
        criteria: The loss function used for training.
        optimizer: The optimizer used for training.
        scheduler: The learning rate scheduler used for training.
        gradient_scaler: The gradient scaler for mixed precision training.
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
        gradient_scaler: GradScaler,
        start_epoch: int = 0,
        start_iteration: int = 0,
        best_metric: float = 0.0,
        model_ema: EMA = None,
    ) -> None:
        super(DefaultTrainer, self).__init__()

        self.opts = opts

        self.model = model
        self.model_ema = model_ema
        self.criteria = criteria
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradient_scaler = gradient_scaler

        self.val_loader = validation_loader
        self.train_loader = training_loader

        self.device = getattr(opts, "dev.device", torch.device("cpu"))

        self.start_epoch = start_epoch
        self.best_metric = best_metric
        self.train_iterations = start_iteration

        self.is_master_node = is_master(opts)
        self.max_iterations_reached = False
        self.max_iterations = getattr(
            self.opts, "scheduler.max_iterations", DEFAULT_ITERATIONS
        )
        self.use_distributed = getattr(self.opts, "ddp.use_distributed")
        self.log_freq = getattr(self.opts, "common.log_freq")
        self.accum_freq = getattr(self.opts, "common.accum_freq")
        self.accum_after_epoch = getattr(self.opts, "common.accum_after_epoch")

        self.mixed_precision_training = getattr(opts, "common.mixed_precision")
        self.mixed_precision_dtype = getattr(opts, "common.mixed_precision_dtype")

        self.train_metric_names = getattr(opts, "stats.train")
        if isinstance(self.train_metric_names, str):
            self.train_metric_names = [self.train_metric_names]

        assert isinstance(
            self.train_metric_names, list
        ), "Type of metric names should be list. Got: {}".format(
            type(self.train_metric_names)
        )

        if "loss" not in self.train_metric_names:
            self.train_metric_names.append(self.train_metric_names)

        (
            self.val_metric_names,
            self.ckpt_metric,
            self.ckpt_submetric,
        ) = parse_validation_metric_names(self.opts)

        self.save_all_checkpoints = getattr(self.opts, "common.save_all_checkpoints")

        self.save_location = getattr(opts, "common.exp_loc")

        self.log_writers = get_log_writers(self.opts, save_location=self.save_location)

        self.set_grad_to_none = getattr(opts, "common.set_grad_to_none")

        # save interval checkpoints every `save_interval_freq` updates on the master node
        self.save_interval_freq = getattr(opts, "common.save_interval_freq")
        self.eval_every_k_iterations = getattr(opts, "common.eval_every_k_iterations")

    def compute_grad_norm(self) -> torch.Tensor:
        """Computes and returns the L2 norm of the gradients."""
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return None

        norm_type = 2.0  # L2 norm

        inv_scale = 1.0 / self.gradient_scaler.get_scale()
        total_norm = torch.norm(
            torch.stack(
                [
                    torch.norm(p.grad.detach() * inv_scale, norm_type).to(self.device)
                    for p in parameters
                ]
            ),
            norm_type,
        )
        if total_norm.isnan() or total_norm.isinf():
            return None
        return total_norm

    def _zero_grad(self) -> None:
        """Sets the gradients to zero.

        ...note:
            If 'set_grad_to_none' is enabled, gradients are set to None instead of zero.
            Caution should be exercised when using this option as setting gradients to None may alter certain behaviors.
            Refer to the PyTorch's 'torch.optim.Optimizer' documentation for detailed explanations.
        """
        if self.set_grad_to_none:
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.zero_grad()

    def _set_training_mode(self) -> None:
        """
        Sets the model and criteria in training mode.
        """
        self.model.train()
        self.criteria.train()

    def _set_eval_mode(self) -> None:
        """
        Sets the model and criteria in eval mode.
        """
        self.model.eval()
        self.criteria.eval()

    def _save_and_evaluate_interval_checkpoint(
        self, epoch: int, iterations: int, loss: torch.Tensor
    ) -> None:
        """
        Optionally save the interval checkpoints and evaluate them.

        Args:
            epoch: Current epoch.
            iterations: Current training iteration.
            loss: Loss value.
        """
        # save the checkpoint every N updates
        if (self.save_interval_freq > 0) and (
            iterations + 1
        ) % self.save_interval_freq == 0:

            # set the model to eval mode and perform evaluation
            self._set_eval_mode()

            save_checkpoint(
                iterations=iterations,
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                best_metric=self.best_metric,
                is_best=False,
                save_dir=self.save_location,
                is_master_node=self.is_master_node,
                gradient_scaler=self.gradient_scaler,
                model_ema=self.model_ema,
                is_ema_best=False,
                max_ckpt_metric=False,
                k_best_checkpoints=-1,
                save_all_checkpoints=True,
                is_interval_ckpt=True,
            )

            # set the model and criteria back to training mode
            self._set_training_mode()

            if self.is_master_node:
                logger.info(
                    "Checkpoints saved after {} updates at: {}".format(
                        iterations, self.save_location
                    ),
                    print_line=True,
                )

        # evaluate the checkpoint every N updates
        if (
            self.eval_every_k_iterations > 0
            and (iterations + 1) % self.eval_every_k_iterations == 0
        ):
            # set the model to eval mode and perform evaluation
            self._set_eval_mode()

            self.val_epoch(epoch=epoch, model=self.model)

            # set the model and criteria back to training mode
            self._set_training_mode()

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train the model for one epoch.

        Args:
            epoch: Current epoch.

        Returns:
            A tuple containing average values of the training loss and the specified checkpoint metric (e.g., top1 accuracy).
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

        self._set_training_mode()

        accum_freq = self.accum_freq if epoch >= self.accum_after_epoch else 1
        max_norm = getattr(self.opts, "common.grad_clip", None)

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

            with autocast_fn(
                enabled=self.mixed_precision_training,
                amp_precision=self.mixed_precision_dtype,
            ):
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
                            "total_loss key is required for loss functions that return outputs as dictionary."
                        )
                    loss = loss_dict_or_tensor["total_loss"]
                elif isinstance(loss_dict_or_tensor, Tensor):
                    loss = loss_dict_or_tensor
                else:
                    logger.error("Loss value should be an instance of Tensor or Dict")

                if isinstance(loss, torch.Tensor) and torch.isnan(loss):
                    logger.error("Nan encountered in the loss.")

            # perform the backward pass with gradient accumulation [Optional]
            self.gradient_scaler.scale(loss).backward()

            if (batch_id + 1) % accum_freq == 0:
                if max_norm is not None:
                    # For gradient clipping, unscale the gradients and then clip them
                    self.gradient_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=max_norm
                    )

                if "grad_norm" in self.train_metric_names:
                    # compute grad_norm for logging purposes.
                    # We can't use the output of clip_grad_norm_ because it returns the total norm before clipping
                    grad_norm = self.compute_grad_norm()

                # optimizer step
                self.gradient_scaler.step(optimizer=self.optimizer)
                # update the scale for next batch
                self.gradient_scaler.update()
                # set the gradient to zero or None
                self._zero_grad()

                self.train_iterations += 1

                if self.model_ema is not None:
                    self.model_ema.update_parameters(self.model)

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

        gc.collect()

        return avg_loss, avg_ckpt_metric

    def val_epoch(
        self, epoch: int, model: BaseAnyNNModel, extra_str=""
    ) -> Tuple[float, float]:
        """Validate the model.

        Args:
            epoch: Current epoch.
            model: The neural network model to be validated.
            extra_str: Extra string to append to the epoch summary. For example, specify whether the model is EMA or not.

        Returns:
            A tuple containing average values of the validation loss and the specified checkpoint metric (e.g., top1 accuracy).
        """
        if self.val_loader is None:
            return 0.0, 0.0

        time.sleep(
            if_test_env(0.5, otherwise=2)
        )  # To prevent possible deadlock during epoch transition
        validation_stats = Statistics(
            opts=self.opts,
            metric_names=self.val_metric_names,
            is_master_node=self.is_master_node,
            is_distributed=self.use_distributed,
            log_writers=self.log_writers,
        )

        model.eval()

        if model.training:
            if self.is_master_node:
                logger.warning(
                    "Model is in training mode. Switching to evaluation mode"
                )
            model.eval()

        self.criteria.eval()
        if self.criteria.training:
            self.criteria.eval()

        with torch.no_grad():
            epoch_start_time = time.time()
            # When validation dataset is an instance of Iterable dataset, then total_samples is redundant.
            total_samples = len(self.val_loader)
            processed_samples = 0
            lr = self.scheduler.retrieve_lr(self.optimizer)
            for batch_id, batch in enumerate(self.val_loader):
                batch = move_to_device(opts=self.opts, x=batch, device=self.device)

                samples, targets = batch["samples"], batch["targets"]

                batch_size = get_batch_size(samples)

                with autocast_fn(
                    enabled=self.mixed_precision_training,
                    amp_precision=self.mixed_precision_dtype,
                ):
                    # prediction
                    pred_label = model(samples)
                    # compute loss
                    loss_dict_or_tensor = self.criteria(
                        input_sample=samples, prediction=pred_label, target=targets
                    )

                processed_samples += batch_size

                validation_stats.update(
                    pred_label=pred_label,
                    target_label=targets,
                    extras={"loss": loss_dict_or_tensor},
                    batch_time=0.0,
                    batch_size=batch_size,
                    # TODO: use is_evaluation?
                )

                if batch_id % self.log_freq == 0 and self.is_master_node:
                    validation_stats.iter_summary(
                        epoch=epoch,
                        n_processed_samples=processed_samples,
                        total_samples=total_samples,
                        elapsed_time=epoch_start_time,
                        learning_rate=lr,
                    )

        validation_stats.epoch_summary(epoch=epoch, stage="validation" + extra_str)
        avg_loss = validation_stats.avg_statistics(
            metric_name="loss", sub_metric_name="total_loss"
        )
        avg_ckpt_metric = validation_stats.avg_statistics(
            metric_name=self.ckpt_metric, sub_metric_name=self.ckpt_submetric
        )

        if avg_ckpt_metric is None:
            avg_ckpt_metric = avg_loss

        gc.collect()

        return avg_loss, avg_ckpt_metric

    def run(
        self, train_sampler: Optional[Union[BaseSampler, BaseSamplerDDP]] = None
    ) -> None:
        """Train and validate the model.

        Args:
            train_sampler: An optional sampler for training dataset.
        """

        train_start_time = time.time()

        cfg_file = getattr(self.opts, "common.config_file", None)
        if cfg_file is not None and self.is_master_node:
            dst_cfg_file = "{}/config.yaml".format(self.save_location)
            shutil.copy(src=cfg_file, dst=dst_cfg_file)
            logger.info(
                "Configuration file is stored here: {}".format(
                    logger.color_text(dst_cfg_file)
                )
            )

        keep_k_best_ckpts = getattr(self.opts, "common.k_best_checkpoints", 5)
        ema_best_metric = self.best_metric
        is_ema_best = False

        try:
            max_epochs = getattr(self.opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
            max_checkpoint_metric = getattr(self.opts, "stats.checkpoint_metric_max")
            for epoch in range(self.start_epoch, max_epochs):
                if train_sampler is not None:
                    # Note that we are using our owm implementations of data samplers
                    # and we have defined this function for both distributed and non-distributed cases
                    train_sampler.set_epoch(epoch)
                    train_sampler.update_scales(
                        epoch=epoch, is_master_node=self.is_master_node
                    )

                train_loss, train_ckpt_metric = self.train_epoch(epoch)

                val_loss, val_ckpt_metric = self.val_epoch(
                    epoch=epoch, model=self.model
                )

                if max_checkpoint_metric:
                    is_best = val_ckpt_metric >= self.best_metric
                    self.best_metric = max(val_ckpt_metric, self.best_metric)
                else:
                    is_best = val_ckpt_metric <= self.best_metric
                    self.best_metric = min(val_ckpt_metric, self.best_metric)

                val_ema_loss = None
                val_ema_ckpt_metric = None
                if self.model_ema is not None:
                    val_ema_loss, val_ema_ckpt_metric = self.val_epoch(
                        epoch=epoch, model=self.model_ema.ema_model, extra_str=" (EMA)"
                    )
                    if max_checkpoint_metric:
                        is_ema_best = val_ema_ckpt_metric >= ema_best_metric
                        ema_best_metric = max(val_ema_ckpt_metric, ema_best_metric)
                    else:
                        is_ema_best = val_ema_ckpt_metric <= ema_best_metric
                        ema_best_metric = min(val_ema_ckpt_metric, ema_best_metric)

                gc.collect()

                save_checkpoint(
                    iterations=self.train_iterations,
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    best_metric=self.best_metric,
                    is_best=is_best,
                    save_dir=self.save_location,
                    is_master_node=self.is_master_node,
                    model_ema=self.model_ema,
                    is_ema_best=is_ema_best,
                    ema_best_metric=ema_best_metric,
                    gradient_scaler=self.gradient_scaler,
                    max_ckpt_metric=max_checkpoint_metric,
                    k_best_checkpoints=keep_k_best_ckpts,
                    save_all_checkpoints=self.save_all_checkpoints,
                    is_interval_ckpt=False,
                )

                if self.is_master_node:
                    lr_list = self.scheduler.retrieve_lr(self.optimizer)

                    for log_writer in self.log_writers:
                        log_metrics(
                            lrs=lr_list,
                            log_writer=log_writer,
                            train_loss=train_loss,
                            val_loss=val_loss,
                            epoch=epoch,
                            best_metric=self.best_metric,
                            val_ema_loss=val_ema_loss,
                            ckpt_metric_name=self.ckpt_metric,
                            train_ckpt_metric=train_ckpt_metric,
                            val_ckpt_metric=val_ckpt_metric,
                            val_ema_ckpt_metric=val_ema_ckpt_metric,
                        )

                if self.max_iterations_reached:
                    if self.use_distributed:
                        dist_barrier()

                    if self.is_master_node:
                        logger.info("Max. iterations for training reached")
                    break
        except KeyboardInterrupt as e:
            if self.is_master_node:
                logger.log("Keyboard interruption. Exiting from early training")
                raise e
        except Exception as e:
            if "out of memory" in str(e):
                logger.log("OOM exception occurred")
                n_gpus = getattr(self.opts, "dev.num_gpus", 1)
                for dev_id in range(n_gpus):
                    mem_summary = torch.cuda.memory_summary(
                        device=torch.device("cuda:{}".format(dev_id)), abbreviated=True
                    )
                    logger.log("Memory summary for device id: {}".format(dev_id))
                    print(mem_summary)

            logger.log(
                f"Exception occurred that interrupted the training:\n{traceback.format_exc()}"
            )
            raise e
        finally:
            use_distributed = getattr(self.opts, "ddp.use_distributed", False)
            if use_distributed:
                torch.distributed.destroy_process_group()

            torch.cuda.empty_cache()

            for log_writer in self.log_writers:
                log_writer.close()

            if self.is_master_node:
                train_end_time = time.time()
                hours, rem = divmod(train_end_time - train_start_time, 3600)
                minutes, seconds = divmod(rem, 60)
                train_time_str = "{:0>2}:{:0>2}:{:05.2f}".format(
                    int(hours), int(minutes), seconds
                )
                logger.log("Training took {}".format(train_time_str))
