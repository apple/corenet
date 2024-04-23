#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import math
from functools import cached_property
from typing import Callable, Optional, Tuple

import torch
from torch.cuda.amp import GradScaler

from corenet.constants import (
    DEFAULT_EPOCHS,
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MAX_ITERATIONS,
)
from corenet.data.data_loaders import create_test_loader, create_train_val_loader
from corenet.data.loader.dataloader import CoreNetDataLoader
from corenet.data.sampler.base_sampler import BaseSampler
from corenet.engine.default_trainer import DefaultTrainer
from corenet.engine.evaluation_engine import Evaluator
from corenet.loss_fn import build_loss_fn
from corenet.loss_fn.base_criteria import BaseCriteria
from corenet.modeling import get_model
from corenet.modeling.misc.averaging_utils import EMA
from corenet.modeling.models.base_model import BaseAnyNNModel
from corenet.optims import build_optimizer
from corenet.optims.base_optim import BaseOptim
from corenet.optims.scheduler import build_scheduler
from corenet.optims.scheduler.base_scheduler import BaseLRScheduler
from corenet.train_eval_pipelines.base import (
    TRAIN_EVAL_PIPELINE_REGISTRY,
    BaseTrainEvalPipeline,
    Callback,
)
from corenet.utils import logger, resources
from corenet.utils.activation_checkpointing_wrapper import activation_checkpointing
from corenet.utils.checkpoint_utils import load_checkpoint, load_model_state
from corenet.utils.common_utils import create_directories, device_setup
from corenet.utils.ddp_utils import distributed_init, is_master


@TRAIN_EVAL_PIPELINE_REGISTRY.register("default")
class DefaultTrainEvalPipeline(BaseTrainEvalPipeline):
    """TrainEvalPipeline class is responsible for instantiating the components of
    training, evaluation, and/or pipelines that use those common components.

    The consumers of this class should be able to get an instance of any component
    by accessing the corresponding property. Example usage:

    >>> cfg = get_training_arguments()
    >>> pipeline = TrainEvalPipeline(cfg)
    >>> dataset, model = pipeline.dataset, pipeline.model

    Args:
        opts: Commandline options.
    """

    @cached_property
    def is_master_node(self) -> bool:
        """
        Returns True iff ddp rank is 0.
        """
        opts = self.opts
        node_rank = getattr(opts, "ddp.rank")
        if node_rank < 0:
            logger.error("--ddp.rank should be >=0. Got {}".format(node_rank))

        return is_master(opts)

    @cached_property
    def device(self) -> torch.device:
        return getattr(self.opts, "dev.device", torch.device("cpu"))

    @cached_property
    def _train_val_loader_sampler(
        self,
    ) -> Tuple[CoreNetDataLoader, CoreNetDataLoader, BaseSampler]:
        """
        Returns (train_loader, val_loader, train_sampler) tuple.
        """
        opts = self.opts
        return create_train_val_loader(opts)

    @cached_property
    def train_val_loader(self) -> Tuple[CoreNetDataLoader, CoreNetDataLoader]:
        """
        Returns (train_loader, val_loader) tuple.
        """
        train_loader, val_loader, _ = self._train_val_loader_sampler
        return train_loader, val_loader

    @cached_property
    def train_sampler(self) -> BaseSampler:
        """
        Returns training sampler.
        """
        _, _, train_sampler = self._train_val_loader_sampler
        return train_sampler

    @cached_property
    def test_loader(self) -> CoreNetDataLoader:
        opts = self.opts
        return create_test_loader(opts)

    @cached_property
    def scheduler(self) -> BaseLRScheduler:
        opts = self.opts
        is_master_node = self.is_master_node

        is_iteration_based = getattr(opts, "scheduler.is_iteration_based")
        if is_iteration_based:
            max_iter = getattr(opts, "scheduler.max_iterations")
            if max_iter is None or max_iter <= 0:
                logger.log("Setting max. iterations to {}".format(DEFAULT_ITERATIONS))
                setattr(opts, "scheduler.max_iterations", DEFAULT_ITERATIONS)
                max_iter = DEFAULT_ITERATIONS
            setattr(opts, "scheduler.max_epochs", DEFAULT_MAX_EPOCHS)
            if is_master_node:
                logger.log("Max. iteration for training: {}".format(max_iter))
        else:
            max_epochs = getattr(opts, "scheduler.max_epochs")
            if max_epochs is None or max_epochs <= 0:
                logger.log("Setting max. epochs to {}".format(DEFAULT_EPOCHS))
                setattr(opts, "scheduler.max_epochs", DEFAULT_EPOCHS)
            setattr(opts, "scheduler.max_iterations", DEFAULT_MAX_ITERATIONS)
            max_epochs = getattr(opts, "scheduler.max_epochs")
            if is_master_node:
                logger.log("Max. epochs for training: {}".format(max_epochs))
        scheduler = build_scheduler(opts=opts)
        if is_master_node:
            logger.log(logger.color_text("Learning rate scheduler"))
            print(scheduler)
        return scheduler

    def _prepare_model(self) -> Tuple[BaseAnyNNModel, Optional[torch.nn.Module]]:
        """
        Returns a model optionally with a module whose activation needs to be checkpointed.
        """
        # set-up the model
        model = get_model(self.opts)

        # print model information on master node
        if self.is_master_node:
            model.info()

        submodule_class_to_checkpoint = None

        if getattr(self.opts, "model.activation_checkpointing"):
            try:
                submodule_class_to_checkpoint = (
                    model.get_activation_checkpoint_submodule_class()
                )
            except NotImplementedError:
                logger.error(
                    f"Activation checkpoint module is not implemented for {model.__class__.__name__}. \
                    Please implement 'get_activation_checkpoint_submodule_class' method."
                )

        # memory format
        memory_format = (
            torch.channels_last
            if getattr(self.opts, "common.channels_last")
            else torch.contiguous_format
        )
        model = model.to(device=self.device, memory_format=memory_format)
        return model, submodule_class_to_checkpoint

    @cached_property
    def model(self) -> torch.nn.Module:
        """
        Returns a model to be used by train and eval pipelines, given the selected yaml configs.
        """
        opts = self.opts
        is_master_node = self.is_master_node
        device = self.device
        dev_id = getattr(opts, "dev.device_id", None)
        use_distributed = getattr(opts, "ddp.use_distributed")

        model, wrapper_cls_for_act_ckpt = self._prepare_model()

        if getattr(opts, "ddp.use_deprecated_data_parallel"):
            logger.warning(
                "DataParallel is not recommended for training, and is not tested exhaustively. \
                    Please use it only for debugging purposes. We will deprecated the support for DataParallel in future and \
                        encourage you to use DistributedDataParallel."
            )
            model = model.to(device=torch.device("cpu"))
            model = torch.nn.DataParallel(model)
            model = model.to(device=device)
        elif use_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[dev_id],
                output_device=dev_id,
                find_unused_parameters=getattr(opts, "ddp.find_unused_params"),
            )
            if is_master_node:
                logger.log("Using DistributedDataParallel.")

        if wrapper_cls_for_act_ckpt is not None:
            activation_checkpointing(
                model=model, submodule_class=wrapper_cls_for_act_ckpt
            )
        return model

    @cached_property
    def criteria(self) -> BaseCriteria:
        opts = self.opts
        device = self.device
        is_master_node = self.is_master_node
        criteria = build_loss_fn(opts)
        if is_master_node:
            logger.log(logger.color_text("Loss function"))
            print(criteria)
        criteria = criteria.to(device=device)
        return criteria

    @cached_property
    def optimizer(self) -> BaseOptim:
        opts = self.opts
        model = self.model
        is_master_node = self.is_master_node

        optimizer = build_optimizer(model, opts=opts)
        if is_master_node:
            logger.log(logger.color_text("Optimizer"))
            print(optimizer)
        return optimizer

    @cached_property
    def gradient_scaler(self) -> GradScaler:
        opts = self.opts
        enable_grad_scaler = (
            getattr(opts, "common.mixed_precision")
            and getattr(opts, "common.mixed_precision_dtype") == "float16"
        )
        return GradScaler(enabled=enable_grad_scaler)

    @cached_property
    def launcher(self) -> Callable[[Callback], None]:
        """
        Creates the entrypoints that spawn training and evaluation subprocesses.

        The number of subprocesses depend on the number of gpus and distributed nodes.

        Returns a function that once called, spawns as many subprocesses as needed for
        training or evaluation. The returned function accepts a Callback as an argument.
        The Callback will be invoked on each subprocess.
        """
        opts = self.opts
        opts = device_setup(opts)
        is_master_node = self.is_master_node

        # create the directory for saving results
        save_dir = getattr(opts, "common.results_loc")
        run_label = getattr(opts, "common.run_label")
        exp_dir = "{}/{}".format(save_dir, run_label)
        setattr(opts, "common.exp_loc", exp_dir)
        create_directories(dir_path=exp_dir, is_master_node=is_master_node)

        num_gpus = getattr(opts, "dev.num_gpus")

        use_deprecated_data_parallel = getattr(opts, "ddp.use_deprecated_data_parallel")
        use_distributed = num_gpus >= 1 and not use_deprecated_data_parallel
        setattr(opts, "ddp.use_distributed", use_distributed)

        if num_gpus > 0:
            assert torch.cuda.is_available(), "We need CUDA for training on GPUs."

        # No of data workers = no of CPUs (if not specified or -1)
        n_cpus = resources.cpu_count()
        dataset_workers = getattr(opts, "dataset.workers")

        num_gpus_ge_1 = max(1, num_gpus)

        if not use_distributed:
            if dataset_workers == -1:
                logger.log(f"Setting dataset.workers to {n_cpus}.")
                setattr(opts, "dataset.workers", n_cpus)

            # adjust the batch size
            train_bsize = getattr(opts, "dataset.train_batch_size0") * num_gpus_ge_1
            val_bsize = getattr(opts, "dataset.val_batch_size0") * num_gpus_ge_1
            setattr(opts, "dataset.train_batch_size0", train_bsize)
            setattr(opts, "dataset.val_batch_size0", val_bsize)
            setattr(opts, "dev.device_id", None)
            return lambda callback: callback(self)

        else:
            # DDP is the default for training
            # get device id
            dev_id = getattr(opts, "ddp.device_id")
            # set the dev.device_id to the same as ddp.device_id.
            # note that dev arguments are not accessible through CLI.
            setattr(opts, "dev.device_id", dev_id)

            if dataset_workers == -1 or dataset_workers is None:
                logger.log(f"Setting dataset.workers to {n_cpus // num_gpus_ge_1}.")
                setattr(opts, "dataset.workers", n_cpus // num_gpus_ge_1)

            start_rank = getattr(opts, "ddp.rank")
            # we need to set rank to None as it is reset inside the _launcher_distributed_spawn_fn function
            setattr(opts, "ddp.rank", None)
            setattr(opts, "ddp.start_rank", start_rank)
            return lambda callback: torch.multiprocessing.spawn(
                fn=self._launcher_distributed_spawn_fn,
                args=(callback, self),
                nprocs=num_gpus_ge_1,
            )

    @cached_property
    def model_ema(self) -> Optional[EMA]:
        opts = self.opts
        device = self.device
        model = self.model
        is_master_node = self.is_master_node

        model_ema = None
        use_ema = getattr(opts, "ema.enable")

        if use_ema:
            ema_momentum = getattr(opts, "ema.momentum")
            model_ema = EMA(model=model, ema_momentum=ema_momentum, device=device)
            if is_master_node:
                logger.log("Using EMA")
        return model_ema

    @cached_property
    def training_engine(self) -> DefaultTrainer:
        opts = self.opts
        is_master_node = self.is_master_node
        train_loader, val_loader = self.train_val_loader

        model = self.model
        criteria = self.criteria
        optimizer = self.optimizer
        gradient_scaler = self.gradient_scaler
        scheduler = self.scheduler

        model_ema = self.model_ema
        best_metric = (
            -math.inf if getattr(opts, "stats.checkpoint_metric_max") else math.inf
        )

        start_epoch = 0
        start_iteration = 0
        resume_loc = getattr(opts, "common.resume")
        finetune_loc = getattr(opts, "common.finetune")
        auto_resume = getattr(opts, "common.auto_resume")
        if resume_loc is not None or auto_resume:
            (
                model,
                optimizer,
                gradient_scaler,
                start_epoch,
                start_iteration,
                best_metric,
                model_ema,
            ) = load_checkpoint(
                opts=opts,
                model=model,
                optimizer=optimizer,
                model_ema=model_ema,
                gradient_scaler=gradient_scaler,
            )
        elif finetune_loc is not None:
            model, model_ema = load_model_state(
                opts=opts, model=model, model_ema=model_ema
            )
            if is_master_node:
                logger.log("Finetuning model from checkpoint {}".format(finetune_loc))

        training_engine = DefaultTrainer(
            opts=opts,
            model=model,
            validation_loader=val_loader,
            training_loader=train_loader,
            optimizer=optimizer,
            criteria=criteria,
            scheduler=scheduler,
            start_epoch=start_epoch,
            start_iteration=start_iteration,
            best_metric=best_metric,
            model_ema=model_ema,
            gradient_scaler=gradient_scaler,
        )
        return training_engine

    @cached_property
    def evaluation_engine(self) -> Evaluator:
        opts = self.opts
        test_loader = self.test_loader
        model = self.model
        criteria = self.criteria
        return Evaluator(
            opts=opts, model=model, test_loader=test_loader, criteria=criteria
        )

    @staticmethod
    def _launcher_distributed_spawn_fn(
        device_id: int,
        callback: Callback,
        train_eval_pipeline: DefaultTrainEvalPipeline,
    ) -> None:
        """
        Wraps a callback function for `torch.multiprocessing.spawn` to spawn DDP workers. The rank information will be set in `opts` before the wrapped callback is invoked.

        Args:
            device_id: GPU device number.
            callback: The wrapped callback function to be invoked after the rank information are set in `opts`.
            train_eval_pipeline: The instance of TrainEvalPipeline that will be passed as the only input argument to `callback`.

        """
        opts = train_eval_pipeline.opts
        setattr(opts, "dev.device_id", device_id)
        torch.cuda.set_device(device_id)
        setattr(opts, "dev.device", torch.device(f"cuda:{device_id}"))

        ddp_rank = getattr(opts, "ddp.rank", None)
        if ddp_rank is None:
            ddp_rank = getattr(opts, "ddp.start_rank", 0) + device_id
            setattr(opts, "ddp.rank", ddp_rank)

        node_rank = distributed_init(opts)
        setattr(opts, "ddp.rank", node_rank)
        callback(train_eval_pipeline)
