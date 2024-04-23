#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import math
from functools import cached_property

import torch
from torch.cuda.amp import GradScaler

from corenet.engine.fsdp_trainer import FSDPTrainer
from corenet.modeling.models.fsdp_wrapper import FullyShardedDataParallelWrapper
from corenet.train_eval_pipelines.default_train_eval import (
    TRAIN_EVAL_PIPELINE_REGISTRY,
    DefaultTrainEvalPipeline,
)
from corenet.utils import logger
from corenet.utils.activation_checkpointing_wrapper import activation_checkpointing
from corenet.utils.checkpoint_utils import load_checkpoint, load_model_state


@TRAIN_EVAL_PIPELINE_REGISTRY.register("fsdp_train_eval_pipeline")
class FSDPTrainEvalPipeline(DefaultTrainEvalPipeline):
    """FSDPTrainEvalPipeline class is responsible for instantiating the components of
    training, evaluation, and/or pipelines that use FSDP.

    Args:
        opts: Commandline options.
    """

    @cached_property
    def gradient_scaler(self) -> GradScaler:
        raise NotImplementedError(
            "FSDP does not use autocast for mixed-precision training and handles it internally."
        )

    @cached_property
    def model(self) -> torch.nn.Module:
        opts = self.opts
        use_distributed = getattr(opts, "ddp.use_distributed")
        assert use_distributed, "DDP needs to be enabled when using FSDP"

        model, wrapper_cls_for_act_ckpt = self._prepare_model()

        fsdp_model = FullyShardedDataParallelWrapper(opts=self.opts, model=model)
        if wrapper_cls_for_act_ckpt is not None:
            activation_checkpointing(
                model=fsdp_model, submodule_class=wrapper_cls_for_act_ckpt
            )

        return fsdp_model

    @cached_property
    def training_engine(self) -> FSDPTrainer:
        opts = self.opts
        is_master_node = self.is_master_node
        train_loader, val_loader = self.train_val_loader

        model = self.model
        criteria = self.criteria
        optimizer = self.optimizer
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
            )
        elif finetune_loc is not None:
            model, model_ema = load_model_state(
                opts=opts, model=model, model_ema=model_ema
            )
            if is_master_node:
                logger.log("Finetuning model from checkpoint {}".format(finetune_loc))

        training_engine = FSDPTrainer(
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
        )
        return training_engine
