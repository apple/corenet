#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import os
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor
from torch.cuda.amp import autocast

from corenet.utils import logger
from corenet.utils.common_utils import create_directories
from corenet.utils.ddp_utils import is_master
from corenet.utils.file_logger import FileLogger

str_to_torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def autocast_fn(enabled: bool, amp_precision: Optional[str] = "float16"):
    if enabled:
        # If AMP is enabled, ensure that:
        # 1. Device is CUDA
        # 2. dtype is FLOAT16 or BFLOAT16
        if amp_precision not in str_to_torch_dtype:
            logger.error(
                "For Mixed-precision training, supported dtypes are {}. Got: {}".format(
                    list(str_to_torch_dtype.keys()), amp_precision
                )
            )

        if not torch.cuda.is_available():
            logger.error("For mixed-precision training, CUDA device is required.")

        return autocast(enabled=enabled, dtype=str_to_torch_dtype[amp_precision])
    else:
        return autocast(enabled=False)


def get_batch_size(x: Union[Tensor, Dict, List]) -> int:
    if isinstance(x, Tensor):
        return x.shape[0]
    elif isinstance(x, Dict):
        for key in ("image", "video", "audio"):
            if key in x:
                return get_batch_size(x[key])
        raise NotImplementedError(f"Invalid dict keys {x.keys()}")
    elif isinstance(x, List):
        return len(x)
    else:
        raise NotImplementedError(f"Invalid type {type(x)}")


def log_metrics(
    lrs: Union[List, float],
    log_writer,
    train_loss: float,
    val_loss: float,
    epoch: int,
    best_metric: float,
    val_ema_loss: Optional[float] = None,
    ckpt_metric_name: Optional[str] = None,
    train_ckpt_metric: Optional[float] = None,
    val_ckpt_metric: Optional[float] = None,
    val_ema_ckpt_metric: Optional[float] = None,
) -> None:
    if not isinstance(lrs, list):
        lrs = [lrs]
    for g_id, lr_val in enumerate(lrs):
        log_writer.add_scalar("LR/Group-{}".format(g_id), round(lr_val, 6), epoch)

    log_writer.add_scalar("Common/Best Metric", round(best_metric, 2), epoch)


def get_log_writers(opts: argparse.Namespace, save_location: Optional[str]):
    is_master_node = is_master(opts)

    log_writers = []
    if not is_master_node:
        return log_writers

    tensorboard_logging = getattr(opts, "common.tensorboard_logging", False)
    if tensorboard_logging and save_location is not None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            logger.log(
                "Unable to import SummaryWriter from torch.utils.tensorboard. Disabling tensorboard logging"
            )
            SummaryWriter = None

        if SummaryWriter is not None:
            exp_dir = "{}/tb_logs".format(save_location)
            create_directories(dir_path=exp_dir, is_master_node=is_master_node)
            log_writers.append(
                SummaryWriter(log_dir=exp_dir, comment="Training and Validation logs")
            )

    bolt_logging = getattr(opts, "common.bolt_logging", False)
    if bolt_logging:
        try:
            from corenet.internal.utils.bolt_logger import BoltLogger
        except ModuleNotFoundError:
            BoltLogger = None

        if BoltLogger is None:
            logger.log("Unable to import bolt. Disabling bolt logging")
        else:
            log_writers.append(BoltLogger())

    hub_logging = getattr(opts, "common.hub.logging", False)
    if hub_logging:
        try:
            from corenet.internal.utils.hub_logger import HubLogger
        except ModuleNotFoundError:
            HubLogger = None

        if HubLogger is None:
            logger.log("Unable to import hub. Disabling hub logging")
        else:
            try:
                hub_logger = HubLogger(opts)
            except Exception as ex:
                logger.log(
                    f"Unable to initialize hub logger. Disabling hub logging: {ex}"
                )
                hub_logger = None
            if hub_logger is not None:
                log_writers.append(hub_logger)

    file_logging = getattr(opts, "common.file_logging")
    if file_logging and save_location is not None:
        log_writers.append(FileLogger(os.path.join(save_location, "stats.pt")))

    return log_writers
