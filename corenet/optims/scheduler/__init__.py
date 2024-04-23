#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.optims.scheduler.base_scheduler import BaseLRScheduler
from corenet.utils import logger
from corenet.utils.registry import Registry

SCHEDULER_REGISTRY = Registry(
    "scheduler",
    base_class=BaseLRScheduler,
    lazy_load_dirs=["corenet/optims/scheduler"],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def build_scheduler(opts: argparse.Namespace, *args, **kwargs) -> BaseLRScheduler:
    scheduler_name = getattr(opts, "scheduler.name").lower()

    # We registered the base class using a special `name` (i.e., `__base__`)
    # in order to access the arguments defined inside those classes. However, these classes are not supposed to
    # be used. Therefore, we raise an error for such cases
    if scheduler_name == "__base__":
        logger.error("__base__ can't be used as a projection name. Please check.")

    scheduler = SCHEDULER_REGISTRY[scheduler_name](opts, *args, **kwargs)
    return scheduler


def general_lr_sch_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        title="LR scheduler arguments", description="LR scheduler arguments"
    )

    group.add_argument(
        "--scheduler.name", type=str, default="cosine", help="LR scheduler name"
    )
    group.add_argument("--scheduler.lr", type=float, default=0.1, help="Learning rate")
    group.add_argument(
        "--scheduler.max-epochs",
        type=int,
        default=None,
        help="Max. epochs for training",
    )
    group.add_argument(
        "--scheduler.max-iterations",
        type=int,
        default=None,
        help="Max. iterations for training",
    )
    group.add_argument(
        "--scheduler.warmup-iterations",
        type=int,
        default=None,
        help="Warm-up iterations",
    )
    group.add_argument(
        "--scheduler.warmup-init-lr", type=float, default=1e-7, help="Warm-up init lr"
    )
    group.add_argument(
        "--scheduler.is-iteration-based",
        action="store_true",
        help="Is iteration type or epoch type",
    )

    group.add_argument(
        "--scheduler.adjust-period-for-epochs",
        action="store_true",
        help="Adjust the period for epoch-based scheduler.",
    )

    return parser


def arguments_scheduler(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = general_lr_sch_args(parser=parser)

    # add scheduler specific arguments
    parser = SCHEDULER_REGISTRY.all_arguments(parser)
    return parser
