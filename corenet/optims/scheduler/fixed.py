#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.optims.scheduler import SCHEDULER_REGISTRY
from corenet.optims.scheduler.base_scheduler import BaseLRScheduler


@SCHEDULER_REGISTRY.register("fixed")
class FixedLRScheduler(BaseLRScheduler):
    """
    Fixed learning rate scheduler with optional linear warm-up strategy
    """

    def __init__(self, opts, **kwargs) -> None:
        is_iter_based = getattr(opts, "scheduler.is_iteration_based", True)
        super(FixedLRScheduler, self).__init__(opts=opts)

        max_iterations = getattr(opts, "scheduler.max_iterations", 150000)

        self.fixed_lr = getattr(opts, "scheduler.fixed.lr", None)
        assert self.fixed_lr is not None

        if self.warmup_iterations > 0:
            self.warmup_step = (
                self.fixed_lr - self.warmup_init_lr
            ) / self.warmup_iterations

        self.period = (
            max_iterations - self.warmup_iterations + 1
            if is_iter_based
            else getattr(opts, "scheduler.max_epochs", 350)
        )

        self.is_iter_based = is_iter_based

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="Fixed LR arguments", description="Fixed LR arguments"
        )

        group.add_argument(
            "--scheduler.fixed.lr", type=float, default=None, help="LR value"
        )

        return parser

    def get_lr(self, epoch: int, curr_iter: int) -> float:
        if curr_iter < self.warmup_iterations:
            curr_lr = self.warmup_init_lr + curr_iter * self.warmup_step
        else:
            curr_lr = self.fixed_lr
        return max(0.0, curr_lr)

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)
        repr_str += "\n\tlr={}".format(self.fixed_lr)
        if self.warmup_iterations > 0:
            repr_str += "\n\twarmup_init_lr={}\n\twarmup_iters={}".format(
                self.warmup_init_lr, self.warmup_iterations
            )

        repr_str += "\n )"
        return repr_str
