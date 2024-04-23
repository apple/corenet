#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.optims.scheduler import SCHEDULER_REGISTRY
from corenet.optims.scheduler.base_scheduler import BaseLRScheduler


@SCHEDULER_REGISTRY.register("multi_step")
class MultiStepLRScheduler(BaseLRScheduler):
    """
    Multi-step learning rate scheduler with optional linear warm-up strategy
    """

    def __init__(self, opts, **kwargs) -> None:
        is_iter_based = getattr(opts, "scheduler.is_iteration_based", True)
        super().__init__(opts=opts)

        max_iterations = getattr(opts, "scheduler.max_iterations", 150000)

        self.lr = getattr(opts, "scheduler.multi_step.lr", None)
        assert self.lr is not None

        if self.warmup_iterations > 0:
            self.warmup_step = (self.lr - self.warmup_init_lr) / self.warmup_iterations

        milestones = getattr(opts, "scheduler.multi_step.milestones", None)
        if milestones is None:
            milestones = [-1]
        elif isinstance(milestones, int):
            milestones = [milestones]

        self.milestones = sorted(
            list(set(milestones))
        )  # remove duplicates and sort them
        self.gamma = getattr(opts, "scheduler.multi_step.gamma", 1.0)

        self.period = (
            max_iterations - self.warmup_iterations + 1
            if is_iter_based
            else getattr(opts, "scheduler.max_epochs", 350)
        )

        self.is_iter_based = is_iter_based

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="{} arguments".format(cls.__name__),
            description="{} arguments".format(cls.__name__),
        )

        group.add_argument(
            "--scheduler.multi-step.lr", type=float, default=0.1, help="LR value"
        )
        group.add_argument(
            "--scheduler.multi-step.gamma",
            type=float,
            default=None,
            help="Decay LR value by this factor",
        )
        group.add_argument(
            "--scheduler.multi-step.milestones",
            type=int,
            nargs="+",
            default=None,
            help="Decay LR value at these epoch",
        )
        return parser

    def get_lr(self, epoch: int, curr_iter: int) -> float:
        if curr_iter < self.warmup_iterations:
            return max(0.0, self.warmup_init_lr + curr_iter * self.warmup_step)
        else:
            if epoch in self.milestones:
                self.lr = self.lr * self.gamma
                self.milestones.remove(epoch)
            return max(0.0, self.lr)

    def __repr__(self) -> str:
        repr_str = "{}(".format(self.__class__.__name__)
        repr_str += "\n\tlr={}\n\tmilestones={}\n\tgamma={}".format(
            self.lr, self.milestones, self.gamma
        )
        if self.warmup_iterations > 0:
            repr_str += "\n\twarmup_init_lr={}\n\twarmup_iters={}".format(
                self.warmup_init_lr, self.warmup_iterations
            )

        repr_str += "\n )"
        return repr_str
