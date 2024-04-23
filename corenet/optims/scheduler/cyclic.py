#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import math

import numpy as np

from corenet.optims.scheduler import SCHEDULER_REGISTRY
from corenet.optims.scheduler.base_scheduler import BaseLRScheduler
from corenet.utils import logger

SUPPORTED_LAST_CYCLES = ["cosine", "linear"]


@SCHEDULER_REGISTRY.register("cyclic")
class CyclicLRScheduler(BaseLRScheduler):
    """
    Cyclic LR: https://arxiv.org/abs/1811.11431
    """

    def __init__(self, opts, **kwargs) -> None:

        cycle_steps = getattr(opts, "scheduler.cyclic.steps", [25])
        if cycle_steps is not None and isinstance(cycle_steps, int):
            cycle_steps = [cycle_steps]
        gamma = getattr(opts, "scheduler.cyclic.gamma", 0.5)
        anneal_type = getattr(opts, "scheduler.cyclic.last_cycle_type", "linear")
        min_lr = getattr(opts, "scheduler.cyclic.min_lr", 0.1)
        end_lr = getattr(opts, "scheduler.cyclic.last_cycle_end_lr", 1e-3)
        ep_per_cycle = getattr(opts, "scheduler.cyclic.epochs_per_cycle", 5)
        warmup_iterations = getattr(opts, "scheduler.warmup_iterations", 0)
        n_cycles = getattr(opts, "scheduler.cyclic.total_cycles", 10) - 1
        max_epochs = getattr(opts, "scheduler.max_epochs", 100)

        if anneal_type not in SUPPORTED_LAST_CYCLES:
            logger.error(
                "Supported anneal types for {} are: {}".format(
                    self.__class__.__name__, SUPPORTED_LAST_CYCLES
                )
            )
        if min_lr < end_lr:
            logger.error(
                "Min LR should be greater than end LR. Got: {} and {}".format(
                    min_lr, end_lr
                )
            )

        super(CyclicLRScheduler, self).__init__(opts=opts)
        self.min_lr = min_lr
        self.cycle_length = ep_per_cycle
        self.end_lr = end_lr
        self.max_lr = self.min_lr * self.cycle_length
        self.last_cycle_anneal_type = anneal_type

        if self.warmup_iterations > 0:
            self.warmup_step = (
                self.min_lr - self.warmup_init_lr
            ) / self.warmup_iterations

        self.n_cycles = n_cycles

        self.cyclic_epochs = self.cycle_length * self.n_cycles
        self.max_epochs = max_epochs
        self.last_cycle_epochs = self.max_epochs - self.cyclic_epochs

        assert self.max_epochs == self.cyclic_epochs + self.last_cycle_epochs

        self.steps = [self.max_epochs] if cycle_steps is None else cycle_steps
        self.gamma = gamma if cycle_steps is not None else 1

        self._lr_per_cycle()
        self.epochs_lr_stepped = []

    def _lr_per_cycle(self) -> None:
        lrs = list(
            np.linspace(self.max_lr, self.min_lr, self.cycle_length, dtype=np.float32)
        )
        lrs = [lrs[-1]] + lrs[:-1]
        self.cycle_lrs = lrs

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(
            title="Cyclic LR arguments", description="Cyclic LR arguments"
        )
        group.add_argument(
            "--scheduler.cyclic.min-lr",
            default=0.1,
            type=float,
            help="Min. lr for a cycle",
        )
        group.add_argument(
            "--scheduler.cyclic.last-cycle-end-lr",
            default=1e-3,
            type=float,
            help="End LR for the last cycle",
        )
        group.add_argument(
            "--scheduler.cyclic.total-cycles",
            default=11,
            type=int,
            help="Number of cycles. Default is 10",
        )
        group.add_argument(
            "--scheduler.cyclic.epochs-per-cycle",
            default=5,
            type=int,
            help="Number of epochs per cycle. Default is 5",
        )
        group.add_argument(
            "--scheduler.cyclic.steps",
            default=None,
            type=int,
            nargs="+",
            help="steps at which LR should be decreased",
        )
        group.add_argument(
            "--scheduler.cyclic.gamma",
            default=0.5,
            type=float,
            help="Factor by which LR should be decreased",
        )
        group.add_argument(
            "--scheduler.cyclic.last-cycle-type",
            default="linear",
            type=str,
            choices=SUPPORTED_LAST_CYCLES,
            help="Annealing in last cycle",
        )
        return parser

    def get_lr(self, epoch: int, curr_iter: int) -> float:
        if curr_iter < self.warmup_iterations:
            curr_lr = self.warmup_init_lr + curr_iter * self.warmup_step
        else:
            if epoch <= self.cyclic_epochs:
                if epoch in self.steps and epoch not in self.epochs_lr_stepped:
                    self.min_lr *= self.gamma ** (self.steps.index(epoch) + 1)
                    self.max_lr *= self.gamma ** (self.steps.index(epoch) + 1)
                    self._lr_per_cycle()
                    self.epochs_lr_stepped.append(epoch)
                idx = epoch % self.cycle_length
                curr_lr = self.cycle_lrs[idx]
            else:
                base_lr = self.min_lr
                if self.last_cycle_anneal_type == "linear":
                    lr_step = (base_lr - self.end_lr) / self.last_cycle_epochs
                    curr_lr = base_lr - (epoch - self.cyclic_epochs + 1) * lr_step
                elif self.last_cycle_anneal_type == "cosine":
                    curr_epoch = epoch - self.cyclic_epochs
                    period = self.max_epochs - self.cyclic_epochs + 1
                    curr_lr = self.end_lr + 0.5 * (base_lr - self.end_lr) * (
                        1 + math.cos(math.pi * curr_epoch / period)
                    )
                else:
                    raise NotImplementedError
        return max(0.0, curr_lr)

    def __repr__(self):
        repr_str = (
            "{}(\n \t C={},\n \t C_length={},\n \t C_last={},\n \t Total_Epochs={}, "
            "\n \t steps={},\n \t gamma={},\n \t last_cycle_anneal_method={} "
            "\n \t min_lr={}, \n\t max_lr={}, \n\t end_lr={}\n)".format(
                self.__class__.__name__,
                self.n_cycles,
                self.cycle_length,
                self.last_cycle_epochs,
                self.max_epochs,
                self.steps,
                self.gamma,
                self.last_cycle_anneal_type,
                self.min_lr,
                self.min_lr * self.cycle_length,
                self.end_lr,
            )
        )
        return repr_str
