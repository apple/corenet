#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.utils import logger


class BaseLRScheduler(object):
    def __init__(self, opts) -> None:
        warmup_iterations = getattr(opts, "scheduler.warmup_iterations", None)
        super().__init__()
        self.opts = opts
        self.round_places = 8
        self.lr_multipliers = getattr(opts, "optim.lr_multipliers", None)

        self.warmup_iterations = (
            max(warmup_iterations, 0) if warmup_iterations is not None else 0
        )

        warmup_init_lr = getattr(opts, "scheduler.warmup_init_lr", 1e-7)
        self.warmup_init_lr = warmup_init_lr

        # Because of variable batch sizes, we can't determine exact number of epochs in warm-up phase. This
        # may result in different LR schedules when we run epoch- and iteration-based schedulers.
        # To reduce these differences, we use adjust_period_for_epochs arguments.
        # For epoch-based scheduler, this parameter value should be enabled.
        self.adjust_period = getattr(opts, "scheduler.adjust_period_for_epochs", False)
        self.warmup_epochs = 0

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def get_lr(self, epoch: int, curr_iter: int):
        raise NotImplementedError

    def update_lr(self, optimizer, epoch: int, curr_iter: int):
        lr = self.get_lr(epoch=epoch, curr_iter=curr_iter)
        lr = max(0.0, lr)
        if self.lr_multipliers is not None:
            assert len(self.lr_multipliers) == len(optimizer.param_groups)
            for g_id, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = round(
                    lr * self.lr_multipliers[g_id], self.round_places
                )
        else:
            for param_group in optimizer.param_groups:
                param_group["lr"] = round(lr, self.round_places)
        return optimizer

    @staticmethod
    def retrieve_lr(optimizer) -> list:
        lr_list = []
        for param_group in optimizer.param_groups:
            lr_list.append(param_group["lr"])
        return lr_list

    def extra_repr(self) -> str:
        """Extra information to be represented in __repr__. Each line in the output
        string should be prefixed with ``\n\t``."""
        return ""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()}\n)"
