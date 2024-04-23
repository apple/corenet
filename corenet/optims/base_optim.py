#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.utils import logger


class BaseOptim(object):
    """Base class for optimizer

    Args:
        opts: Command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        self.eps = 1e-8
        self.lr = getattr(opts, "scheduler.lr")
        self.weight_decay = getattr(opts, "optim.weight_decay")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add optimizer arguments"""
        if cls != BaseOptim:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--optim.name",
            type=str,
            default="sgd",
            help="Name of the optimizer. Defaults to SGD.",
        )
        group.add_argument(
            "--optim.eps",
            type=float,
            default=1e-8,
            help="Optimizer epsilon value. Defaults to 1.e-8.",
        )
        group.add_argument(
            "--optim.weight-decay",
            default=4e-5,
            type=float,
            help="Weight decay (or L2 penalty). Defaults to 4.e-5.",
        )
        group.add_argument(
            "--optim.no-decay-bn-filter-bias",
            action="store_true",
            default=False,
            help="When enabled, the weight in normalization layers and biases in the model are not decayed."
            "Defaults to False.",
        )
        group.add_argument(
            "--optim.bypass-parameters-check",
            action="store_true",
            default=False,
            help="Bypass parameter check when creating optimizer. Defaults to False",
        )
        return parser

    def __repr__(self) -> str:
        group_dict = dict()
        for i, group in enumerate(self.param_groups):
            for key in sorted(group.keys()):
                if key == "params":
                    continue
                if key not in group_dict:
                    group_dict[key] = [group[key]]
                else:
                    group_dict[key].append(group[key])

        format_string = self.__class__.__name__ + " ("
        format_string += "\n"
        for k, v in group_dict.items():
            format_string += "\t {0}: {1}\n".format(k, v)
        format_string += ")"
        return format_string
