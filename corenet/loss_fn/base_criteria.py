#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import abc
import argparse
from typing import Any

from torch import nn

from corenet.utils import logger


class BaseCriteria(nn.Module, abc.ABC):
    """Base class for defining loss functions. Sub-classes must implement compute_loss function.

    Args:
        opts: command line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super(BaseCriteria, self).__init__()
        self.opts = opts
        # small value for numerical stability purposes that sub-classes may want to use.
        self.eps = 1e-7

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add criteria-specific arguments to the parser."""
        if cls != BaseCriteria:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--loss.category",
            type=str,
            default=None,
            help="Loss function category (e.g., classification). Defaults to None.",
        )
        return parser

    @abc.abstractmethod
    def forward(
        self, input_sample: Any, prediction: Any, target: Any, *args, **kwargs
    ) -> Any:
        """Compute the loss.

        Args:
            input_sample: Input to the model.
            prediction: Model's output
            target: Ground truth labels
        """
        raise NotImplementedError

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return "{}({}\n)".format(self.__class__.__name__, self.extra_repr())
