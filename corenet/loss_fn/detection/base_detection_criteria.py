#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse

from corenet.loss_fn import LOSS_REGISTRY, BaseCriteria


@LOSS_REGISTRY.register(name="__base__", type="detection")
class BaseDetectionCriteria(BaseCriteria):
    """Base class for defining detection loss functions. Sub-classes must implement forward function.

    Args:
        opts: command line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BaseDetectionCriteria:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--loss.detection.name",
            type=str,
            default=None,
            help=f"Name of the loss function in {cls.__name__}. Defaults to None.",
        )
        return parser
