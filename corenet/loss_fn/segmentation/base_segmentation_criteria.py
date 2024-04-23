#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse

from corenet.loss_fn import LOSS_REGISTRY, BaseCriteria


@LOSS_REGISTRY.register(name="__base__", type="segmentation")
class BaseSegmentationCriteria(BaseCriteria):
    """Base class for defining segmentation loss functions. Sub-classes must implement forward function.

    Args:
        opts: command line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BaseSegmentationCriteria:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--loss.segmentation.name",
            type=str,
            default=None,
            help="Name of the loss function. Defaults to None.",
        )
        return parser
