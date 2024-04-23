#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Optional

from corenet.loss_fn.base_criteria import BaseCriteria
from corenet.utils import logger
from corenet.utils.registry import Registry

# Registry for loss functions.
LOSS_REGISTRY = Registry(
    registry_name="loss_functions",
    base_class=BaseCriteria,
    lazy_load_dirs=["corenet/loss_fn"],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def build_loss_fn(
    opts: argparse.Namespace, category: Optional[str] = "", *args, **kwargs
) -> BaseCriteria:
    """Helper function to build loss function from command-line arguments.

    Args:
        opts: command-line arguments
        category: Optional task category (e.g., classification). Specifying category may be useful for
            building composite loss functions. See `loss_fns.composite_loss.CompositeLoss.build_composite_loss_fn`
            function for an example

    Returns:
        Loss function module
    """

    if not category:
        # If category is not specified, then read it from command-line arguments
        category = getattr(opts, "loss.category")

    if category is None:
        logger.error(
            "Please specify loss name using --loss.category. For composite loss function, see configuration"
            "example in `loss_fns.composite_loss.CompositeLoss`. Got None"
        )

    # Get the name of loss function for a given category.
    # Note that loss functions (e.g., NeuralAugmentation) that are not task-specific does not have this
    # argument defined. In such case, we set the loss function name the same as category
    if hasattr(opts, f"loss.{category}.name"):
        loss_fn_name = getattr(opts, f"loss.{category}.name")
    else:
        loss_fn_name = category

    # We registered the base criteria classes for different categories using a special `name` (i.e., `__base__`)
    # in order to access the arguments defined inside those classes. However, these classes are not supposed to
    # be used as a loss function. Therefore, we raise an error for such cases
    if loss_fn_name == "__base__":
        logger.error("__base__ can't be used as a loss function name. Please check.")

    loss_fn = LOSS_REGISTRY[loss_fn_name, category](opts, *args, **kwargs)
    return loss_fn


def add_loss_fn_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """This method gets a parser object, and for every loss that is registered in the
    LOSS_REGISTRY adds its arguments to it."""
    parser = BaseCriteria.add_arguments(parser=parser)

    parser = LOSS_REGISTRY.all_arguments(parser)
    return parser
