#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#


import argparse

from corenet.modeling.image_projection_layers.base_image_projection import (
    BaseImageProjectionHead,
)
from corenet.utils import logger
from corenet.utils.registry import Registry

IMAGE_PROJECTION_HEAD_REGISTRY = Registry(
    "image_projection_head",
    base_class=BaseImageProjectionHead,
    lazy_load_dirs=["corenet/modeling/image_projection_layers"],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def arguments_image_projection_head(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Register arguments of all image projection heads."""
    # add arguments for base image projection layer
    parser = BaseImageProjectionHead.add_arguments(parser)

    # add class specific arguments
    parser = IMAGE_PROJECTION_HEAD_REGISTRY.all_arguments(parser)
    return parser


def build_image_projection_head(
    opts: argparse.Namespace, in_dim: int, out_dim: int, *args, **kwargs
) -> BaseImageProjectionHead:
    """Helper function to build an image projection head from command-line arguments.

    Args:
        opts: Command-line arguments
        in_dim: Input dimension to the projection head.
        out_dim: Output dimension of the projection head.

    Returns:
        Image projection head module.
    """

    # Get the name of image projection head
    image_projection_head_name = getattr(opts, "model.image_projection_head.name")

    # We registered the base class using a special `name` (i.e., `__base__`)
    # in order to access the arguments defined inside those classes. However, these classes are not supposed to
    # be used. Therefore, we raise an error for such cases
    if image_projection_head_name == "__base__":
        logger.error("__base__ can't be used as a projection name. Please check.")

    image_projection_head = IMAGE_PROJECTION_HEAD_REGISTRY[image_projection_head_name](
        opts, in_dim, out_dim, *args, **kwargs
    )
    return image_projection_head
