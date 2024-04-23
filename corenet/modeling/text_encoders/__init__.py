#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.modeling.text_encoders.base_text_encoder import BaseTextEncoder
from corenet.utils import logger
from corenet.utils.registry import Registry

TEXT_ENCODER_REGISTRY = Registry(
    "text_encoder",
    base_class=BaseTextEncoder,
    lazy_load_dirs=["corenet/modeling/text_encoders"],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def arguments_text_encoder(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register arguments of all text encoders."""
    # add arguments for text_encoder
    parser = BaseTextEncoder.add_arguments(parser)

    # add class specific arguments
    parser = TEXT_ENCODER_REGISTRY.all_arguments(parser)
    return parser


def build_text_encoder(opts, projection_dim: int, *args, **kwargs) -> BaseTextEncoder:
    """Helper function to build the text encoder from command-line arguments.

    Args:
        opts: Command-line arguments
        projection_dim: The dimensionality of the projection head after text encoder.

    Returns:
        Text encoder module.
    """
    text_encoder_name = getattr(opts, "model.text.name")

    # We registered the base class using a special `name` (i.e., `__base__`)
    # in order to access the arguments defined inside those classes. However, these classes are not supposed to
    # be used. Therefore, we raise an error for such cases
    if text_encoder_name == "__base__":
        logger.error("__base__ can't be used as a projection name. Please check.")

    text_encoder = TEXT_ENCODER_REGISTRY[text_encoder_name](
        opts, projection_dim, *args, **kwargs
    )
    return text_encoder
