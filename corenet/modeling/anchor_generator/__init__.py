#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.modeling.anchor_generator.base_anchor_generator import BaseAnchorGenerator
from corenet.utils import logger
from corenet.utils.registry import Registry

# register anchor generator
ANCHOR_GEN_REGISTRY = Registry(
    "anchor_gen",
    base_class=BaseAnchorGenerator,
    lazy_load_dirs=["corenet/modeling/anchor_generator"],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def arguments_anchor_gen(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments related to anchor generator for object detection"""
    group = parser.add_argument_group("Anchor generator", "Anchor generator")
    group.add_argument(
        "--anchor-generator.name", type=str, help="Name of the anchor generator"
    )

    # add class specific arguments
    parser = ANCHOR_GEN_REGISTRY.all_arguments(parser)
    return parser


def build_anchor_generator(opts, *args, **kwargs):
    """Build anchor generator for object detection"""
    anchor_gen_name = getattr(opts, "anchor_generator.name")

    # We registered the base class using a special `name` (i.e., `__base__`)
    # in order to access the arguments defined inside those classes. However, these classes are not supposed to
    # be used. Therefore, we raise an error for such cases
    if anchor_gen_name == "__base__":
        logger.error("__base__ can't be used as a projection name. Please check.")

    anchor_gen = ANCHOR_GEN_REGISTRY[anchor_gen_name](opts, *args, **kwargs)
    return anchor_gen
