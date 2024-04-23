#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.modeling.matcher_det.base_matcher import BaseMatcher
from corenet.utils import logger
from corenet.utils.registry import Registry

# register BOX Matcher
MATCHER_REGISTRY = Registry(
    "matcher",
    base_class=BaseMatcher,
    lazy_load_dirs=["corenet/modeling/matcher_det"],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def arguments_box_matcher(parser: argparse.ArgumentParser):
    group = parser.add_argument_group("Matcher", "Matcher")
    group.add_argument(
        "--matcher.name",
        type=str,
        help="Name of the matcher. Matcher matches anchors with GT box coordinates",
    )

    # add segmentation specific arguments
    parser = MATCHER_REGISTRY.all_arguments(parser)
    return parser


def build_matcher(opts, *args, **kwargs):
    matcher_name = getattr(opts, "matcher.name", None)
    # We registered the base class using a special `name` (i.e., `__base__`)
    # in order to access the arguments defined inside those classes. However, these classes are not supposed to
    # be used. Therefore, we raise an error for such cases
    if matcher_name == "__base__":
        logger.error("__base__ can't be used as a projection name. Please check.")

    matcher = MATCHER_REGISTRY[matcher_name](opts, *args, **kwargs)
    return matcher
