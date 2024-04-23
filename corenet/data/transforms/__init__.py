#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.data.transforms.base_transforms import BaseTransformation
from corenet.utils.registry import Registry

TRANSFORMATIONS_REGISTRY = Registry(
    "transformation",
    base_class=BaseTransformation,
    lazy_load_dirs=["corenet/data/transforms"],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def arguments_augmentation(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # add arguments for base image projection layer
    parser = BaseTransformation.add_arguments(parser)

    # add augmentation specific arguments
    parser = TRANSFORMATIONS_REGISTRY.all_arguments(parser)
    return parser
