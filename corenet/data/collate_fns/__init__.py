#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.utils import logger
from corenet.utils.registry import Registry

COLLATE_FN_REGISTRY = Registry(
    "collate_fn",
    lazy_load_dirs=["corenet/data/collate_fns"],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def arguments_collate_fn(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add arguments related to collate function"""
    group = parser.add_argument_group("Collate function arguments")
    group.add_argument(
        "--dataset.collate-fn-name-train",
        type=str,
        default="pytorch_default_collate_fn",
        help="Name of collate function for training. Defaults to pytorch_default_collate_fn.",
    )
    group.add_argument(
        "--dataset.collate-fn-name-val",
        type=str,
        default="pytorch_default_collate_fn",
        help="Name of collate function for validation. Defaults to pytorch_default_collate_fn.",
    )
    group.add_argument(
        "--dataset.collate-fn-name-test",
        type=str,
        default="pytorch_default_collate_fn",
        help="Name of collate function used for evaluation. "
        "Default is pytorch_default_collate_fn.",
    )
    return parser


def build_collate_fn(opts, *args, **kwargs):
    collate_fn_name_train = getattr(opts, "dataset.collate_fn_name_train")

    if collate_fn_name_train is None:
        logger.error(
            "Please specify collate function for training dataset using "
            "--dataset.collate-fn-name-train"
        )

    collate_fn_name_val = getattr(opts, "dataset.collate_fn_name_val")
    if collate_fn_name_val is None:
        logger.error(
            "Please specify collate function for training dataset using "
            "--dataset.collate-fn-name-val"
        )

    collate_fn_train = COLLATE_FN_REGISTRY[collate_fn_name_train]
    collate_fn_val = COLLATE_FN_REGISTRY[collate_fn_name_val]
    return collate_fn_train, collate_fn_val


def build_test_collate_fn(opts, *args, **kwargs):
    collate_fn_name_test = getattr(opts, "dataset.collate_fn_name_test")
    # for test time
    if collate_fn_name_test is not None:
        return COLLATE_FN_REGISTRY[collate_fn_name_test]
    return None
