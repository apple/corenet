#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.data.datasets.multi_modal_img_text.zero_shot_image_classification.base_zero_shot_image_classification import (
    BaseZeroShotImageClassificationDataset,
)
from corenet.utils.registry import Registry

ZERO_SHOT_IMAGE_CLASSIFICATION_DATASET_REGISTRY = Registry(
    registry_name="zero_shot_datasets",
    base_class=BaseZeroShotImageClassificationDataset,
    lazy_load_dirs=[
        "corenet/data/datasets/multi_modal_img_text/zero_shot_image_classification"
    ],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def arguments_zero_shot_image_classification_dataset(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Helper function to get zero-shot dataset arguments"""
    parser = BaseZeroShotImageClassificationDataset.add_arguments(parser=parser)
    parser = ZERO_SHOT_IMAGE_CLASSIFICATION_DATASET_REGISTRY.all_arguments(parser)
    return parser


def build_zero_shot_image_classification_dataset(
    opts: argparse.Namespace, *args, **kwargs
) -> BaseZeroShotImageClassificationDataset:
    """Helper function to build the zero shot image classification dataset."""
    zero_shot_dataset_name = getattr(
        opts, "dataset.multi_modal_img_text.zero_shot_img_cls_dataset_name"
    )
    return ZERO_SHOT_IMAGE_CLASSIFICATION_DATASET_REGISTRY[zero_shot_dataset_name](
        opts, *args, **kwargs
    )
