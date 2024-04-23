#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Optional, Tuple

from corenet.constants import if_test_env
from corenet.data.datasets.classification.base_image_classification_dataset import (
    BaseImageClassificationDataset,
)
from corenet.data.datasets.classification.base_imagenet_shift_dataset import (
    BaseImageNetShiftDataset,
)
from corenet.data.datasets.dataset_base import (
    BaseDataset,
    BaseImageDataset,
    BaseIterableDataset,
    BaseVideoDataset,
)
from corenet.data.datasets.detection.base_detection import BaseDetectionDataset
from corenet.data.datasets.language_modeling.base_lm import BaseLMIterableDataset
from corenet.data.datasets.multi_modal_img_text import arguments_multi_modal_img_text
from corenet.data.datasets.segmentation.base_segmentation import (
    BaseImageSegmentationDataset,
)
from corenet.utils import logger
from corenet.utils.ddp_utils import is_master
from corenet.utils.registry import Registry

DATASET_REGISTRY = Registry(
    registry_name="dataset_registry",
    base_class=BaseDataset,
    lazy_load_dirs=["corenet/data/datasets"]
    + if_test_env(
        then=["tests/data/datasets"],
        otherwise=[],
    ),
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def build_dataset_from_registry(
    opts: argparse.Namespace,
    is_training: bool = True,
    is_evaluation: bool = False,
    *args,
    **kwargs,
) -> BaseDataset:
    """Helper function to build a dataset from dataset registry

    Args:
        opts: Command-line arguments
        is_training: Training mode or not. Defaults to True.
        is_evaluation: Evaluation mode or not. Defaults to False.

    Returns:
        An instance of BaseDataset


    ...note:
        `is_training` is used to indicate whether the dataset is used for training or validation
        On the other hand, `is_evaluation` mode is used to indicate the dataset is used for testing.

        Theoretically, `is_training=False` and `is_evaluation=True` should be the same. However, for some datasets
        (especially segmentation), validation dataset transforms are different from
        test transforms because each image has different resolution, making it difficult to construct
        batches. Therefore, we treat these two modes different. For datasets, where validation and testing
        transforms are the same, we set evaluation transforms the same as the validation transforms (e.g., in ImageNet
        object classification).
    """

    dataset_category = getattr(opts, "dataset.category")

    if dataset_category is None:
        logger.error("Please specify dataset category using --dataset.category")

    dataset_name = getattr(opts, f"dataset.name")
    if dataset_name is None:
        logger.error("Please specify dataset name using --dataset.name")

    dataset = DATASET_REGISTRY[dataset_name, dataset_category](
        opts=opts, is_training=is_training, is_evaluation=is_evaluation, *args, **kwargs
    )
    return dataset


def get_test_dataset(opts: argparse.Namespace, *args, **kwargs) -> BaseDataset:
    """Helper function to build a dataset for testing.

    Args:
        opts: Command-line arguments

    Returns:
        An instance of BaseDataset
    """

    test_dataset = build_dataset_from_registry(
        opts, is_training=False, is_evaluation=True, *args, **kwargs
    )

    if is_master(opts):
        logger.log("Evaluation dataset details: ")
        print("{}".format(test_dataset))

    return test_dataset


def get_train_val_datasets(
    opts: argparse.Namespace, *args, **kwargs
) -> Tuple[BaseDataset, Optional[BaseDataset]]:
    """Helper function to build a dataset for training and validation.

    Args:
        opts: Command-line arguments

    Returns:
        Training and (optionally) validation datasets.
    """

    disable_val = getattr(opts, "dataset.disable_val")
    is_master_node = is_master(opts)
    train_dataset = build_dataset_from_registry(
        opts, is_training=True, is_evaluation=False, *args, **kwargs
    )
    if is_master_node:
        logger.log("Training dataset details are given below")
        print(train_dataset)

    valid_dataset = None
    if not disable_val:
        valid_dataset = build_dataset_from_registry(
            opts, is_training=False, is_evaluation=False, *args, **kwargs
        )
        if is_master_node:
            logger.log("Validation dataset details are given below")
            print(valid_dataset)

    return train_dataset, valid_dataset


def arguments_dataset(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add dataset-specific arguments from BaseDataset, BaseImageDataset,
    BaseImageClassificationDataset, BaseImageNetShiftDataset,
    BaseVideoDataset, zero-shot datasets, and DATASET_REGISTRY.
    """
    parser = BaseDataset.add_arguments(parser)
    parser = BaseImageDataset.add_arguments(parser)
    parser = BaseImageSegmentationDataset.add_arguments(parser)
    parser = BaseVideoDataset.add_arguments(parser)
    parser = BaseImageClassificationDataset.add_arguments(parser)
    parser = BaseImageNetShiftDataset.add_arguments(parser)
    parser = BaseDetectionDataset.add_arguments(parser)
    parser = BaseLMIterableDataset.add_arguments(parser)

    try:
        from corenet.internal.utils.server_utils import dataset_server_args

        parser = dataset_server_args(parser)
    except ImportError:
        pass

    # add multi-modal and zero-shot arguments
    parser = arguments_multi_modal_img_text(parser=parser)

    # add dataset specific arguments
    parser = DATASET_REGISTRY.all_arguments(parser)
    return parser
