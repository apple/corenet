#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Type

import pytest

from corenet.data.loader.dataloader import CoreNetDataLoader
from corenet.data.sampler import build_sampler
from tests.configs import get_config
from tests.data.datasets.classification.mock_imagenet import (
    MockImageNetADataset,
    MockImageNetDataset,
    MockImageNetRDataset,
    MockImageNetSketchDataset,
)


@pytest.mark.parametrize(
    "config_file_path,mock_dataset_class,eval_only",
    [
        (
            "tests/data/datasets/classification/dummy_configs/imagenet.yaml",
            MockImageNetDataset,
            False,
        ),
        (
            "tests/data/datasets/classification/dummy_configs/imagenet_a.yaml",
            MockImageNetADataset,
            True,
        ),
        (
            "tests/data/datasets/classification/dummy_configs/imagenet_r.yaml",
            MockImageNetRDataset,
            True,
        ),
        (
            "tests/data/datasets/classification/dummy_configs/imagenet_sketch.yaml",
            MockImageNetSketchDataset,
            True,
        ),
    ],
)
def test_imagenet_dataset_train(
    config_file_path: str,
    mock_dataset_class: Type[MockImageNetDataset],
    eval_only: bool,
) -> None:
    """Test for ImageNet dataset and its variants.

    This test mocks `__init__` and `read_image_pil` functions to test the `__getitem__`
    function (that includes data transforms). Training and validation datasets are
    wrapped inside the respective data loaders to test the collate function used by
    dataset.

    Args:
      config_file_path: Path to a dummy config for initializing the dataset.
      mock_dataset_class: The class type for ImageNet or one of its shifts.
      eval_only: If set, the dataset is tested to be for evaluation-only.
    """
    opts = get_config(config_file=config_file_path)

    if eval_only:
        with pytest.raises(Exception):
            training_imagenet_dataset = mock_dataset_class(
                opts, is_training=True, is_evaluation=False
            )
        return
    else:
        training_imagenet_dataset = mock_dataset_class(
            opts, is_training=True, is_evaluation=False
        )

    train_sampler = build_sampler(
        opts, n_data_samples=len(training_imagenet_dataset), is_training=True
    )

    train_loader = CoreNetDataLoader(
        dataset=training_imagenet_dataset,
        batch_sampler=train_sampler,
        batch_size=1,
        num_workers=0,
    )

    for batch in train_loader:
        assert batch.keys() == {"samples", "targets", "sample_id"}
        # bounds from the config file
        assert 128 <= batch["samples"].shape[-2] <= 320
        assert 128 <= batch["samples"].shape[-1] <= 320
        assert batch["samples"].dim() == 4, (
            "Expecting input images in " "[batch, image_channels, height, width] format"
        )
        assert batch["targets"].dim() == 1, "Expecting labels in [batch, ] format"
        assert (
            batch["sample_id"].dim() == 1
        ), "Expecting sample_id's in [batch, ] format"


@pytest.mark.parametrize(
    "config_file_path,mock_dataset_class",
    [
        (
            "tests/data/datasets/classification/dummy_configs/imagenet.yaml",
            MockImageNetDataset,
        ),
        (
            "tests/data/datasets/classification/dummy_configs/imagenet_a.yaml",
            MockImageNetADataset,
        ),
        (
            "tests/data/datasets/classification/dummy_configs/imagenet_r.yaml",
            MockImageNetRDataset,
        ),
        (
            "tests/data/datasets/classification/dummy_configs/imagenet_sketch.yaml",
            MockImageNetSketchDataset,
        ),
    ],
)
def test_imagenet_dataset_val(
    config_file_path: str,
    mock_dataset_class: Type[MockImageNetDataset],
) -> None:
    """Test for ImageNet dataset and its variants.

    This test mocks `__init__` and `read_image_pil` functions to test the `__getitem__`
    function (that includes data transforms). Training and validation datasets are
    wrapped inside the respective data loaders to test the collate function used by
    dataset.

    Args:
      config_file_path: Path to a dummy config for initializing the dataset.
      mock_dataset_class: The class type for ImageNet or one of its shifts.
    """
    opts = get_config(config_file=config_file_path)

    valid_imagenet_dataset = mock_dataset_class(
        opts, is_training=False, is_evaluation=False
    )

    val_sampler = build_sampler(
        opts, n_data_samples=len(valid_imagenet_dataset), is_training=False
    )

    val_loader = CoreNetDataLoader(
        dataset=valid_imagenet_dataset,
        batch_sampler=val_sampler,
        batch_size=1,
        num_workers=0,
    )

    for batch in val_loader:
        assert batch.keys() == {"samples", "targets", "sample_id"}
        # values from config file
        assert [*batch["samples"].shape[2:]] == [224, 224]
        assert batch["samples"].dim() == 4, (
            "Expecting input images in " "[batch, image_channels, height, width] format"
        )
        assert batch["targets"].dim() == 1, "Expecting labels in [batch, ] format"
        assert (
            batch["sample_id"].dim() == 1
        ), "Expecting sample_id's in [batch, ] format"
