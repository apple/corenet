#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import functools

import pytest

from corenet.data.collate_fns import build_collate_fn
from corenet.data.data_loaders import CoreNetDataLoader
from corenet.data.datasets.classification.base_image_classification_dataset import (
    BaseImageClassificationDataset,
)
from corenet.data.sampler import build_sampler
from tests.configs import get_config


@pytest.mark.parametrize(
    "num_samples_per_category, percentage_of_samples, is_training",
    [
        (-1, None, True),
        (0, None, True),
        (1, None, True),
        (1, None, False),
        (2, None, True),
        (2, None, False),
        (None, 50, True),
        (None, 50, False),
        (None, -10, True),
        (None, 0, True),
        (None, 120, True),
    ],
)
def test_base_image_classification_dataset(
    num_samples_per_category: int, percentage_of_samples: float, is_training: bool
) -> None:
    """
    Test for corenet.data.datasets.classification.base_image_classification_dataset.BaseImageClassificationDataset
    """

    config_file_path = "tests/data/datasets/classification/dummy_configs/image_classification_dataset.yaml"
    opts = get_config(config_file=config_file_path)
    num_classes = 2
    num_samples = 4
    # we can't control this parameter from config as it is not defined in parser.
    setattr(opts, "ddp.use_distributed", False)
    if num_samples_per_category is not None:
        setattr(opts, "dataset.num_samples_per_category", num_samples_per_category)
    else:
        setattr(opts, "dataset.percentage_of_samples", percentage_of_samples)
    dataset = BaseImageClassificationDataset(
        opts, is_training=is_training, is_evaluation=False
    )
    sampler = build_sampler(opts, n_data_samples=len(dataset), is_training=is_training)
    collate_fn_train, collate_fn_val = build_collate_fn(opts=opts)

    if is_training:
        crop_size_width = getattr(opts, "sampler.bs.crop_size_width")
        crop_size_height = getattr(opts, "sampler.bs.crop_size_height")
        batch_size = getattr(opts, "dataset.train_batch_size0")
        collate_fn = collate_fn_train
        if num_samples_per_category and num_samples_per_category > 0:
            expected_num_samples = num_classes * num_samples_per_category
        elif percentage_of_samples and 0 < percentage_of_samples < 100:
            expected_num_samples = num_samples * percentage_of_samples / 100
        else:
            expected_num_samples = num_samples
    else:
        crop_size_width = getattr(opts, "image_augmentation.center_crop.size")
        crop_size_height = getattr(opts, "image_augmentation.center_crop.size")
        batch_size = getattr(opts, "dataset.val_batch_size0")
        collate_fn = collate_fn_val
        expected_num_samples = num_samples

    loader = CoreNetDataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        batch_size=1,
        num_workers=0,
        collate_fn=functools.partial(collate_fn, opts=opts),
    )

    assert len(dataset) == expected_num_samples

    for batch in loader:
        assert batch.keys() == {"samples", "sample_id", "targets"}
        assert [*batch["samples"].shape] == [
            batch_size,
            3,
            crop_size_height,
            crop_size_width,
        ]
        assert [*batch["sample_id"].shape] == [batch_size]
        assert [*batch["targets"].shape] == [batch_size]
