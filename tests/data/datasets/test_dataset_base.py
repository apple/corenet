#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Any, Union

import pytest

from corenet.data.datasets.dataset_base import BaseDataset
from tests.configs import get_config
from tests.data.datasets.classification.mock_imagenet import MockImageNetDataset


def test_repr_with_imagenet():
    opts = get_config()
    setattr(opts, "image_augmentation.random_horizontal_flip.enable", True)
    setattr(opts, "image_augmentation.random_resized_crop.enable", True)

    mock_imagenet_dataset = MockImageNetDataset(
        opts, is_training=True, is_evaluation=False
    )
    actual_repr = repr(mock_imagenet_dataset)
    # ignore some extra spaces
    actual_repr = actual_repr.replace("\t ", "\t").replace(" \n", "\n")

    expected_repr = """\
MockImageNetDataset(
    root=None
    is_training=True
    num_samples=10
    transforms=Compose(
            RandomResizedCrop(scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), size=(256, 256), interpolation=bilinear),
            RandomHorizontalFlip(p=0.5),
            ToTensor(dtype=torch.float32, norm_factor=255)
        )
    num_classes=1000
)""".replace(
        " " * 4, "\t"
    )

    assert actual_repr == expected_repr


@pytest.mark.parametrize("mean", [0.0, [0.0, 0.0, 0.0]])
@pytest.mark.parametrize("std", [1.0, [1.0, 1.0, 1.0]])
def test_repr_with_imagenet_with_mean_std_norm(
    mean: Union[float, list], std: Union[float, list]
) -> None:
    opts = get_config()
    setattr(opts, "image_augmentation.random_horizontal_flip.enable", True)
    setattr(opts, "image_augmentation.random_resized_crop.enable", True)
    setattr(opts, "image_augmentation.to_tensor.mean_std_normalization.enable", True)
    setattr(opts, "image_augmentation.to_tensor.mean_std_normalization.mean", mean)
    setattr(opts, "image_augmentation.to_tensor.mean_std_normalization.std", std)

    mock_imagenet_dataset = MockImageNetDataset(
        opts, is_training=True, is_evaluation=False
    )
    actual_repr = repr(mock_imagenet_dataset)
    # ignore some extra spaces
    actual_repr = actual_repr.replace("\t ", "\t").replace(" \n", "\n")

    expected_repr = f"""\
MockImageNetDataset(
    root=None
    is_training=True
    num_samples=10
    transforms=Compose(
            RandomResizedCrop(scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), size=(256, 256), interpolation=bilinear),
            RandomHorizontalFlip(p=0.5),
            ToTensor(dtype=torch.float32, norm_factor=255, mean_std_normalization_enable=True, normalization_mean={mean}, normalization_std={std})
        )
    num_classes=1000
)""".replace(
        " " * 4, "\t"
    )

    assert actual_repr == expected_repr


def test_item_metadata_exceptions():
    class MyDataset(BaseDataset):
        def __getitem__(self, sample_size_and_index: Any) -> Any:
            return 2

    opts = get_config()
    dataset = MyDataset(opts)
    # Make sure datasets that don't need item_metadata work without exception:
    assert dataset[0] == 2

    # The proper error to be raised is NotImplementedError, so that the users know they
    # should implement which method
    with pytest.raises(NotImplementedError):
        dataset.get_item_metadata(0)
