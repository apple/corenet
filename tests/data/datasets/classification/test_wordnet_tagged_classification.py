#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import sys

import pytest
import torch

from tests.configs import get_config
from tests.data.datasets.classification.mock_wordnet_tagged_classification import (
    MockWordnetTaggedClassificationDataset,
)


@pytest.mark.parametrize("image_size", [16, 32])
def test_wordnet_tagged_classification_dataset(image_size: int) -> None:
    """Test for WordnetTaggedClassificationDataset dataset."""

    if "nltk" in sys.modules:
        config_file = "tests/data/datasets/classification/dummy_configs/wordnet_tagged_classification.yaml"
        opts = get_config(config_file=config_file)

        dataset = MockWordnetTaggedClassificationDataset(
            opts, is_training=True, is_evaluation=False
        )

        sample_index = 0
        data_item = dataset.__getitem__((image_size, image_size, sample_index))
        assert "samples" in data_item
        assert "targets" in data_item
        assert list(data_item["samples"].shape) == [3, image_size, image_size]
        assert list(data_item["targets"].shape) == [10]

        exptected_target_label = torch.tensor([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        assert torch.all(
            data_item["targets"]
            == exptected_target_label.to(
                dtype=data_item["targets"].dtype, device=data_item["targets"].device
            )
        )
