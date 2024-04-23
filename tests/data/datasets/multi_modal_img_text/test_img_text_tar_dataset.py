#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#


import pytest

from tests.configs import get_config
from tests.data.datasets.multi_modal_img_text.mock_img_text_tar_dataset import (
    MockImgTextTarDataset,
)


@pytest.mark.parametrize("image_size", [16, 32])
@pytest.mark.parametrize("context_length", [12, 77])
def test_img_text_dataset(image_size: int, context_length: int) -> None:
    """Test for ImgTextTarDataset dataset."""

    config_file = (
        "tests/data/datasets/multi_modal_img_text/dummy_img_text_tar_dataset.yaml"
    )
    opts = get_config(config_file=config_file)
    setattr(opts, "dataset.multi_modal_img_text.context_length", context_length)

    dataset = MockImgTextTarDataset(opts, is_training=True, is_evaluation=False)

    sample_index = 0
    data_item = dataset.__getitem__((image_size, image_size, sample_index))
    assert "samples" in data_item
    assert "targets" in data_item
    assert data_item["targets"] == -1
    assert "image" in data_item["samples"]
    assert list(data_item["samples"]["image"].shape) == [3, image_size, image_size]
    assert list(data_item["samples"]["text"].shape) == [context_length]
