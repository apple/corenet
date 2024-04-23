#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import pytest

from corenet.data.data_loaders import CoreNetDataLoader
from corenet.data.sampler import build_sampler
from tests.configs import get_config
from tests.data.datasets.segmentation.mock_coco_stuff import MockCOCOStuffDataset


@pytest.mark.parametrize("mode", ["train", "val", "eval"])
def test_cocostuff_dataset(mode: str) -> None:
    """Test for COCOStuff

    This test mocks `__init__`, `read_image_pil`, and `read_mask_pil` functions to test the
    `__getitem__` function (that includes data transforms). Training and validation datasets are wrapped
    inside the respective data loaders to test the collate function used by dataset.
    """
    config_file_path = "tests/data/datasets/segmentation/dummy_cocostuff_config.yaml"
    opts = get_config(config_file=config_file_path)

    # TEST TRAINING DATASET, SAMPLER, COLLATE_FN
    expected_input_shape = [4, 3, 64, 64]

    is_training = mode == "train"
    is_evaluation = mode == "eval"
    if is_evaluation:
        expected_input_shape = [2, 3, 128, 64]
        # enable resizing based on aspect ratio for evaluation
        setattr(opts, "evaluation.segmentation.resize_input_images", True)
        # overwrite the validation argument
        setattr(
            opts, "dataset.val_batch_size0", getattr(opts, "dataset.eval_batch_size0")
        )

    dataset = MockCOCOStuffDataset(
        opts, is_training=is_training, is_evaluation=is_evaluation
    )
    sampler = build_sampler(opts, n_data_samples=len(dataset), is_training=is_training)
    data_loader = CoreNetDataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        # our samplers take care of batch size, so we set to 1 here
        batch_size=1,
        num_workers=0,
    )
    for batch in data_loader:
        assert batch.keys() == {"samples", "targets"}
        assert [*batch["samples"].shape] == expected_input_shape
        assert (
            batch["samples"].dim() == 4
        ), "Expecting input images in [batch, image_channels, height, width] format"
        if mode in ["train", "val"]:
            assert (
                batch["targets"].dim() == 3
            ), "Expecting labels in [batch, height, width ] format"
        else:
            assert batch["targets"].keys() == {
                "mask",
                "file_name",
                "im_width",
                "im_height",
            }
            assert (
                batch["targets"]["mask"].dim() == 3
            ), "Expecting labels in [batch, height, width] format"

            # file names are expected as [file_1, file_2, ...]
            assert (
                len(batch["targets"]["file_name"]) == 2
            ), "Expecting file names as a list."

            # check the widths of input images. 20 is specified as dummy dimension for labels in read_mask_pil function.
            assert [*batch["targets"]["im_width"]] == [20, 20]
            # check the widths of input images. 40 is specified as dummy dimension for labels in read_mask_pil function.
            assert [*batch["targets"]["im_height"]] == [40, 40]
