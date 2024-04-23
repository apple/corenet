#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from corenet.data.loader.dataloader import CoreNetDataLoader
from corenet.data.sampler import build_sampler
from tests.configs import get_config
from tests.data.datasets.classification.mock_coco import MockCOCOClassification


def test_coco_dataset() -> None:
    """Test for COCO classification dataset."""
    config_file_path = "tests/data/datasets/classification/dummy_configs/coco.yaml"
    opts = get_config(config_file=config_file_path)

    dataset = MockCOCOClassification(opts)

    train_sampler = build_sampler(opts, n_data_samples=len(dataset), is_training=True)

    train_loader = CoreNetDataLoader(
        dataset=dataset,
        batch_sampler=train_sampler,
        batch_size=1,
        num_workers=0,
    )

    for batch in train_loader:
        assert batch.keys() == {"samples", "targets", "sample_id"}
        # bounds from the config file
        assert list(batch["samples"].shape) == [
            2,
            3,
            64,
            64,
        ], "The output shape should be [2, 3, 64, 64]."
        assert list(batch["targets"].shape) == [
            2,
            80,
        ], "Batch size should be 2 and number of classes should be 80."
        assert (
            batch["sample_id"].dim() == 1
        ), "Expecting sample_id's in [batch, ] format"
