#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import pytest
import torch

from tests.configs import get_config
from tests.data.datasets.language_modeling.mock_general_lm import (
    MockImgGeneralLMDataset,
)


def _iterate_and_test_dataset(
    dataset: MockImgGeneralLMDataset, max_iterations: int, expected_sequence_length: int
):
    for sample_id, sample in enumerate(dataset):
        assert any(set(sample.keys()) & {"samples", "targets"})
        assert sample["samples"].numel() == expected_sequence_length
        assert sample["targets"].numel() == expected_sequence_length
        # inputs are shifted by 1 to obtain targets
        assert torch.all(sample["samples"][1:] == sample["targets"][:-1])
        if sample_id == max_iterations:
            break


@pytest.mark.parametrize("max_iterations", [10, 13, 16])
def test_general_lm_dataset(max_iterations: int) -> None:
    """Test for GeneralLMDataset dataset."""

    config_file = "tests/data/datasets/language_modeling/dummy_lm_dataset.yaml"
    opts = get_config(config_file=config_file)
    sequence_length = 12
    setattr(opts, "dataset.language_modeling.sequence_length", sequence_length)

    dataset = MockImgGeneralLMDataset(opts)
    _iterate_and_test_dataset(
        dataset, max_iterations=max_iterations, expected_sequence_length=sequence_length
    )

    # resume the state and iterate for 5 more iterations.
    setattr(
        opts,
        "dataset.language_modeling.general_lm.data_state",
        ["results/run_1/data_states/data_state_0_0.pkl"],
    )
    dataset = MockImgGeneralLMDataset(opts)
    dataset._load_data_state()

    # Our first file has 12 samples and second has 2 files, so total of 14 files (i.e., 1 epoch is 14 files)
    if max_iterations == 10:
        # In this case, only 10 samples from first file are finished, so we expect chunk state to resume from 10
        assert dataset._target_state["chunk"] == 10
        assert dataset._target_state["file"] == None
        assert dataset._target_state["epoch"] == 0
    elif max_iterations == 13:
        # in this case, first file has finished and we are iterating second file, so we expect the file has the name
        # of first file and chunks state to be 1 because one sample of the second file was processed
        print("Resuming from second file")
        assert dataset._target_state["chunk"] == 1
        assert dataset._target_state["file"].endswith("sample.jsonl")
        assert dataset._target_state["epoch"] == 0
    elif max_iterations == 16:
        # in this case, we have iterated over the data once and processed 2 samples from first file.
        # So, we expect epoch to be 1, chunk to be 2, and file to be None
        assert dataset._target_state["chunk"] == 2
        assert dataset._target_state["file"] == None
        assert dataset._target_state["epoch"] == 1
    else:
        raise NotImplementedError("Max iteration is not supported")

    max_iterations = 5
    _iterate_and_test_dataset(
        dataset, max_iterations=max_iterations, expected_sequence_length=sequence_length
    )
