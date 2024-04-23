#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import json
import tempfile

import yaml

from corenet.data.datasets.language_modeling import commonsense_170k
from corenet.options.utils import flatten_yaml_as_dict
from tests.configs import get_config
from tests.data.datasets.language_modeling import test_general_lm


def write_data(filename: str) -> None:
    data = [
        {
            "instruction": "Please answer the following question with true or false. Question: is the sky blue?",
            "input": "",
            "output": "the correct answer is true",
            "answer": "true",
        }
    ] * 5
    # Make input non-empty for one data point.
    data[0]["input"] = "This is an example input."
    with open(filename, "w+") as f:
        json.dump(data, f)


def test_general_lm_dataset() -> None:
    """Test for GeneralLMDataset dataset."""
    sequence_length = 5
    with tempfile.NamedTemporaryFile() as tmp:
        write_data(tmp.name)
        config_file = (
            "tests/data/datasets/language_modeling/dummy_commonsense_170k.yaml"
        )
        opts = get_config(config_file=config_file)
        setattr(opts, "dataset.language_modeling.sequence_length", 5)
        setattr(opts, "dataset.language_modeling.commonsense_170k.path", tmp.name)

        dataset = commonsense_170k.CommonSense170k(opts)
        max_iterations = 12

        test_general_lm._iterate_and_test_dataset(
            dataset,
            max_iterations=max_iterations,
            expected_sequence_length=sequence_length,
        )
