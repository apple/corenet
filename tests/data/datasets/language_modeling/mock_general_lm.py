#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import gzip
import json

from corenet.constants import DATA_CACHE_DIR
from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.language_modeling.general_lm import GeneralLMDataset


def _generate_dummy_json_data() -> None:
    data = [
        {
            "text": "Hello world, CoreNet serves as a versatile research library catering to a wide array of purposes. It has been used for small- and large-scale training, with numerous research papers leveraging its functionalities and contributing to various domains of study.",
        }
    ] * 12

    with open(f"{DATA_CACHE_DIR}/sample.jsonl", "w") as outfile:
        for entry in data:
            print(json.dumps(entry), file=outfile)


def _generate_dummy_json_gz_data() -> None:
    data = [{"text": "   !"}] * 2

    with gzip.open(f"{DATA_CACHE_DIR}/sample.json.gz", "w") as outfile:
        for text in data:
            json_str = json.dumps(text) + "\n"
            json_bytes = json_str.encode("utf-8")
            outfile.write(json_bytes)


@DATASET_REGISTRY.register(name="mock_general_lm", type="language_modeling")
class MockImgGeneralLMDataset(GeneralLMDataset):
    """A wrapper around GeneralLMDataset that generates dummy data for CI/CD.

    Args:
        opts: Command-line arguments.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        _generate_dummy_json_data()
        _generate_dummy_json_gz_data()
        super().__init__(opts, *args, **kwargs)
