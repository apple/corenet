#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import base64
import io
import os
import pickle
import subprocess

from PIL import Image

from corenet.constants import DATA_CACHE_DIR
from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.classification.wordnet_tagged_classification import (
    WordnetTaggedClassificationDataset,
)


def _generate_dummy_data() -> None:
    """Generate dummy data for CI/CD."""

    total_samples = 4
    num_samples_per_tar = 2

    for sample_idx in range(total_samples):
        folder_idx = sample_idx // num_samples_per_tar
        folder_path = f"{DATA_CACHE_DIR}/{folder_idx}"
        os.makedirs(folder_path, exist_ok=True)
        with open(f"{folder_path}/{sample_idx}.pkl", "wb") as f:
            image = Image.new("RGB", (32, 32), color="black")
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="JPEG")
            image_bytes.seek(0)
            encoded_image = base64.b64encode(image_bytes.read())

            data = {"image": encoded_image, "text": "An image of black dog."}
            pickle.dump(data, f)

    metadata = {
        "total_tar_files": 2,
        "max_files_per_tar": num_samples_per_tar,
        "tar_file_names": ["0.tar.gz", "1.tar.gz"],
    }

    with open(f"{DATA_CACHE_DIR}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    vocab = {
        "n5928118": 52418327,  # image
        "n13333833": 46393897,  # stock
        "n4960277": 38781582,  # black
        "n7947958": 36532096,
        "n9638875": 34564013,
        "n928077": 30290822,
        "n7996689": 28076676,
        "n10787470": 24182531,
        "n5938976": 23817664,
        "n8559508": 23476398,
    }
    with open(f"{DATA_CACHE_DIR}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)


@DATASET_REGISTRY.register(
    name="mock_wordnet_tagged_classification", type="classification"
)
class MockWordnetTaggedClassificationDataset(WordnetTaggedClassificationDataset):
    """A wrapper around WordnetTaggedClassificationDataset that generates dummy data for CI/CD.

    Args:
        opts: Command-line arguments.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        import nltk

        # download nltk datasets
        nltk.download("all")
        _generate_dummy_data()
        super().__init__(opts, *args, **kwargs)
