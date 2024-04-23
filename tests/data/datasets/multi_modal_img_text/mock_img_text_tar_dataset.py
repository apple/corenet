#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import base64
import io
import os
import pickle

from PIL import Image

from corenet.constants import DATA_CACHE_DIR
from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.multi_modal_img_text.img_text_tar_dataset import (
    ImgTextTarDataset,
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

            data = {"image": encoded_image, "text": "Testing img-text dataset."}
            pickle.dump(data, f)

    metadata = {
        "total_tar_files": 2,
        "max_files_per_tar": num_samples_per_tar,
        "tar_file_names": ["0.tar.gz", "1.tar.gz"],
    }

    with open(f"{DATA_CACHE_DIR}/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)


@DATASET_REGISTRY.register(name="mock_img_text_tar", type="multi_modal_image_text")
class MockImgTextTarDataset(ImgTextTarDataset):
    """A wrapper around ImgTextTarDataset that generates dummy data for CI/CD.

    Args:
        opts: Command-line arguments.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        _generate_dummy_data()
        super().__init__(opts, *args, **kwargs)
