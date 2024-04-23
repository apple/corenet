#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import fcntl
import glob
import io
import os
import pickle
import random
import shutil
import tarfile
import time
from pathlib import Path
from typing import Tuple

import pybase64
from PIL import Image, ImageFile

from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.multi_modal_img_text.base_multi_modal_img_text import (
    BaseMultiModalImgText,
)
from corenet.utils import logger
from corenet.utils.download_utils import get_local_path

# To enable reading truncated images, we update the default values of following variables in PIL
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

TAR_FILE_EXTN = "tar.gz"
TAR_FILE_EXTRACTION_CODE = "r:gz"
SAMPLE_FILE_EXTN = "pkl"


@DATASET_REGISTRY.register(name="img_text_tar", type="multi_modal_image_text")
class ImgTextTarDataset(BaseMultiModalImgText):
    """ImgTextTarDataset class for datasets that store Image-Text pairs as tar files, each tar file with multiple image-text pairs.

    Assuming that dataset has 10M image-text pairs and each tar file can contains 10000 files, then we can create
    1000 tar files. The first tar file, 0.tar.gz, would contain files from 0 to 9999, the second tar file, 1.tar.gz,
    would contain files from 10000 to 19999, and so on.

    With 'image_text_data' as the root directory, the expected structure for the training data would be:

        image_text_data/
        ├── metadata.pkl
        ├── 0.tar.gz
        ├── 1.tar.gz
        ├── 2.tar.gz
        ├── 3.tar.gz
        ....
        ├── 1000.tar.gz


    The metadata is a dictionary containing information about the dataset, and is stored as a pickle file.
    It should have following keys:
        1. total_tar_files: Total number of tar files in the dataset
        2. max_files_per_tar: Maximum number of files inside each tar file.
        3. tar_file_names: List containing names of the tar files.

    Each tar file contains the information about image-text files in the pickle format. For example, the content
    after extracting '0.tar.gz' (assuming it contains 1000 pickle files) would look like this.

        0/
        ├── 0.pkl
        │   ├── image
        │   ├── text
        ├── 1.pkl
        ├── ...
        ├── 9999.pkl

    Each image-text pair is stored as a dictionary with 'image' and 'text' as keys in a pickle file. The value corresponding to key 'image' corresponds
    to an image in binary format and the value corresponding to key 'text' corresponds to text caption.

    ...note:
        1. Recommended size of each tar file is about 1GB.
        2. Metadata should be stored in the same folder as the tar files.
        3. Metadata should be stored as a pickle file.
        4. We assume that data is stored in S3. Please make appropriate changes for other storage clients.
        5. We generally expect the same number of files per tar. However, due to filtering, the number of files
           in a given tar could be below the expected number of files. In such cases, we over-sample from that tar file.

    ...note:
        For evaluation, we use standard image classification dataset. Please see 'BaseImageClassificationDataset' for expected dataset structure for classification datasets.
    """

    def _metadata_file_path(self) -> str:
        """Read metadata file path from command-line arguments."""
        opts = self.opts

        metadata_file_path = getattr(
            opts, f"dataset.multi_modal_img_text.img_text_tar.metadata_file"
        )

        if not metadata_file_path:
            logger.error(
                f"Please specify metadata file path using 'dataset.multi_modal_img_text.img_text_tar.metadata_file'."
            )
        return metadata_file_path

    def _metadata(self):
        """Read the metadata content.

        ...note:
            The metadata file is expected to have following keys:
            1. total_tar_files: Total number of tar files in the dataset
            2. max_files_per_tar: Maximum number of files inside each tar file.
            3. tar_file_names: List containing names of the tar files.
        """
        opts = self.opts
        metadata_file_path = self._metadata_file_path()

        # download the metadata file
        metadata_file_local_path = get_local_path(
            opts,
            path=metadata_file_path,
            force_delete=False,
            use_start_rank=True,
            sync_ranks=False,
        )

        with open(metadata_file_local_path, "rb") as handle:
            metadata = pickle.load(handle)

        if not {"total_tar_files", "max_files_per_tar", "tar_file_names"}.issubset(
            metadata.keys()
        ):
            logger.error(
                f"Metadata file in {self.__class__.__name__} should have following keys: \
                    total_tar_files, max_files_per_tar, tar_file_names"
            )
        return metadata

    @property
    def total_tar_files(self) -> int:
        """Total number of tar files in the dataset."""
        metadata = self._metadata()
        return metadata["total_tar_files"]

    @property
    def max_files_per_tar(self) -> int:
        """Maximum number of files inside each tar file."""
        metadata = self._metadata()
        return metadata["max_files_per_tar"]

    def get_image_text_dataset(self) -> None:
        """Override the parent class function to return nothing.

        Because tar files are downloaded on-the-fly, so any dataset specific pre-processings are skipped here.
        """
        pass

    def _get_folder_index(self, sample_index) -> int:
        """Returns the index of the folder containing the file corresponding to the given sample index.

        Args:
            sample_index: Sample index.

        Returns:
            Folder index.

        ...note:
            Each folder is expected to contain a maximum of `max_files_per_tar` files.
        """
        return sample_index // self.max_files_per_tar

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add ImgTextTarDataset dataset-specific arguments to the parser."""
        if cls == ImgTextTarDataset:
            group = parser.add_argument_group(title=cls.__name__)

            group.add_argument(
                "--dataset.multi-modal-img-text.img-text-tar.metadata-file",
                type=str,
                default=None,
                help="Location of the metadata file storing information about file indices and corresponding tar files. "
                "Defaults to None.",
            )
        return parser

    def _download_and_extract_tar_file(self, sample_index: int) -> int:
        """Download and extract the tar file.

        The tar files are pre-assumably stored in remote location (e.g., S3 bucket) and, if required, are downloaded and
        extracted to local directory @self.cache_loc. Because of distributed and multi-process training, we first extract
        them in the same location as downloaded, and then move to @self.cache_loc.

        Args:
            sample_index: Sample index.

        Returns:
            Index of the folder in which sample may be present.

        ...note:
            Each tar file may have samples less than @self.max_files_per_tar because of filtering criteria.
        """
        # Retrieve the folder index that may contain the sample.
        folder_idx = self._get_folder_index(sample_index)

        metadata_file_path = self._metadata_file_path()
        remote_directory = os.path.dirname(metadata_file_path)
        remote_file_path = f"{remote_directory}/{folder_idx}.{TAR_FILE_EXTN}"

        with open(
            f"{self.cache_loc}/{folder_idx}.{TAR_FILE_EXTN}.lock", "a"
        ) as lock_file:
            try:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                if os.path.isdir(f"{self.cache_loc}/{folder_idx}"):
                    return folder_idx

                transfer_client = self._get_transfer_client(
                    file_path=metadata_file_path
                )

                local_tar_file_path = transfer_client.download(
                    remote_file_paths=remote_file_path, dst_dir=self.cache_loc
                )

                # extract the tar file in the same location where tar file is downloaded
                tar_file_basename = os.path.basename(local_tar_file_path)
                with tarfile.open(local_tar_file_path, TAR_FILE_EXTRACTION_CODE) as tar:
                    tar.extractall(
                        path=local_tar_file_path.replace(tar_file_basename, "")
                    )

                # move extracted tar file to @self.cache_loc
                shutil.move(
                    local_tar_file_path.replace(f".{TAR_FILE_EXTN}", ""), self.cache_loc
                )

                # Delete the tar file
                if os.path.exists(local_tar_file_path):
                    os.remove(local_tar_file_path)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

        return folder_idx

    def get_image_text_dataset_pair(self, sample_index: int) -> Tuple[Image.Image, str]:
        """Get image-text pair from the dataset.

        Args:
            sample_index: Sample index.

        Returns:
            Returns a tuple of image and text caption for a given sample index.
        """

        # Check if this folder exists. If not, then download the tar file and extract it.
        folder_idx = self._download_and_extract_tar_file(sample_index=sample_index)

        file_name = f"{self.cache_loc}/{folder_idx}/{sample_index}.{SAMPLE_FILE_EXTN}"
        if not Path(file_name).exists():
            # Each tar file is supposed to have certain number of samples, but
            # it may not have all samples (because some samples may be corrupted and are filtered).
            # Therefore, if file does not exist, we randomly sample the file from a folder and return its content.
            # This helps in avoiding errors related to tensor mismatch shapes (usually arises when each GPU has different batch size)
            # when gathering the image and text embeddings from all GPUs in contrastive loss.
            files_in_folder = glob.glob(
                f"{self.cache_loc}/{folder_idx}/*.{SAMPLE_FILE_EXTN}"
            )
            assert len(files_in_folder) > 0
            file_name = random.choice(files_in_folder)

        with open(file_name, "rb") as handle:
            data = pickle.load(handle)
        img_bytes = pybase64.b64decode(data["image"], validate=True)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGBA").convert("RGB")
        if "texts" in data:
            caption_str = data["texts"]
        elif "text" in data:
            caption_str = data["text"]
        else:
            raise NotImplementedError("Text key not found.")

        return image, caption_str

    def __len__(self) -> int:
        """Number of samples in the dataset.

        The dataset comprises of multiple tar files, with each tar file containing 'max_files_per_tar'. Therefore,
        the dataset could have maximum of @total_tar_files * @max_files_per_tar samples.

        ...note:
            For evaluation, we use standard image classification datasets. Therefore, during evaluation, we return length of such
            datasets.
        """
        if self.is_training:
            return self.total_tar_files * self.max_files_per_tar
        else:
            return super().__len__()

    def extra_repr(self) -> str:
        extra_repr_str = super().extra_repr()
        return (
            extra_repr_str + f"\n\tnum_tar_files={self.total_tar_files}"
            f"\n\tmax_files_per_tar={self.max_files_per_tar}"
            f"\n\tpadding_index={self.padding_index}"
        )
