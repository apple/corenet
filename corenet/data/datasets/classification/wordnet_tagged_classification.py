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
from pathlib import Path
from typing import Any, List, Mapping, Tuple
from urllib.parse import urlsplit

import pybase64
import torch
from PIL import Image, ImageFile

try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.corpus.reader.wordnet import Synset
    from nltk.stem import WordNetLemmatizer

    NLTK_INSTALLED = True
except ModuleNotFoundError:
    wn = None
    Synset = None
    WordNetLemmatizer = None

    NLTK_INSTALLED = False

from corenet.constants import DATA_CACHE_DIR
from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.classification.base_image_classification_dataset import (
    BaseImageClassificationDataset,
)
from corenet.data.datasets.dataset_base import BaseImageDataset
from corenet.data.datasets.utils.text import caption_preprocessing
from corenet.data.io.transfer_clients import BaseClient, get_transfer_client
from corenet.data.transforms import BaseTransformation
from corenet.utils import logger
from corenet.utils.download_utils import get_local_path

# To enable reading truncated images, we update the default values of following variables in PIL
# TODO: Investigate later if below Image flags can be moved to where Image is read.
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

TAR_FILE_EXTN = "tar.gz"
TAR_FILE_EXTRACTION_CODE = "r:gz"
SAMPLE_FILE_EXTN = "pkl"


def extract_pos_offset_info_from_synset(synset: Synset) -> str:
    """Extracts part-of-speech and offset information from the input @synset.

    Args:
        synset: WordNet synset.

    Returns:
        A string containing part-of-speech and offset information about the synset.
    """
    offset = synset.offset()
    pos = synset.pos()
    return f"{pos}{offset}"


def check_valid_noun_synset(synset: Synset, word: str) -> bool:
    """Check if input synset and word are the same.

    Args:
        synset: Input synset.
        word: Input word.

    Returns:
        A boolean indicating if input synset and word are the same or not.
    """
    return synset.name() == f"{word}.n.01"


@DATASET_REGISTRY.register(name="wordnet_tagged_classification", type="classification")
class WordnetTaggedClassificationDataset(BaseImageDataset):
    """WordNet tagged classification dataset.

    This class converts the image-text dataset into multi-label classification dataset. The expected data structure of
    the input data should be the same as 'corenet.data.datasets.multi_modal_img_text.img_text_tar_dataset.ImgTextTarDataset'

    Args:
        opts: Command-line arguments.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        if not NLTK_INSTALLED:
            logger.error(
                "Please install NLTK library using 'pip install nltk==3.8.1' and 'python3 -m nltk.downloader all' commands."
            )

        super().__init__(opts=opts, *args, **kwargs)
        self._transfer_client = None
        self.vocab = self._get_vocab()
        self.__post_init__()

    def __post_init__(self) -> None:
        """Post init checks."""
        if not self.is_training:
            raise NotImplementedError("Only training is supported for now.")

    def _get_vocab(self) -> List[str]:
        """Retrieves the vocabulary as a list.

        The vocabulary is structured as a dictionary where synsets are represented as keys in pos-offset format,
        and their corresponding frequencies in the dataset are stored as values. The key-value pairs are arranged
        in descending order based on the frequencies. An example is shown below:

        {
            "n5928118": 52418327,
            "n13333833": 46393897,
            "n4960277": 38781582,
            "n7947958": 36532096,
            "n9638875": 34564013,
            "n928077": 30290822,
            "n7996689": 28076676,
            "n10787470": 24182531,
            "n5938976": 23817664,
            "n8559508": 23476398
        }

        Returns:
            A list containing the name of top-k synsets. The value of 'k' is specified using
            'dataset.wordnet_tagged_classification.vocab_size' argument.
        """
        vocab_file_path = getattr(
            self.opts, "dataset.wordnet_tagged_classification.vocab_file"
        )
        if vocab_file_path is None:
            logger.error(f"Vocab path can't be None in {self.__class__.__name__}.")

        vocab_file_path = get_local_path(
            self.opts,
            path=vocab_file_path,
            force_delete=False,
            use_start_rank=True,
            sync_ranks=False,
        )

        with open(vocab_file_path, "rb") as f:
            vocab = pickle.load(f)

        vocab_size = getattr(
            self.opts, "dataset.wordnet_tagged_classification.vocab_size"
        )
        if vocab_size is None or vocab_size < 0:
            logger.error(
                f"Vocabulary size should be a positive number. Got: {vocab_size}. Please specify by 'dataset.wordnet_tagged_classification.vocab_size' argument."
            )
        return list(vocab.keys())[:vocab_size]

    def _metadata_file_path(self) -> str:
        """Returns metadata file path from command-line arguments."""
        opts = self.opts

        metadata_file_path = getattr(
            opts, f"dataset.wordnet_tagged_classification.metadata_file"
        )

        if not metadata_file_path:
            logger.error(
                f"Please specify metadata file path using 'dataset.wordnet_tagged_classification.metadata_file'."
            )
        return metadata_file_path

    def _metadata(self):
        """Reads the metadata content.

        ...note:
            The metadata file is expected to have following keys:
            1. total_tar_files: Total number of tar files in the dataset.
            2. max_files_per_tar: Maximum number of files inside each tar.
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

    def _download_and_extract_tar_file(self, sample_index: int) -> int:
        """Downloads and extracts the tar file.

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
        folder_idx = sample_index // self.max_files_per_tar

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

    def _convert_caption_to_labels(self, captions_str: str) -> List[int]:
        """Converts the caption into multi-class labels.

        The input caption is tokenized into words, and noun synsets are extracted for each word. Subsequently, the
        parts of speech (POS) and offsets of the extracted noun synsets are compared with those in the vocabulary
        to generate a list of multi-class labels.

        Args:
            captions_str: Input caption as a string.

        Returns:
            A list of integers, where each integer corresponds to the index of the matching synset in the vocabulary.
            In case there are no matching synsets, an empty list is returned.
        """
        captions_str = caption_preprocessing(captions_str)
        # process caption and find synsets

        tagged_words = nltk.pos_tag(nltk.word_tokenize(captions_str))
        lemmatzr = WordNetLemmatizer()
        labels = []
        for word, pos in tagged_words:
            # use lemmatizer to reduce text ambiguity.
            # words like bicycle and bicycles are converted to bicycle
            try:
                word = lemmatzr.lemmatize(word)
                noun_synset = wn.synset(f"{word}.n.01")
            except Exception as e:
                # No lemma 'is' with part of speech 'n', then nltk.corpus.reader.wordnet.WordNetError is raised.
                # Skip such cases
                continue

            if not check_valid_noun_synset(noun_synset, word):
                continue
            noun_synset = extract_pos_offset_info_from_synset(noun_synset)
            if noun_synset in self.vocab:
                # add the indices of the labels
                labels.append(self.vocab.index(noun_synset))
        return labels

    def _read_sample_with_wordnet_label_mining(
        self, sample_index: int
    ) -> Tuple[Image.Image, List[str]]:
        """Reads the sample with WordNet derived labels.

        The function extracts the image and caption corresponding to input @sample_index. It then
        converts the caption into multi-class labels.

        Args:
            sample_index: Sample index.

        Returns:
            Returns a tuple of image and mult-class labels for a given sample index.
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

        labels = self._convert_caption_to_labels(captions_str=caption_str)
        return image, labels

    def _training_transforms(self, *args, **kwargs) -> BaseTransformation:
        """Image transformations to be applied on input image during training.

        See 'BaseImageClassificationDataset' for the supported transformations for the classification task.
        """
        return BaseImageClassificationDataset._training_transforms(
            self, *args, **kwargs
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add dataset-specific arguments to the parser."""
        if cls == WordnetTaggedClassificationDataset:
            group = parser.add_argument_group(title=cls.__name__)

            group.add_argument(
                "--dataset.wordnet-tagged-classification.vocab-file",
                type=str,
                default=None,
                help="Location of vocab pickle file. Defaults to None.",
            )

            group.add_argument(
                "--dataset.wordnet-tagged-classification.metadata-file",
                type=str,
                default=None,
                help="Metadata file containing information about img-text pairs. Defaults to None.",
            )

            group.add_argument(
                "--dataset.wordnet-tagged-classification.vocab-size",
                type=int,
                default=None,
                help="Vocabulary threshold. Synsets in the ordered vocabulary dictionary beyond this threshold will not be used. Defaults to None (i.e., user needs to specify the value).",
            )
        return parser

    def _get_transfer_client(self, file_path: str) -> BaseClient:
        """Get transfer client for a given file path.

        Args:
            file_path: File path.

        Returns:
            An instance of BaseClient.

        ...note:
            Some of the clients are not pickle-able (e.g., S3). Therefore, this function should not be
            called inside the '__init__' function.
        """
        if self._transfer_client is None:
            opts = self.opts
            client_name = urlsplit(file_path).scheme.lower()

            self._transfer_client = get_transfer_client(
                opts,
                transfer_client_name=client_name,
                force_delete=False,
                only_download_on_start_rank=False,
                synchronize_distributed_ranks=False,
                parallel_download=False,
            )
        return self._transfer_client

    def __getitem__(
        self, sample_size_and_index: Tuple[int, int, int]
    ) -> Mapping[str, Any]:
        """Returns the sample corresponding to the input sample index.

        Returned sample is transformed into the size specified by the input.

        Args:
            sample_size_and_index: Tuple of the form (crop_size_h, crop_size_w, sample_index).

        Returns:
            A dictionary with 'samples', 'targets', and 'sample_id' as keys corresponding to input,
             label, and index of a sample, respectively.

        Shapes:
            The shape of values in output dictionary, output_data, are as follows:

            output_data["samples"]: Shape is [Channels, Height, Width]
            output_data["targets"]: Shape is [vocab_size]
            output_data["sample_id"]: Shape is [1]
        """

        crop_size_h, crop_size_w, sample_index = sample_size_and_index
        transform_fn = self.get_augmentation_transforms(size=(crop_size_h, crop_size_w))

        image, labels = self._read_sample_with_wordnet_label_mining(sample_index)

        # convert labels to one hot vector
        targets = torch.zeros((self.vocab_size), dtype=torch.long)
        if labels is not None and len(labels) > 0:
            targets[labels] = 1

        output_data = {
            "samples": transform_fn({"image": image})["image"],
            "targets": targets,
            "sample_id": sample_index,
        }
        return output_data

    def __len__(self) -> int:
        return self.total_tar_files * self.max_files_per_tar

    @property
    def cache_loc(self) -> str:
        return DATA_CACHE_DIR

    @property
    def vocab_size(self):
        return len(self.vocab)

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

    def extra_repr(self) -> str:
        return super().extra_repr() + (
            f"\n\ttotal_tar_files={self.total_tar_files}"
            f"\n\tmax_files_per_tar={self.max_files_per_tar}"
            f"\n\tnum_synsets={self.vocab_size}"
        )
