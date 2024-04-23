#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#


import argparse
import fcntl
import math
import os
import pickle
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
from pyarrow import parquet as pq
from torch import Tensor

from corenet.constants import DATA_CACHE_DIR
from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.language_modeling.base_lm import BaseLMIterableDataset
from corenet.options.parse_args import JsonValidator
from corenet.utils import logger
from corenet.utils.common_utils import construct_local_path_from_remote
from corenet.utils.download_utils import get_local_path


def check_list_of_dicts_with_mandatory_keys(
    data: List[Dict], mandatory_keys: List[str]
) -> None:
    """
    Check if a variable is a list of dictionaries, and each dictionary contains mandatory keys.

    Args:
        data: The data to check.
        mandatory_keys: The list of mandatory keys that must be present in each dictionary.

    Raises:
        Raises an error if the data is not a list of dictionaries or if any dictionary is missing mandatory keys.
    """
    if not isinstance(data, list):
        logger.error(f"Expected type of data is list. Got: {type(data)}.")

    for item in data:
        if not isinstance(item, dict):
            logger.error(f"Item in the list is not a dictionary. Got: {item}.")
        item_keys = set(item.keys())
        if not item_keys.intersection(mandatory_keys):
            logger.error(
                f"Dictionary is missing mandatory keys. Got: {item_keys}, expected: {mandatory_keys}."
            )


@DATASET_REGISTRY.register(name="general_lm", type="language_modeling")
class GeneralLMDataset(BaseLMIterableDataset):
    """
    A dataset class for general language modeling tasks.

    The class can read and yield data from variety of file formats. Currently supported formats are
    '.parquet', '.jsonl', and '.json.gz'.

    Args:
        opts: Command-line arguments.

    """

    _dataset_name = "general_lm"

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

        data_info = self._prepare_dataset()
        self.data_info = data_info

        # During training, we will store the data state. This helps in resuming
        # training in case of failures.
        self._state = None
        self._target_state = None

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Arguments related to General LM dataset."""
        if cls == GeneralLMDataset:
            group = parser.add_argument_group(cls.__name__)
            for mode in ["train", "val", "test"]:
                group.add_argument(
                    f"--dataset.language-modeling.general-lm.{mode}-data-info",
                    type=JsonValidator(List[Dict[str, Any]]),
                    default=None,
                    nargs="+",
                    help=f"Name of the parquet files for the {mode} set. Defaults to None (i.e., user needs to specify the value).",
                )
            group.add_argument(
                "--dataset.language-modeling.general-lm.data-state",
                default=None,
                type=str,
                nargs="+",
                help="A list containing the filenames that each process was processing before crash. Defaults to None.",
            )
            group.add_argument(
                "--dataset.language-modeling.general-lm.reader-chunk-size",
                type=int,
                default=1024,
                help="Number of documents to read from a dataset file at a time. Defaults to 1024.",
            )
            group.add_argument(
                "--dataset.language-modeling.general-lm.document-split-size",
                default=2048,
                type=int,
                help="The length of each sequence when splitting a larger document. Defaults to 2048 words.",
            )
            group.add_argument(
                "--dataset.language-modeling.general-lm.data-state-save-interval",
                default=15,
                type=int,
                help="Data state save interval in minutes. Defaults to 15 minutes.",
            )

        return parser

    def _reset_data_state(self) -> None:
        """Reset the data state.

        The data state has following keys:
            1. epoch: It stores the current epoch index.
            2. file: Name of the file it is currently processing.
            3. chunk: Chunk index. Note that each file may contains multiple documents, and for efficiency,
                we read them in chunks.
            4. _time: Time (in seconds) at which state is saved.
        """
        self._state = {
            "epoch": 0,
            "file": None,
            "chunk": 0,
            "_time": 0,
        }
        self._target_state = {
            "epoch": 0,
            "file": None,
            "chunk": 0,
            "_time": 0,
        }

    def extra_repr(self) -> str:
        return super().extra_repr() + (
            f"\n\tnum_files={self.num_files}"
            f"\n\tdocument_split_size={self.document_split_size}"
            f"\n\treader_chunk_size={self.reader_chunk_size}"
        )

    @property
    def reader_chunk_size(self) -> int:
        """Number of documents to read from a dataset file at a time."""
        return getattr(
            self.opts, "dataset.language_modeling.general_lm.reader_chunk_size"
        )

    @property
    def save_loc(self) -> str:
        """Location path where artifacts (e.g., data state) will be stored."""
        save_dir = getattr(self.opts, "common.results_loc")
        run_label = getattr(self.opts, "common.run_label")
        res_dir = "{}/{}".format(save_dir, run_label)
        Path(res_dir).mkdir(exist_ok=True, parents=True)
        return res_dir

    @property
    def num_files(self) -> int:
        """Total number of files."""
        return len(self.data_info["file_paths"])

    @property
    def document_split_size(self) -> int:
        """The length of each sequence when splitting a larger document."""
        return getattr(
            self.opts, "dataset.language_modeling.general_lm.document_split_size"
        )

    def _load_data_state(self) -> None:
        """Load data state.

        The data state file is read using 'dataset.language_modeling.general_lm.data_state' argument. The
        values for this argument are expected as a list, and can be specified in a configuration file or via command line.
        Recommended way to specify this argument is using configuration file as the number of state files could be
        large.
        """
        data_state_file_paths = getattr(
            self.opts, "dataset.language_modeling.general_lm.data_state"
        )
        if data_state_file_paths is not None:
            for data_state_fpath in data_state_file_paths:
                if not data_state_fpath.endswith(
                    f"data_state_{self.rank}_{self.worker_id}.pkl"
                ):
                    continue

                # Load only for the current process
                data_state_fpath = get_local_path(
                    self.opts,
                    path=data_state_fpath,
                    force_delete=False,
                    use_start_rank=False,
                    sync_ranks=False,
                )
                with open(data_state_fpath, "rb") as fh:
                    self._target_state = pickle.load(fh)

                logger.info(
                    f"Loaded dataset state {self._target_state} from {data_state_fpath} for {self.worker_id} worker on {self.rank}."
                )
                break

    def _save_data_state(self, **kwargs) -> None:
        """Save data state.

        The data states are saved for each worker on each rank. These states help us resume the
        training in case it crashes.
        """
        ((key, value),) = kwargs.items()
        state = self._state
        # The time is in seconds, so minutes are converted to seconds by multiplying with 60
        save_every_k_seconds = (
            getattr(
                self.opts,
                "dataset.language_modeling.general_lm.data_state_save_interval",
            )
            * 60
        )
        if key == "chunk":
            # We read files in chunks since each file could contain millions of documents.
            # Saving the chunk index allows us to resume training from nearly the same document in case of a failure.
            if time.time() < state["_time"] + save_every_k_seconds:
                return
            state["chunk"] = value
        elif key == "file":
            # The pre-training corpora consists of multiple files, each containing several documents.
            # We save the file name to resume training from the same file in case of a failure.
            state["chunk"] = 0
            state["file"] = value
        elif key == "epoch":
            # As workers or processes may complete iteration over files at different rates due to varying content lengths,
            # we store the epoch index to ensure correct shuffling and enable seamless resuming of training in case of failure.
            state["chunk"] = 0
            state["file"] = None
            state["epoch"] = value
        else:
            raise KeyError(f"Got unexpected key={key}.")

        state["_time"] = time.time()

        # save the file information in a file so that we can use it to resume training (if it fails)
        Path(f"{self.save_loc}/data_states").mkdir(exist_ok=True, parents=True)
        local_file_path = (
            f"{self.save_loc}/data_states/data_state_{self.rank}_{self.worker_id}.pkl"
        )
        with open(local_file_path + ".lock", "a") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                with open(
                    f"{self.save_loc}/data_states/data_state_{self.rank}_{self.worker_id}.pkl",
                    "wb",
                ) as fh:
                    pickle.dump(self._state, fh)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _prepare_dataset(self) -> Dict[str, List[str]]:
        """Prepare the dataset.

        Each dataset contains multiple files and each file could contain several documents. Specifying
        each file for each dataset is cubersome. To address this, we use file identifier in each dataset.

        Let's say that dataset, 'dummy_dataset', contain 26 files with a format '.json.gz' and it's structure
        is similar to below structure

        dummy_dataset/
        ├── dummy_0000.json.gz
        ├── dummy_0001.json.gz
        ├── dummy_0002.json.gz
        ├── ...
        ├── dummy_0025.json.gz

        Instead of individually specifying each file, we use a dictionary with three keys:
            1. "file_name": This contains the base name of the files. If multiple files are present,
                each filename is expected to include a file identifier.
            2. "text_key": Documents within each file are stored as dictionaries with various keys
                for different information, including text content and metadata. This parameter
                specifies the key within each document that corresponds to the text content.
            3. "file_id_range": This specifies the range of file identifiers present in the dataset.

        An example for above dummy dataset is given below:
        {
            "file_name": "dummy_dataset/dummy_{file_id:04d}.json.gz,
            "text_key": "text",
            "file_id_range": [0, 26]
        }

        Returns:
            A dictionary with the following information:
                1. 'file_paths': A list containing file paths.
                2. 'text_keys': A list containing text keys.

        ...note:
            Some datasets may have single file. In such a case, we do not need to specify 'file_id' as
            part of the 'file_name'. However, we still need to pass 'file_id_range' as it is an expected key.
            An example of specifying such datasets is shown below:
                {
                    "file_name": "dummy_book.jsonl,
                    "text_key": "text",
                    "file_id_range": [0, 1]
                }
        """
        opts = self.opts
        mode = self.mode
        assert mode in (
            "train",
            "val",
            "test",
        ), f"Mode should be either train or test or val. Got {mode}."
        data_info = getattr(
            opts,
            f"dataset.language_modeling.{self._dataset_name}.{mode}_data_info",
        )
        if data_info is None:
            logger.error(
                f"Please specify dataset information using 'dataset.language_modeling.{self._dataset_name}.{mode}_data_info' variable in config file. Got None."
            )
        mandatory_keys = ["file_name", "text_key", "file_id_range"]
        check_list_of_dicts_with_mandatory_keys(data_info, mandatory_keys)

        file_names = []
        text_keys = []
        for item in data_info:
            file_id_range = item["file_id_range"]
            if isinstance(file_id_range, list) and len(file_id_range) == 2:
                file_names_i = []
                for file_id in range(file_id_range[0], file_id_range[1]):
                    file_name = item["file_name"].format(file_id=file_id)
                    file_names_i.append(file_name)

                file_names.extend(file_names_i)
                # each file is expcted to have the same text key.
                text_keys.extend([item["text_key"]] * len(file_names_i))
            else:
                logger.error(
                    f"File id range is expected as a list of 2 elements. Got: {file_id_range}."
                )

        if len(file_names) != len(text_keys):
            logger.error(
                f"The number of file names does not match the number of text keys. Got: {len(file_names)} and {len(text_keys)}."
            )

        if self.shuffle_data:
            file_names, text_keys = self._shuffle_fn(
                file_names=file_names, text_keys=text_keys
            )

        return {
            "file_paths": file_names,
            "text_keys": text_keys,
        }

    def _download_if_required(self, remote_file_path: str) -> str:
        """Optionally download the files.

        This function allows us to download files from remote location (e.g., S3).

        Args:
            remote_file_path: Remote file path.

        Return:
            The local file path of the downloaded file.

        ...note:
            This repository has implemented standard transfer clients such as HTTP, HTTPS, and S3.
            However, users may utilize other data storage clients. In such cases, custom clients
            should be implemented and registered to ensure proper functionality of this function
        """

        local_file_path = construct_local_path_from_remote(
            remote_path=remote_file_path, local_dir=DATA_CACHE_DIR
        )

        with open(local_file_path + ".lock", "a") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                if os.path.isfile(local_file_path):
                    return local_file_path

                local_file_path = get_local_path(
                    opts=self.opts,
                    path=remote_file_path,
                    cache_loc=DATA_CACHE_DIR,
                    max_retries=20,
                    force_delete=False,
                    use_start_rank=False,
                    sync_ranks=False,
                )
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        return local_file_path

    def _skip_document_chunks(self, chunks: Iterable[Any]) -> Iterator[Any]:
        """Yield chunks of data from an iterable, optionally skipping chunks based on a resume chunk index.

        Args:
            chunks: An iterable containing data to be chunked.

        Yields:
            Chunks of the data.
        """
        for chunk_counter, chunk in enumerate(chunks):
            if chunk_counter < self._target_state["chunk"]:
                continue
            else:
                self._target_state["chunk"] = 0
            self._save_data_state(chunk=chunk_counter)
            yield chunk

    def _split_document_into_sequences(
        self, documents: List[str], word_separator: str = " "
    ) -> Iterator[Any]:
        """Split document into sequences.

        Some documents may be very large and contain millions of words. Tokenizing such documents
        is very slow and slows down the training. To deal with such large documents, we divide them
        into small sequences and process each sequence independently.

        Args:
            documents: List of text documents.
            word_separator: The delimiter used to separate individual words. Defaults to space.

        Yields:
            A text sequence with desired number of words.
        """
        for document in documents:
            words = document.split(word_separator)
            for i in range(0, len(words), self.document_split_size):
                text = word_separator.join(words[i : i + self.document_split_size])
                yield text

    def _read_data_from_parquet(
        self, file_path: str, text_key: str, **reader_kwargs
    ) -> Iterator[Dict[str, Tensor]]:
        """Read data from the parquet file.

        Args:
            file_path: Path of the parquet file.
            text_key: Key name corresponding to content in the parquet file.

        Yields:
            A dictionary containing 1D tensors with token indices for input samples and target labels.
            The shape of the tensors matches that of the output from the '_tokenize_text' function.
        """

        pq_table = pq.ParquetFile(file_path, **reader_kwargs)
        for document_chunks in self._skip_document_chunks(
            pq_table.iter_batches(
                batch_size=self.reader_chunk_size,
                columns=[text_key],
            )
        ):
            document_chunks_pd = document_chunks.to_pandas()[text_key]
            for text in self._split_document_into_sequences(document_chunks_pd):
                tokenized_text = self._tokenize_text(text)
                if tokenized_text is not None:
                    yield tokenized_text

    def _read_data_from_json(
        self, file_path: str, text_key: str, **reader_kwargs
    ) -> Iterator[Dict[str, Tensor]]:
        """Read data from the jsonl and json.gz files.

        When the format is 'json.gz', then reader_kwargs contain 'compression' as a key with value 'gzip'.
        See '_get_file_reader' function for reader_kwargs.

        Args:
            file_path: Path of the parquet file.

        Yields:
            A dictionary containing 1D tensors with token indices for input samples and target labels.
            The shape of the tensors matches that of the output from the '_tokenize_text' function.
        """
        document_chunks = pd.read_json(
            file_path, lines=True, chunksize=self.reader_chunk_size, **reader_kwargs
        )
        for documents in self._skip_document_chunks(document_chunks):
            # each chunk contains multiple text
            for text in self._split_document_into_sequences(documents[text_key]):
                tokenized_text = self._tokenize_text(text)
                if tokenized_text is not None:
                    yield tokenized_text

    def _get_file_reader(self, file_path: str) -> Callable:
        """Returns the function used to read a file based on its extension."""
        reader_kwargs = {}
        if file_path.endswith("parquet"):
            reader = self._read_data_from_parquet
        elif file_path.endswith("json.gz"):
            reader = self._read_data_from_json
            reader_kwargs["compression"] = "gzip"
        elif file_path.endswith("jsonl"):
            reader = self._read_data_from_json
        else:
            raise NotImplementedError(f"File format is not supported.")
        return partial(reader, **reader_kwargs)

    def generate_sample(
        self, scaled_rank: int, scaled_world_size: int
    ) -> Iterator[Any]:
        """Generate input and labels.

        Args:
            scaled_rank: Scaled rank.
            scaled_world_size: Scaled world size.

        Yields:
            Yields a dictionary containing 'samples' and 'targets' as keys corresponding to
            the input and label of a sample, respectively. The shape of the tensors matches that
            of the output from the '_tokenize_text' function.
        """
        self._reset_data_state()
        self._load_data_state()

        file_paths = self.data_info["file_paths"]
        text_keys = self.data_info["text_keys"]

        total_files = len(file_paths)
        if total_files % scaled_world_size != 0:
            padding = (
                math.ceil(total_files / scaled_world_size) * scaled_world_size
            ) - total_files

            padd_file_paths, padd_text_keys = self._shuffle_fn(
                file_names=file_paths, text_keys=text_keys, k=padding
            )

            file_paths += padd_file_paths
            text_keys += padd_text_keys
            total_files = len(file_paths)

        # split files among each node
        file_paths_process_i = file_paths[scaled_rank:total_files:scaled_world_size]
        text_keys_process_i = text_keys[scaled_rank:total_files:scaled_world_size]

        prev_file_path = None

        # some of the files may have fewer tokens compared to other files.
        # Therefore, some processes may finish before other processes. To deal with such
        # scenarios, sample data indefinitely.
        epoch_counter = -1
        while True:
            epoch_counter += 1
            if epoch_counter > 0:
                # shuffle
                file_paths_process_i, text_keys_process_i = self._shuffle_fn(
                    file_names=file_paths_process_i, text_keys=text_keys_process_i
                )
            if epoch_counter < self._target_state["epoch"]:
                # increment the epoch counter till we reach the point where current process crashed.
                continue
            else:
                self._target_state["epoch"] = 0

            for remote_file_path, text_key in zip(
                file_paths_process_i, text_keys_process_i
            ):
                if (
                    self._target_state["file"] is not None
                    and self._target_state["file"] in file_paths_process_i
                ):
                    if remote_file_path != self._target_state["file"]:
                        # skip the files till we reach the current file path that was being used before training crashed.
                        continue
                    else:
                        # We reached the file that was fully iterated before training crashed.
                        # Skip this file and reset the state
                        self._target_state["file"] = None
                        continue

                local_file_path = self._download_if_required(remote_file_path)

                logger.info(
                    f"Processsing {local_file_path} on worker {self.worker_id} of rank {self.rank}"
                )

                if prev_file_path is None:
                    prev_file_path = local_file_path

                reader = self._get_file_reader(file_path=local_file_path)

                yield from reader(file_path=local_file_path, text_key=text_key)
                self._save_data_state(file=remote_file_path)

            self._save_data_state(epoch=epoch_counter + 1)

    def _shuffle_fn(
        self, file_names: List[str], text_keys: List[str], k: Optional[int] = None
    ) -> Tuple[List[str], List[str]]:
        """Shuffle the file names and text keys.

        Args:
            file_names: List of file names.
            text_keys: List of text keys.
            k: Number of files to randomly select.

        Returns:
            A tuple containing file names and text keys. When 'k' is specified, a tuple containing k-sized list of
            file names and text keys is returned.

        ...note:
            This shuffling function is only applicable during training mode or when k is specified. For validation
            and testing when k is not specified, it does not perform any operation.
        """
        if self.is_training or k is not None:
            _temp = list(zip(file_names, text_keys))
            if k is not None:
                _temp = self._rng.choices(_temp, k=k)
            else:
                self._rng.shuffle(_temp)
            return zip(*_temp)
        return file_names, text_keys
