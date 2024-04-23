#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#


import argparse
import fcntl
import os
from typing import Dict, Iterator

import pandas as pd
from torch import Tensor

from corenet.constants import DATA_CACHE_DIR
from corenet.data.datasets import DATASET_REGISTRY
from corenet.data.datasets.language_modeling.base_lm import BaseLMIterableDataset
from corenet.utils.common_utils import construct_local_path_from_remote
from corenet.utils.download_utils import get_local_path


@DATASET_REGISTRY.register(name="commonsense_170k", type="language_modeling")
class CommonSense170k(BaseLMIterableDataset):
    """
    The CommonSense 170k dataset, as defined in LLM-Adapters (https://arxiv.org/pdf/2304.01933.pdf).

    This is a simple concatenation of:
      - boolq
      - piqa
      - social_i_qa
      - hellaswag
      - winogrande
      - ARC-Easy
      - ARC-Challenge
      - openbookqa

    CommonSense170k processes prompts uniformly for all of its sub-datasets.
    See @generate_prompt_and_response.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

        self.json_path = getattr(
            opts, "dataset.language_modeling.commonsense_170k.path"
        )
        if self.json_path is None:
            raise ValueError(
                "--dataset.language-modeling.commonsense-170k.path " "is required."
            )
        json_path = self._download_if_required(self.json_path)
        self.data = pd.read_json(json_path)

    def generate_sample(
        self, scaled_rank: int, scaled_world_size: int
    ) -> Iterator[Dict[str, Tensor]]:
        num_elems = len(self.data)
        chosen_elems = list(range(scaled_rank, num_elems, scaled_world_size))
        self._rng.shuffle(chosen_elems)
        shuffled_data = self.data.loc[chosen_elems]

        for sample in shuffled_data.iterrows():
            sample = sample[1].to_dict()
            sample = generate_prompt_and_response(sample)
            tokenized_sample = self._tokenize_text(sample)
            yield tokenized_sample

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls == CommonSense170k:
            group = parser.add_argument_group(cls.__name__)
            group.add_argument(
                "--dataset.language-modeling.commonsense-170k.path",
                default=None,
                help="Path to the commonsense 170k json dataset file. "
                "Default is None. Note, the dataset file is currently "
                "available in the LLM-Adapters repository at "
                "https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main"
                "/ft-training_set/commonsense_170k.json.",
            )
        return parser

    def _download_if_required(self, remote_file_path: str) -> str:
        """
        Download @remote_file_path if it isn't already downloaded.

        Args:
            remote_file_path: The file to possibly download.
        Returns:
            The local path to the file.
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


def generate_prompt_and_response(data_point: Dict[str, str]) -> str:
    """
    Generate the prompt and response for a given data point.

    Adapted from LLM-Adapters: https://github.com/AGI-Edgerunners/LLM-Adapters

    Args:
        data_point: A data point with the following keys: "instruction",
            "input", "output", and "answer". The "input" field can contain
            an empty string if an input is not used by the evaluation.
    """
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}
                
                ### Input:
                {data_point["input"]}
                
                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}
                
                ### Response:
                {data_point["output"]}"""  # noqa: E501
