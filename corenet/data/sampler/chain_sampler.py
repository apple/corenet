#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import argparse
import copy
import itertools
import json
from typing import Iterator, List, Mapping, Optional, Tuple

from torch.utils.data.sampler import Sampler

from corenet.data.sampler import SAMPLER_REGISTRY, build_sampler
from corenet.options.utils import flatten_yaml_as_dict
from corenet.utils import logger


@SAMPLER_REGISTRY.register(name="chain_sampler")
class ChainSampler(Sampler):
    """
    This class is a wrapper for iterating over datasets for multiple or similar tasks, typically useful for
    multi-task training.
    `task_name` and `sampler_config` are two mandatory keys that allows us to use task-specific data samplers.
    For specifying batch sizes, we use `train_batch_size0`, and `val_batch_size0` as keys for
    training and validation sets. Note that the batch sizes are scaled automatically depending on the number of GPUs.

    Args:
        opts: Command-line arguments
        data_samplers: dictionary containing different samplers

    Example::
    # Example yaml config for combining different samplers is given below.
    # Please note that configuration for each sampler should start with `-` in `chain_sampler`.

    sampler:
        name: "chain_sampler"
        chain_sampler_mode: "sequential"
        chain_sampler:
          - task_name: "segmentation"
            train_batch_size0: 10
            sampler_config:
              name: "variable_batch_sampler"
              use_shards: false
              num_repeats: 4
              truncated_repeat_aug_sampler: false
              vbs:
                crop_size_width: 512
                crop_size_height: 512
                max_n_scales: 25
                min_crop_size_width: 256
                max_crop_size_width: 768
                min_crop_size_height: 256
                max_crop_size_height: 768
                check_scale: 16
          - task_name: "classification"
            train_batch_size0: 20
            sampler_config:
              name: "batch_sampler"
              bs:
                crop_size_width: 512
                crop_size_height: 512
    """

    _SUPPORTED_SAMPLING_MODES = ["sequential", "interleave"]

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        data_samplers = ChainSampler.build_chain_sampler(opts, *args, **kwargs)
        sampling_mode = getattr(opts, "sampler.chain_sampler_mode")

        if sampling_mode is None:
            logger.error(f"Sampling mode can't be None in {self.__class__.__name__}")
        if not isinstance(sampling_mode, str):
            logger.error(
                f"Sampling mode in {self.__class__.__name__} should be a type of string. Got: {type(sampling_mode)}"
            )

        if sampling_mode not in self._SUPPORTED_SAMPLING_MODES:
            logger.error(
                f"Supported sampling mode in {self.__class__.__name__} are {self._SUPPORTED_SAMPLING_MODES}. "
                f"Got: {sampling_mode}"
            )
        self.samplers_dict = data_samplers
        self.sampling_mode = sampling_mode

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add arguments for chain sampler."""
        if cls != ChainSampler:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument("--sampler.chain-sampler", type=json.loads, action="append")
        group.add_argument(
            "--sampler.chain-sampler-mode",
            type=str,
            default="sequential",
            choices=cls._SUPPORTED_SAMPLING_MODES,
            help="Chain sampler mode. Defaults to sequential.",
        )
        return parser

    @classmethod
    def build_chain_sampler(
        cls,
        opts: argparse.Namespace,
        n_data_samples: Mapping[str, int],
        is_training: bool = False,
        *args,
        **kwargs,
    ) -> Mapping[str, Sampler]:
        """Build chain sampler from command line arguments and sampler registry
        Args:
            opts: command-line arguments
            n_data_samples: Mapping containing the task name and number of dataset samples in task-specific dataset
            is_training: Training mode or not

        Returns:
            A dictionary, sampler_dict, containing information about sampler name and module.
        """

        chain_sampler_opts = getattr(opts, "sampler.chain_sampler")

        if chain_sampler_opts is None:
            logger.error(
                f"sampler.chain_sampler in {cls.__name__} can't be None. Please specify "
                f"sampler.chain_sampler using a yaml file."
            )

        if not isinstance(chain_sampler_opts, List):
            logger.error(
                f"Chain sampler options are expected as a List. "
                f"Got type: {type(chain_sampler_opts)} and values: {chain_sampler_opts}"
            )

        num_samplers = len(chain_sampler_opts)
        if num_samplers < 1:
            logger.error("We need at least one sampler if using chain sampler")
        sampler_dict = {}

        for i, sampler_opts_ in enumerate(chain_sampler_opts):
            task_name = sampler_opts_.get("task_name", None)
            if task_name is None:
                logger.error("task_name is a mandatory key when using chain sampler")

            # get sampler configuration
            sampler_opts_as_dict = sampler_opts_.get("sampler_config", None)

            if sampler_opts_as_dict is None:
                logger.error(
                    "sampler_config is a mandatory key when using chain sampler"
                )

            train_batch_size = sampler_opts_.get("train_batch_size0", None)
            val_batch_size = sampler_opts_.get("val_batch_size0", None)

            # flatten the dictionary
            sampler_opts_as_dict = flatten_yaml_as_dict(sampler_opts_as_dict)

            # create a local copy and override the global opts with task-specific sampler opts
            sampler_opts = copy.deepcopy(opts)

            # `sampler_opts_as_dict` only contains the values of command-line arguments that are
            # defined in the yaml file. Therefore, if a user misses few arguments, we won't have access
            # to default values, leading to an error. To avoid this, we create a local copy of global
            # command-line arguments and update it with `sampler_opts_as_dict` arguments
            for k, v in sampler_opts_as_dict.items():
                # we need to prefix each argument with sampler because we define individual samplers as
                # `sampler.vbs.*` and not `vbs.*`
                setattr(sampler_opts, "sampler." + k, v)

            # override the batch size of sampler
            if train_batch_size is not None:
                setattr(sampler_opts, "dataset.train_batch_size0", train_batch_size)
            if val_batch_size is not None:
                setattr(sampler_opts, "dataset.val_batch_size0", val_batch_size)

            if not isinstance(n_data_samples, Mapping):
                logger.error(
                    "For chain sampler, we need n_data_samples as a dictionary with key as a task name "
                    f"and value as number of data points. Got: {n_data_samples}"
                )

            if task_name not in n_data_samples:
                logger.error(
                    f"Sample mapping from dataset has following keys ({n_data_samples.keys()}) "
                    f"and does not contain {task_name}. Please check."
                )

            # build the sampler for the task
            sampler_dict[task_name] = build_sampler(
                opts=sampler_opts,
                n_data_samples=n_data_samples[task_name],
                is_training=is_training,
                *args,
                **kwargs,
            )
        # see if the keys in n_data_samples and sampler_dict are the same or not
        # i.e., intersection is null.
        is_intersection = n_data_samples.keys().isdisjoint(sampler_dict)
        assert is_intersection is False, (
            f"The keys in n_data_samples and sampler_dict are not the same. "
            f"Got: {n_data_samples.keys()} and {sampler_dict.keys()}"
        )
        return sampler_dict

    def _sequential_sampling(self) -> List[Tuple]:
        """Assuming we have samples from N datasets, then this function first iterates over
        the entire first dataset, then the entire second dataset, and so on

        Example:
            Dataset 1: [A, B, C]
            Dataset 2: [D, E, F, G, H]

            The result of this sampler would be something like this [A, B, C, D, E, F, G, H]

        """
        for task_name, dataset_sampler in self.samplers_dict.items():
            for batch_data in dataset_sampler:
                # append dataset name to the batch data
                yield [x + (task_name,) for x in batch_data]

    def _interleave_sampling(self) -> List[Tuple]:
        """Assuming we have samples from N datasets, then this function yields a batch from first dataset,
        then a batch from second dataset, and so on. In other words, batches are sampled from N datasets
        in a round-robin fashion.

        Example:
            Dataset 1: [A, B, C]
            Dataset 2: [D, E, F, G, H]

            The result of this sampler would be [A, D, B, E, C, F, G, H]
        """

        items = self.samplers_dict.items()
        task_names, sampler_names = zip(*items)
        num_active_samplers = len(sampler_names)
        next_samplers = itertools.cycle(
            iter(data_sampler).__next__ for data_sampler in sampler_names
        )

        while num_active_samplers:
            try:
                for i, next_sampler in enumerate(next_samplers):
                    yield [
                        x + (task_names[i % num_active_samplers],)
                        for x in next_sampler()
                    ]
            except StopIteration:
                # Remove the sampler that we just exhausted from the cycle.
                num_active_samplers -= 1
                next_samplers = itertools.cycle(
                    itertools.islice(next_samplers, num_active_samplers)
                )

    def __iter__(self) -> Iterator[Tuple]:
        if self.sampling_mode == "sequential":
            return self._sequential_sampling()
        elif self.sampling_mode == "interleave":
            return self._interleave_sampling()

    def __len__(self) -> int:
        return sum(
            [
                len(dataset_sampler)
                for task_name, dataset_sampler in self.samplers_dict.items()
            ]
        )

    def set_epoch(self, epoch: int) -> None:
        """Helper function to set epoch in each sampler.
        Args:
            epoch: Current epoch

        Returns:
            Nothing
        """
        for task_name, dataset_sampler in self.samplers_dict.items():
            if hasattr(dataset_sampler, "set_epoch"):
                dataset_sampler.set_epoch(epoch)

    def update_scales(
        self, epoch: int, is_master_node: Optional[bool] = False, *args, **kwargs
    ) -> None:
        """Helper function to update scales in each sampler. This is typically useful for variable-batch samplers

        Args:
            epoch: Current epoch
            is_master_node: Master node or not.

        Returns:
            Nothing
        """
        for task_name, dataset_sampler in self.samplers_dict.items():
            if hasattr(dataset_sampler, "update_scales"):
                dataset_sampler.update_scales(
                    epoch, is_master_node=is_master_node, *args, **kwargs
                )

    def update_indices(self, new_indices: List[int]) -> None:
        """Update sample indices of the datasets with these new indices.

        Args:
            new_indices: Filtered indices of the samples that needs to be used in next epoch.

        Returns:
            Nothing

        ...note:
            This function is useful for sample-efficient training. This function may be implemented
            in future (depending on use-case)
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}(\n"
        for k, v in self.samplers_dict.items():
            repr_str += f"{k} --> " + v.__repr__().replace("\n\t", "\n\t\t").replace(
                "\n)", "\n\t)"
            )
            repr_str += "\n"
        repr_str += "\n)"
        return repr_str
