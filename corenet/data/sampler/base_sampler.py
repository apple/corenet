#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import copy
import math
import random
from typing import Any, Iterator, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class BaseSampler(Sampler):
    """Base class for standard and DataParallel Sampler.

    Every subclass should implement `__iter__` method, providing a way to iterate
    over indices of dataset elements.

    Args:
        opts: Command line arguments.
        n_data_samples: Number of samples in the dataset.
        is_training: Training mode or not.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        n_data_samples: int,
        is_training: bool = False,
        *args,
        **kwargs,
    ) -> None:
        # max between 1 and number of available GPUs. 1 because for supporting CPUs
        n_gpus: int = max(1, torch.cuda.device_count())
        batch_size_gpu0: int = get_batch_size_from_opts(opts, is_training=is_training)

        n_samples_per_gpu = int(math.ceil(n_data_samples * 1.0 / n_gpus))
        total_size = n_samples_per_gpu * n_gpus

        indexes = [idx for idx in range(n_data_samples)]
        # This ensures that we can divide the batches evenly across GPUs
        indexes += indexes[: (total_size - n_data_samples)]
        assert total_size == len(indexes)

        self.img_indices = indexes
        self.n_samples = total_size
        self.batch_size_gpu0 = batch_size_gpu0
        self.n_gpus = n_gpus
        self.shuffle = True if is_training else False
        self.epoch = 0

        self.num_repeats = 1
        self.trunc_rep_aug = False
        self.start_shuffling_from_epoch = getattr(
            opts, "sampler.start_shuffling_from_epoch"
        )
        if is_training:
            # enable these arguments for repeated data augmentation
            # https://openaccess.thecvf.com/content_CVPR_2020/papers/Hoffer_Augment_Your_Batch_Improving_Generalization_Through_Instance_Repetition_CVPR_2020_paper.pdf
            self.num_repeats = getattr(opts, "sampler.num_repeats")
            self.trunc_rep_aug = getattr(opts, "sampler.truncated_repeat_aug_sampler")

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BaseSampler:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        # add sampler-specific arguments
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--sampler.name",
            type=str,
            default=None,
            help=(
                "Name of the sampler. Defaults to None (i.e., user needs to specify the sampler if using MAP-style datasets)."
                "Note that this argument is not applicable to iterable datasets."
            ),
        )
        group.add_argument(
            "--sampler.num-repeats",
            type=int,
            default=1,
            help="Repeat the training dataset samples by this factor in each epoch (aka repeated augmentation). "
            "This effectively increases samples per epoch. As an example, if dataset has 10000 samples "
            "and sampler.num_repeats is set to 2, then total samples in each epoch would be 20000. "
            "Defaults to 1.",
        )

        group.add_argument(
            "--sampler.truncated-repeat-aug-sampler",
            action="store_true",
            default=False,
            help="When enabled, it restricts the sampler to load a subset of the training dataset such that"
            "number of samples obtained after repetition are the same as the original dataset."
            "As an example, if dataset has 10000 samples, sampler.num_repeats is set to 2, and "
            "sampler.truncated_repeat_aug_sampler is enabled, then the sampler would sample "
            "10000 samples in each epoch. Defaults to False.",
        )

        group.add_argument(
            "--sampler.start-shuffling-from-epoch",
            default=0,
            type=int,
            help="Shuffle data indices during training from this epoch onwards. Defaults to 0 (i.e., shuffle from the first epoch).",
        )
        return parser

    def get_indices(self) -> List[int]:
        """Returns a list of indices of dataset elements to iterate over.

        ...note:
            If repeated augmentation is enabled, then indices will be repeated.
        """
        img_indices = copy.deepcopy(self.img_indices)
        if self.shuffle:
            random.seed(self.epoch)
            if self.epoch >= self.start_shuffling_from_epoch:
                random.shuffle(img_indices)

            if self.num_repeats > 1:
                # Apply repeated augmentation
                """Assume that we have [0, 1, 2, 3] samples. With repeated augmentation,
                we first repeat the samples [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] and then select 4
                samples [0, 0, 0, 1]. Note that we do shuffle at the beginning, so samples are not the
                same at every iteration.
                """
                n_samples_before_repeat = len(img_indices)
                img_indices = np.repeat(img_indices, repeats=self.num_repeats)
                img_indices = list(img_indices)
                if self.trunc_rep_aug:
                    img_indices = img_indices[:n_samples_before_repeat]
        return img_indices

    def __iter__(self) -> Iterator[Tuple[Any, ...]]:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.img_indices) * (1 if self.trunc_rep_aug else self.num_repeats)

    def set_epoch(self, epoch: int) -> None:
        """Helper function to set epoch in each sampler."""
        self.epoch = epoch

    def update_scales(
        self, epoch: int, is_master_node: bool = False, *args, **kwargs
    ) -> None:
        """Helper function to update scales in each sampler. This is typically useful in variable-batch sampler.

        Subclass is expected to implement this function. By default, we do not do anything
        """

    def update_indices(self, new_indices: List[int]) -> None:
        """Update indices to new indices. This function might be useful for sample-efficient training."""
        self.img_indices = new_indices

    def extra_repr(self) -> str:
        extra_repr_str = (
            f"\n\t num_repeat={self.num_repeats}"
            f"\n\t trunc_rep_aug={self.trunc_rep_aug}"
        )
        return extra_repr_str

    def __repr__(self) -> str:
        return "{}({}\n)".format(self.__class__.__name__, self.extra_repr())


class BaseSamplerDDP(Sampler):
    """Base class for DistributedDataParallel Sampler.

    Every subclass should implement `__iter__` method, providing a way to iterate
    over indices of dataset elements.

    Args:
        opts: Command line arguments.
        n_data_samples: Number of samples in the dataset.
        is_training: Training or validation mode.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        n_data_samples: int,
        is_training: bool = False,
        *args,
        **kwargs,
    ) -> None:
        batch_size_gpu0: int = get_batch_size_from_opts(opts, is_training=is_training)

        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")

        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
        gpus_node_i = max(1, torch.cuda.device_count())

        num_samples_per_replica = int(math.ceil(n_data_samples * 1.0 / num_replicas))
        total_size = num_samples_per_replica * num_replicas

        img_indices = [idx for idx in range(n_data_samples)]
        img_indices += img_indices[: (total_size - n_data_samples)]
        assert len(img_indices) == total_size

        self.img_indices = img_indices
        self.n_samples_per_replica = num_samples_per_replica
        self.shuffle = True if is_training else False
        self.epoch = 0
        self.rank = rank
        self.batch_size_gpu0 = batch_size_gpu0
        self.num_replicas = num_replicas
        self.skip_sample_indices = []
        self.node_id = rank // gpus_node_i

        self.num_nodes = max(1, num_replicas // gpus_node_i)
        self.local_rank = rank % gpus_node_i
        self.num_gpus_node_i = gpus_node_i

        self.sharding = False
        self.num_repeats = 1
        self.trunc_rep_aug = False
        self.disable_shuffle_sharding = False
        if is_training:
            self.sharding = getattr(opts, "sampler.use_shards")
            self.num_repeats = getattr(opts, "sampler.num_repeats")
            self.trunc_rep_aug = getattr(opts, "sampler.truncated_repeat_aug_sampler")
            self.disable_shuffle_sharding = getattr(
                opts, "sampler.disable_shuffle_sharding"
            )

        sample_multiplier = 1 if self.trunc_rep_aug else self.num_repeats
        self.n_samples_per_replica = num_samples_per_replica * sample_multiplier
        self.start_shuffling_from_epoch = getattr(
            opts, "sampler.start_shuffling_from_epoch"
        )

    def get_indices_rank_i(self) -> List[int]:
        """Returns a list of indices of dataset elements for each rank to iterate over.

        ...note:
            1. If repeated augmentation is enabled, then indices will be repeated.
            2. If sharding is enabled, then each rank will process a subset of the dataset.
        """
        img_indices = copy.deepcopy(self.img_indices)
        if self.shuffle:
            random.seed(self.epoch)

            if self.sharding:
                """If we have 8 samples, say [0, 1, 2, 3, 4, 5, 6, 7], and we have two nodes,
                then node 0 will receive first 4 samples and node 1 will receive last 4 samples.

                note:
                    This strategy is useful when dataset is large and we want to process subset of dataset on each node.
                """

                # compute number pf samples per node.
                # Each node may have multiple GPUs
                # Node id = rank // num_gpus_per_rank
                samples_per_node = int(math.ceil(len(img_indices) / self.num_nodes))
                indices_node_i = img_indices[
                    self.node_id
                    * samples_per_node : (self.node_id + 1)
                    * samples_per_node
                ]

                # Ensure that each node has equal number of samples
                if len(indices_node_i) < samples_per_node:
                    indices_node_i += indices_node_i[
                        : (samples_per_node - len(indices_node_i))
                    ]

                # Note: For extremely large datasets, we may want to disable shuffling for efficient data loading
                if (
                    not self.disable_shuffle_sharding
                    and self.epoch >= self.start_shuffling_from_epoch
                ):
                    # shuffle the indices within a node.
                    random.shuffle(indices_node_i)

                if self.num_repeats > 1:
                    """Assume that we have [0, 1, 2, 3] samples in rank_i. With repeated augmentation,
                    we first repeat the samples [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3] and then select 4
                    samples [0, 0, 0, 1]. Note shuffling at the beginning
                    """
                    # Apply repeated augmentation
                    n_samples_before_repeat = len(indices_node_i)
                    indices_node_i = np.repeat(indices_node_i, repeats=self.num_repeats)
                    indices_node_i = list(indices_node_i)
                    if self.trunc_rep_aug:
                        indices_node_i = indices_node_i[:n_samples_before_repeat]

                # divide the samples among each GPU in a node
                indices_rank_i = indices_node_i[
                    self.local_rank : len(indices_node_i) : self.num_gpus_node_i
                ]
            else:
                """If we have 8 samples, say [0, 1, 2, 3, 4, 5, 6, 7], and we have two nodes,
                then node 0 will receive [0, 2, 4, 6] and node 1 will receive [1, 3, 4, 7].

                note:
                    This strategy is useful when each data sample is stored independently, and is
                    default in many frameworks
                """
                if self.epoch >= self.start_shuffling_from_epoch:
                    random.shuffle(img_indices)

                if self.num_repeats > 1:
                    # Apply repeated augmentation
                    n_samples_before_repeat = len(img_indices)
                    img_indices = np.repeat(img_indices, repeats=self.num_repeats)
                    img_indices = list(img_indices)
                    if self.trunc_rep_aug:
                        img_indices = img_indices[:n_samples_before_repeat]

                # divide the samples among each GPU in a node
                indices_rank_i = img_indices[
                    self.rank : len(img_indices) : self.num_replicas
                ]
        else:
            indices_rank_i = img_indices[
                self.rank : len(self.img_indices) : self.num_replicas
            ]
        return indices_rank_i

    def __iter__(self) -> Iterator[Tuple[Any, ...]]:
        raise NotImplementedError

    def __len__(self) -> int:
        return (len(self.img_indices) // self.num_replicas) * (
            1 if self.trunc_rep_aug else self.num_repeats
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BaseSamplerDDP:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--sampler.use-shards",
            action="store_true",
            default=False,
            help="Use data sharding. Only applicable to DDP. Defaults to False.",
        )
        group.add_argument(
            "--sampler.disable-shuffle-sharding",
            action="store_true",
            default=False,
            help="Disable shuffling while sharding for extremely large datasets. Defaults to False.",
        )

        return parser

    def set_epoch(self, epoch: int) -> None:
        """Helper function to set epoch in each sampler."""
        self.epoch = epoch

    def update_scales(
        self, epoch: int, is_master_node: bool = False, *args, **kwargs
    ) -> None:
        """Helper function to update scales in each sampler. This is typically useful in variable-batch sampler

        Subclass is expected to implement this function. By default, we do not do anything
        """

    def update_indices(self, new_indices: List[int]) -> None:
        """Update indices to new indices. This function might be useful for sample-efficient training."""
        self.img_indices = new_indices

    def extra_repr(self) -> str:
        extra_repr_str = (
            f"\n\t num_repeat={self.num_repeats}"
            f"\n\t trunc_rep_aug={self.trunc_rep_aug}"
            f"\n\t sharding={self.sharding}"
            f"\n\t disable_shuffle_sharding={self.disable_shuffle_sharding}"
        )
        return extra_repr_str

    def __repr__(self):
        return "{}({}\n)".format(self.__class__.__name__, self.extra_repr())


def get_batch_size_from_opts(
    opts: argparse.Namespace, is_training: bool = False
) -> int:
    """Helper function to extract batch size for training or validation/test

    Args:
        opts: command line argument
        is_training: Training or validation mode. Default: False

    Returns:
        Returns an integer
    """
    batch_size_gpu0 = int(
        getattr(opts, "dataset.train_batch_size0")
        if is_training
        else getattr(opts, "dataset.val_batch_size0")
    )
    return batch_size_gpu0
