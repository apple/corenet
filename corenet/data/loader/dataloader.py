#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import sys
from typing import Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader

from corenet.data.datasets.dataset_base import BaseDataset, BaseIterableDataset
from corenet.data.sampler import Sampler


class CoreNetDataLoader(DataLoader):
    """Data loader class that combines a dataset and a sampler, and provides an iterable over
    the given dataset.

    Args:
        dataset: Dataset from which to load the data.
        batch_size: How many samples per batch to load.
        batch_sampler: Defines the strategy to
            returns a batch of indices at a time.
        num_workers: Number of subprocesses to use for data loading. Defaults to 1.
        pin_memory: If ``True``, the data loader will copy Tensors into device/CUDA pinned
            memory before returning them.
        persistent_workers: If ``True``, the data loader will not shutdown
            the worker processes after a dataset has been consumed once. This allows to
            maintain the workers `Dataset` instances alive. Defaults to False.
        collate_fn: Merges a list of samples to form a mini-batch of Tensor(s). Typically used
            when using batched loading from a map-style dataset.
        prefetch_factor: Number of batches loaded in advance by each worker. The value of ``2`` means
            there will be a total of 2 * num_workers batches prefetched across all workers. Default value depends
            on the value for num_workers. If num_workers=0, then the default value of prefetch_factor is ``None``.
            Otherwise, it is ``2`` for num_workers>0.
    """

    def __init__(
        self,
        dataset: Union[BaseDataset, BaseIterableDataset],
        batch_size: int,
        batch_sampler: Sampler,
        num_workers: Optional[int] = 1,
        pin_memory: Optional[bool] = False,
        persistent_workers: Optional[bool] = False,
        collate_fn: Optional[
            Callable[[List[Union[Dict, torch.Tensor]]], Union[Dict, torch.Tensor]]
        ] = None,
        prefetch_factor: Optional[int] = None,
        *args,
        **kwargs
    ):
        if num_workers == 0 and prefetch_factor is not None:
            # prefecting can only be done during multiprocessing, so disabling it.
            prefetch_factor = None
        if num_workers > 0 and prefetch_factor is None:
            # setting prefetch factor to 2 (same as PyTorch's default value for this condition.)
            prefetch_factor = 2

        super(CoreNetDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=collate_fn,
            prefetch_factor=prefetch_factor,
            *args,
            **kwargs
        )

    def update_indices(self, new_indices: List, *args, **kwargs):
        """Update indices in the dataset class"""
        if hasattr(self.batch_sampler, "img_indices") and hasattr(
            self.batch_sampler, "update_indices"
        ):
            self.batch_sampler.update_indices(new_indices)

    def __len__(self) -> int:
        """Number of samples in the dataset.

        ...note:
            The length of Iterable datasets is set to 'sys.maxsize' because '__len__' could be
            inaccurate with naive multi-processing data loading for such datasets, as samples may
            be duplicated. Similar to PyTorch's recommendation, we also discourage the use of '__len__'
            for iterable datasets, and generally trust the corresponding iterable dataset class for
            correct implementation.
        """
        return (
            sys.maxsize
            if isinstance(self.dataset, BaseIterableDataset)
            else len(self.dataset)
        )

    def get_sample_indices(self) -> List:
        """Sample IDs"""
        return self.batch_sampler.img_indices
