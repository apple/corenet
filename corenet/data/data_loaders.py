#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from functools import partial
from typing import Mapping, Optional, Tuple, Union

from torch.utils.data import IterableDataset
from torch.utils.data.sampler import Sampler

from corenet.data.collate_fns import build_collate_fn, build_test_collate_fn
from corenet.data.datasets import BaseDataset, get_test_dataset, get_train_val_datasets
from corenet.data.loader.dataloader import CoreNetDataLoader
from corenet.data.sampler import build_sampler
from corenet.utils import logger
from corenet.utils.ddp_utils import is_master
from corenet.utils.tensor_utils import image_size_from_opts


def create_test_loader(opts: argparse.Namespace) -> CoreNetDataLoader:
    """Helper function to create and return a dataset loader for test dataset from command-line arguments"""
    is_master_node = is_master(opts)
    test_dataset = get_test_dataset(opts)
    if isinstance(test_dataset, IterableDataset):
        test_sampler = None
        batch_size = getattr(opts, "dataset.eval_batch_size0")
        assert batch_size > 0, (
            f"For iterable datasets, we need batch size > 0 but got {batch_size}. "
            f"Please specify batch size using 'dataset.eval_batch_size0' argument in the config file."
        )
    else:
        n_test_samples = get_num_data_samples_as_int_or_mapping(test_dataset)

        # The sampler reads the batch size for validation/test jobs using 'dataset.val_batch_size0'
        # So, we override the value of validation batch size argument with evaluation batch size argument.
        setattr(
            opts,
            "dataset.val_batch_size0",
            getattr(opts, "dataset.eval_batch_size0"),
        )

        # we don't need variable batch sampler for evaluation
        sampler_name = getattr(opts, "sampler.name", "batch_sampler")
        crop_size_h, crop_size_w = image_size_from_opts(opts)
        if sampler_name.find("var") > -1:
            setattr(opts, "sampler.name", "batch_sampler")
            setattr(opts, "sampler.bs.crop_size_width", crop_size_w)
            setattr(opts, "sampler.bs.crop_size_height", crop_size_h)

        test_sampler = build_sampler(
            opts=opts,
            n_data_samples=n_test_samples,
            is_training=False,
            get_item_metadata=test_dataset.get_item_metadata,
        )
        # for non-iterable dataset, batch size is handled inside the sampler.
        batch_size = 1
        if is_master_node:
            logger.log(f"Evaluation sampler details: {test_sampler}")

    collate_fn_test = build_test_collate_fn(opts=opts)

    data_workers = getattr(opts, "dataset.workers")
    persistent_workers = getattr(opts, "dataset.persistent_workers") and (
        data_workers > 0
    )
    pin_memory = getattr(opts, "dataset.pin_memory")

    test_loader = CoreNetDataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        batch_sampler=test_sampler,
        num_workers=data_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=(
            partial(collate_fn_test, opts=opts) if collate_fn_test is not None else None
        ),
    )
    return test_loader


def create_train_val_loader(
    opts: argparse.Namespace,
) -> Tuple[CoreNetDataLoader, Optional[CoreNetDataLoader], Sampler]:
    """Helper function to create training and validation data loaders.

    Args:
        opts: Command-line arguments

    Returns:
        A tuple containing training data loader, (optional) validation data loader, and training data sampler.
    """
    is_master_node = is_master(opts)
    train_dataset, valid_dataset = get_train_val_datasets(opts)

    if isinstance(train_dataset, IterableDataset):
        train_sampler = None
        train_batch_size = getattr(opts, "dataset.train_batch_size0")
        assert train_batch_size > 0, (
            f"For iterable datasets, we need batch size > 0 but got {train_batch_size}. "
            f"Please specify batch size using 'dataset.train_batch_size0' argument in the config file."
        )
    else:
        n_train_samples = get_num_data_samples_as_int_or_mapping(train_dataset)
        train_sampler = build_sampler(
            opts=opts,
            n_data_samples=n_train_samples,
            is_training=True,
            get_item_metadata=train_dataset.get_item_metadata,
        )
        # for non-iterable dataset, batch size is handled inside the sampler.
        train_batch_size = 1

    if valid_dataset is None:
        # Validation is disabled.
        valid_sampler = None
        val_batch_size = 1
    elif isinstance(valid_dataset, IterableDataset):
        # validation dataset is iterable
        valid_sampler = None
        val_batch_size = getattr(opts, "dataset.val_batch_size0")
        assert val_batch_size > 0, (
            f"For iterable datasets, we need batch size > 0 but got {val_batch_size}. "
            f"Please specify batch size using 'dataset.val_batch_size0' argument in the config file."
        )
    else:
        # validation dataset is map-style
        n_valid_samples = get_num_data_samples_as_int_or_mapping(valid_dataset)
        valid_sampler = build_sampler(
            opts=opts,
            n_data_samples=n_valid_samples,
            is_training=False,
            get_item_metadata=valid_dataset.get_item_metadata,
        )
        # for non-iterable dataset, batch size is handled inside the sampler.
        val_batch_size = 1

    data_workers = getattr(opts, "dataset.workers")
    persistent_workers = getattr(opts, "dataset.persistent_workers") and (
        data_workers > 0
    )
    pin_memory = getattr(opts, "dataset.pin_memory")
    prefetch_factor = getattr(opts, "dataset.prefetch_factor")

    collate_fn_train, collate_fn_val = build_collate_fn(opts=opts)

    train_loader = CoreNetDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        num_workers=data_workers,
        pin_memory=pin_memory,
        batch_sampler=train_sampler,
        persistent_workers=persistent_workers,
        collate_fn=(
            partial(collate_fn_train, opts=opts)
            if collate_fn_train is not None
            else None
        ),
        prefetch_factor=prefetch_factor,
    )

    if valid_dataset is not None:
        val_loader = CoreNetDataLoader(
            dataset=valid_dataset,
            batch_size=val_batch_size,
            batch_sampler=valid_sampler,
            num_workers=data_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            collate_fn=(
                partial(collate_fn_val, opts=opts)
                if collate_fn_val is not None
                else None
            ),
        )
    else:
        val_loader = None

    if is_master_node:
        if train_sampler is not None:
            logger.log(f"Training sampler details: {train_sampler}")

        if valid_sampler is not None:
            logger.log(f"Validation sampler details: {valid_sampler}")

        logger.log("Number of data workers: {}".format(data_workers))

    return train_loader, val_loader, train_sampler


def get_num_data_samples_as_int_or_mapping(
    dataset: BaseDataset,
) -> Union[int, Mapping[str, int]]:
    """Return the number of samples in the dataset.

    The dataset can be a single or composition of multiple datasets (as in multi-task learning). For a single
    dataset, the number of samples is integer while for multiple datasets, a dictionary is returned with task name and
    number of samples per task.

    Args:
        dataset: An instance of `corenet.data.datasets.BaseDataset` class

    Returns:
        An integer for single dataset and mapping for composite datasets.

    """
    if hasattr(dataset, "get_dataset_length_as_mapping"):
        return dataset.get_dataset_length_as_mapping()
    else:
        return len(dataset)
