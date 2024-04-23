#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Any, List, Mapping

import torch
from torch import Tensor
from torch.utils.data import default_collate

from corenet.data.collate_fns import COLLATE_FN_REGISTRY
from corenet.utils import logger


@COLLATE_FN_REGISTRY.register(name="pytorch_default_collate_fn")
def pytorch_default_collate_fn(batch: Any, *args, **kwargs) -> Any:
    """A wrapper around PyTorch's default collate function."""
    batch = default_collate(batch)
    return batch


@COLLATE_FN_REGISTRY.register(name="unlabeled_image_data_collate_fn")
def unlabeled_image_data_collate_fn(
    batch: List[Mapping[str, Any]], opts: argparse.Namespace
) -> Mapping[str, Any]:
    """
    Combines a list of dictionaries into a single dictionary by concatenating matching fields.

    Each input dictionary is expected to have items with `samples` and `sample_id` as keys. The value for
    `samples` is expected to be a tensor and the value for `sample_id` is expected to be an integer.

    This function adds `targets` field to the output dictionary with dummy values to meet the expectations of
    training engine.

    Args:
        batch: A list of dictionaries
        opts: An argparse.Namespace instance.

    Returns:
        A dictionary with `samples`, `sample_id` and `targets` as keys.
    """
    batch_size = len(batch)
    sample_size = [batch_size, *batch[0]["samples"].shape]
    img_dtype = batch[0]["samples"].dtype

    samples = torch.zeros(size=sample_size, dtype=img_dtype)
    sample_ids = torch.zeros(size=[batch_size], dtype=torch.long)
    for i, batch_i in enumerate(batch):
        samples[i] = batch_i["samples"]
        sample_ids[i] = batch_i["sample_id"]

    channels_last = getattr(opts, "common.channels_last")
    if channels_last:
        samples = samples.to(memory_format=torch.channels_last)

    # Add dummy labels to meet the expectations of training engine.
    dummy_labels = torch.full(size=[batch_size], fill_value=0, dtype=torch.long)

    return {"samples": samples, "sample_id": sample_ids, "targets": dummy_labels}


@COLLATE_FN_REGISTRY.register(name="image_classification_data_collate_fn")
def image_classification_data_collate_fn(
    batch: List[Mapping[str, Any]], opts: argparse.Namespace
) -> Mapping[str, Any]:
    """Combines a list of dictionaries into a single dictionary by concatenating matching fields.

    Each input dictionary is expected  to have items with `samples`,`sample_id` and `targets` as keys. The value for
    `samples` is expected to be a tensor and the values for `sample_id` and `targets` are expected to be integers.

    Args:
        batch: A list of dictionaries
        opts: An argparse.Namespace instance.

    Returns:
        A dictionary with `samples`, `sample_id` and `targets` as keys.
    """
    batch_size = len(batch)
    img_size = [batch_size, *batch[0]["samples"].shape]
    img_dtype = batch[0]["samples"].dtype

    images = torch.zeros(size=img_size, dtype=img_dtype)
    sample_ids = torch.zeros(size=[batch_size], dtype=torch.long)
    labels = torch.full(size=[batch_size], fill_value=-1, dtype=torch.long)
    valid_indexes = []
    for i, batch_i in enumerate(batch):
        images[i] = batch_i["samples"]
        sample_ids[i] = batch_i["sample_id"]

        label_i = batch_i["targets"]
        labels[i] = label_i
        if label_i != -1:
            valid_indexes.append(i)

    valid_indexes = torch.tensor(valid_indexes, dtype=torch.long)
    images = torch.index_select(images, dim=0, index=valid_indexes)
    sample_ids = torch.index_select(sample_ids, dim=0, index=valid_indexes)
    labels = torch.index_select(labels, dim=0, index=valid_indexes)

    channels_last = getattr(opts, "common.channels_last")
    if channels_last:
        images = images.to(memory_format=torch.channels_last)

    return {"samples": images, "targets": labels, "sample_id": sample_ids}


@COLLATE_FN_REGISTRY.register(name="default_collate_fn")
def default_collate_fn(
    batch: List[Mapping[str, Tensor]], opts: argparse.Namespace
) -> Mapping[str, Tensor]:
    """Combines a list of dictionaries into a single dictionary by concatenating matching fields.

    Args:
        batch: A list of dictionaries
        opts: An argparse.Namespace instance.

    Returns:
        A dictionary with the same keys as batch[0].
    """
    batch_size = len(batch)

    # get the keys for first element in the list, assuming all elements have the same keys
    keys = list(batch[0].keys())

    new_batch = {k: [] for k in keys}
    for b in range(batch_size):
        for k in keys:
            new_batch[k].append(batch[b][k])

    # stack the keys
    for k in keys:
        batch_elements = new_batch.pop(k)

        if isinstance(batch_elements[0], (int, float)):
            # list of ints or floats
            batch_elements = torch.as_tensor(batch_elements)
        else:
            # stack tensors (including 0-dimensional)
            try:
                batch_elements = torch.stack(batch_elements, dim=0).contiguous()
            except Exception as e:
                logger.error("Unable to stack the tensors. Error: {}".format(e))

        new_batch[k] = batch_elements

    return new_batch
