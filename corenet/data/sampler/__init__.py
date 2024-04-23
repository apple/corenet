#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Callable, Dict, Mapping, Optional, Union

from torch.utils.data.sampler import Sampler

from corenet.data.sampler.base_sampler import BaseSampler, BaseSamplerDDP
from corenet.utils.registry import Registry

SAMPLER_REGISTRY = Registry(
    registry_name="data_samplers",
    base_class=Sampler,
    # lazily import the samplers
    lazy_load_dirs=["corenet/data/sampler"],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def build_sampler(
    opts: argparse.Namespace,
    n_data_samples: Union[int, Mapping[str, int]],
    is_training: bool = False,
    get_item_metadata: Optional[Callable[[int], Dict]] = None,
    *args,
    **kwargs
) -> Sampler:
    """Helper function to build data sampler from command-line arguments

    Args:
        opts: Command-line arguments
        n_data_samples: Number of data samples. It can be an integer specifying number of data samples for a given task
            or a mapping of task name and data samples per task in case of a chain sampler.
        get_item_metadata: A callable that provides sample metadata, given sample index.
        is_training: Training mode or not. Defaults to False.

    Returns:
        Data sampler over which we can iterate.
    """
    sampler_name = getattr(opts, "sampler.name")
    is_distributed = getattr(opts, "ddp.use_distributed")

    if (
        is_distributed
        and sampler_name.split("_")[-1] != "ddp"
        and sampler_name != "chain_sampler"
    ):
        # In case of a DDP environment, add `_ddp` to sampler name if not present
        # with an exception to chain_sampler (which is nothing but a loop over existing samplers)
        sampler_name = sampler_name + "_ddp"

    sampler = SAMPLER_REGISTRY[sampler_name](
        opts,
        n_data_samples=n_data_samples,
        is_training=is_training,
        get_item_metadata=get_item_metadata,
    )
    return sampler


def add_sampler_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add sampler arguments to parser from SAMPLER_REGISTRY,
    BaseSampler, and BaseSamplerDDP"""
    parser = SAMPLER_REGISTRY.all_arguments(parser)
    parser = BaseSampler.add_arguments(parser)
    parser = BaseSamplerDDP.add_arguments(parser)
    return parser
