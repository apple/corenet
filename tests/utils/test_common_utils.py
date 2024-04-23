#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os

import torch
import torch.distributed as dist

from corenet.utils.common_utils import unwrap_model_fn


def check_models(
    original_unwrapped_model: torch.nn.Module, model_after_unwrapping: torch.nn.Module
) -> None:
    """Helper function to test original and unwrapped models are the same."""
    for layer_id in range(len(original_unwrapped_model)):
        # for unwrapped models, we should be able to index them
        assert repr(model_after_unwrapping[layer_id]) == repr(
            original_unwrapped_model[layer_id]
        )


def test_unwrap_model_fn():
    """Test for unwrap_model_fn"""

    dummy_model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.Linear(20, 40),
    )

    # test DataParallel wrapping
    wrapped_model_dp = torch.nn.DataParallel(dummy_model)
    unwrapped_model_dp = unwrap_model_fn(wrapped_model_dp)
    check_models(dummy_model, unwrapped_model_dp)

    # Initialize the distributed environment
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1234"
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

    # test DDP wrapping
    wrapped_model_ddp = torch.nn.parallel.DistributedDataParallel(dummy_model)
    unwrapped_model_ddp = unwrap_model_fn(wrapped_model_ddp)

    check_models(dummy_model, unwrapped_model_ddp)
    # clean up DDP environment
    dist.destroy_process_group()
