#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import torch
from packaging import version
from torch import Tensor

from corenet.constants import MIN_TORCH_VERSION
from corenet.utils import logger
from corenet.utils.ddp_utils import is_master


def construct_local_path_from_remote(remote_path: str, local_dir: str) -> str:
    """
    Obtain the local file path given a remote file path WITHOUT downloading the file. The directories
    in the returned path are created if they don't exist.

    Args:
        remote_path: The remote file path.
        local_dir: Local directory.

    Returns:
        The corresponding local file path.

    Example:
        Assume 'remote_path=s3://bucket_name/file.pkl' and 'local_dir=/tmp', then
        the file path on a local machine will be `/tmp/bucket_name/file.pkl'.

        >>> remote_path="s3://bucket_name/file.pkl"
        >>> local_dir="/tmp"
        >>> local_path = construct_local_path_from_remote(remote_path, local_dir)
        >>> print(local_path)
        /tmp/bucket_name/file.pkl
    """
    parsed_path = urlparse(remote_path)
    local_path = os.path.join(local_dir, parsed_path.netloc + parsed_path.path)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    return local_path


def unwrap_model_fn(model: torch.nn.Module) -> torch.nn.Module:
    """Helper function to unwrap the model.

    Args:
        model: An instance of torch.nn.Module.

    Returns:
        Unwrapped instance of torch.nn.Module.
    """
    unwrapped_model = model
    while True:
        if hasattr(unwrapped_model, "module"):
            # added by DataParallel and DistributedDataParallel
            unwrapped_model = unwrapped_model.module
        elif hasattr(unwrapped_model, "_fsdp_wrapped_module"):
            # FSDP wraps the model with _fsdp_wrapped_module
            unwrapped_model = unwrapped_model._fsdp_wrapped_module
        elif hasattr(unwrapped_model, "_orig_mod"):
            # torch.compile wraps the model with _orig_mod
            return unwrapped_model._orig_mod
        else:
            break
    return unwrapped_model


def check_compatibility() -> None:
    curr_torch_version = torch.__version__
    if version.parse(curr_torch_version) < version.parse(MIN_TORCH_VERSION):
        logger.error(
            "Min. pytorch version required is {}. Got: {}".format(
                MIN_TORCH_VERSION, curr_torch_version
            )
        )


def check_frozen_norm_layer(model: torch.nn.Module) -> Tuple[bool, int]:
    from corenet.modeling.layers import norm_layers_tuple

    unwrapped_model = unwrap_model_fn(model)

    count_norm = 0
    frozen_state = False
    for m in unwrapped_model.modules():
        if isinstance(m, norm_layers_tuple):
            frozen_state = m.weight.requires_grad

    return frozen_state, count_norm


def device_setup(opts: argparse.Namespace) -> argparse.Namespace:
    """Helper function for setting up the device.

    Args:
        opts: An instance of argparse.Namespace.

    Returns:
        An instance of argparse.Namespace with updated `dev.*` entries.
    """
    random_seed = getattr(opts, "common.seed", 0)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    is_master_node = is_master(opts)
    if is_master_node:
        logger.log("Random seeds are set to {}".format(random_seed))
        logger.log("Using PyTorch version {}".format(torch.__version__))

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        if is_master_node:
            logger.warning("No GPUs available. Using CPU")
        device = torch.device("cpu")
        n_gpus = 0
    else:
        if is_master_node:
            logger.log("Available GPUs: {}".format(n_gpus))
        device = torch.device("cuda")

        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn

            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            if is_master_node:
                logger.log("CUDNN is enabled")

        allow_tf32 = not getattr(opts, "common.disable_tf32", False)
        if torch.cuda.is_available():
            # TF32 is enabled by default in PyTorch < 1.12, but disabled in new versions.
            # See for details: https://github.com/pytorch/pytorch/issues/67384
            # Disable it using common.disable_tf32 flag
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    setattr(opts, "dev.device", device)
    setattr(opts, "dev.num_gpus", n_gpus)

    world_size = getattr(opts, "ddp.world_size")
    if world_size == -1:
        setattr(opts, "ddp.world_size", n_gpus)
        logger.log("Setting --ddp.world-size the same as the number of available gpus.")

    return opts


def create_directories(dir_path: str, is_master_node: bool) -> None:
    """Helper function to create directories"""
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        if is_master_node:
            logger.log("Directory created at: {}".format(dir_path))
    else:
        if is_master_node:
            logger.log("Directory exists at: {}".format(dir_path))


def move_to_device(
    opts,
    x: Any,
    device: Optional[str] = "cpu",
    non_blocking: Optional[bool] = True,
    *args,
    **kwargs
) -> Any:
    """Helper function to move data to a device"""
    if isinstance(x, Dict):
        for k, v in x.items():
            x[k] = move_to_device(
                opts=opts, x=v, device=device, non_blocking=non_blocking
            )

    elif isinstance(x, Tensor):
        # only tensors can be moved to a device
        x = x.to(device=device, non_blocking=non_blocking)
    elif isinstance(x, List):
        x = [move_to_device(opts, a, device, non_blocking) for a in x]
    return x


def is_coreml_conversion(opts) -> bool:
    if getattr(opts, "common.enable_coreml_compatible_module", False):
        return True
    return False
