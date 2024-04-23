#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch import distributed as dist
from torch.distributed.nn import all_gather as all_gather_with_backward

from corenet.constants import (
    DEFAULT_IMAGE_CHANNELS,
    DEFAULT_IMAGE_HEIGHT,
    DEFAULT_IMAGE_WIDTH,
    DEFAULT_VIDEO_FRAMES,
)


def image_size_from_opts(opts) -> Tuple[int, int]:
    try:
        sampler_name = getattr(opts, "sampler.name", "variable_batch_sampler").lower()
        if sampler_name.find("var") > -1:
            im_w = getattr(opts, "sampler.vbs.crop_size_width", DEFAULT_IMAGE_WIDTH)
            im_h = getattr(opts, "sampler.vbs.crop_size_height", DEFAULT_IMAGE_HEIGHT)
        elif sampler_name.find("multi") > -1:
            im_w = getattr(opts, "sampler.msc.crop_size_width", DEFAULT_IMAGE_WIDTH)
            im_h = getattr(opts, "sampler.msc.crop_size_height", DEFAULT_IMAGE_HEIGHT)
        else:
            im_w = getattr(opts, "sampler.bs.crop_size_width", DEFAULT_IMAGE_WIDTH)
            im_h = getattr(opts, "sampler.bs.crop_size_height", DEFAULT_IMAGE_HEIGHT)
    except Exception as e:
        im_h = DEFAULT_IMAGE_HEIGHT
        im_w = DEFAULT_IMAGE_WIDTH
    return im_h, im_w


def video_size_from_opts(opts) -> Tuple[int, int, int]:
    try:
        sampler_name = getattr(opts, "sampler.name", "video_batch_sampler").lower()
        if sampler_name.find("var") > -1:
            im_w = getattr(opts, "sampler.vbs.crop_size_width", DEFAULT_IMAGE_WIDTH)
            im_h = getattr(opts, "sampler.vbs.crop_size_height", DEFAULT_IMAGE_HEIGHT)
            n_frames = getattr(
                opts, "sampler.vbs.num_frames_per_clip", DEFAULT_IMAGE_HEIGHT
            )
        else:
            im_w = getattr(opts, "sampler.bs.crop_size_width", DEFAULT_IMAGE_WIDTH)
            im_h = getattr(opts, "sampler.bs.crop_size_height", DEFAULT_IMAGE_HEIGHT)
            n_frames = getattr(
                opts, "sampler.bs.num_frames_per_clip", DEFAULT_IMAGE_HEIGHT
            )
    except Exception as e:
        im_h = DEFAULT_IMAGE_HEIGHT
        im_w = DEFAULT_IMAGE_WIDTH
        n_frames = DEFAULT_VIDEO_FRAMES
    return im_h, im_w, n_frames


def create_rand_tensor(
    opts, device: Optional[str] = "cpu", batch_size: Optional[int] = 1
) -> Tensor:
    sampler = getattr(opts, "sampler.name", "batch_sampler")
    if sampler.lower().find("video") > -1:
        video_stack = getattr(opts, "video_reader.frame_stack_format", "channel_first")
        im_h, im_w, n_frames = video_size_from_opts(opts=opts)
        if video_stack == "channel_first":
            inp_tensor = torch.randint(
                low=0,
                high=255,
                size=(batch_size, DEFAULT_IMAGE_CHANNELS, n_frames, im_h, im_w),
                device=device,
            )
        else:
            inp_tensor = torch.randint(
                low=0,
                high=255,
                size=(batch_size, n_frames, DEFAULT_IMAGE_CHANNELS, im_h, im_w),
                device=device,
            )
    else:
        im_h, im_w = image_size_from_opts(opts=opts)
        inp_tensor = torch.randint(
            low=0,
            high=255,
            size=(batch_size, DEFAULT_IMAGE_CHANNELS, im_h, im_w),
            device=device,
        )
    inp_tensor = inp_tensor.float().div(255.0)
    return inp_tensor


def reduce_tensor(inp_tensor: torch.Tensor) -> torch.Tensor:
    size = dist.get_world_size() if dist.is_initialized() else 1
    inp_tensor_clone = inp_tensor.clone().detach()
    # dist_barrier()
    dist.all_reduce(inp_tensor_clone, op=dist.ReduceOp.SUM)
    inp_tensor_clone /= size
    return inp_tensor_clone


def reduce_tensor_sum(inp_tensor: torch.Tensor) -> torch.Tensor:
    inp_tensor_clone = inp_tensor.clone().detach()
    # dist_barrier()
    dist.all_reduce(inp_tensor_clone, op=dist.ReduceOp.SUM)
    return inp_tensor_clone


def all_gather_list(data: Union[List, Tensor, Dict[str, Tensor]]):
    world_size = dist.get_world_size()
    data_list = [None] * world_size
    # dist_barrier()
    dist.all_gather_object(data_list, data)
    return data_list


def gather_all_features(features: Tensor, dim=0):
    return torch.cat(all_gather_with_backward(features), dim=dim)
    # world_size = dist.get_world_size()
    # gathered_data = [torch.zeros_like(features)] * world_size
    # dist.all_gather(gathered_data, features)
    # gathered_data = torch.cat(gathered_data, dim=dim)
    # return gathered_data


def tensor_to_python_float(
    inp_tensor: Union[int, float, torch.Tensor],
    is_distributed: bool,
    reduce_op: str = "mean",
) -> Union[int, float, np.ndarray]:
    """
    Given a number or a Tensor (potentially in distributed setting) returns the float value.
    If is_distributed is true, the Tensor must be aggregated first.

    Args:
        inp_tensor: the input tensor
        is_distributed: indicates whether we are in distributed mode
        reduce_op: reduce operation for aggregation
            If equals to mean, will reduce using mean, otherwise sum operation
    """
    if is_distributed and isinstance(inp_tensor, torch.Tensor):
        if reduce_op == "mean":
            inp_tensor = reduce_tensor(inp_tensor=inp_tensor)
        else:
            inp_tensor = reduce_tensor_sum(inp_tensor=inp_tensor)

    if isinstance(inp_tensor, torch.Tensor) and inp_tensor.numel() > 1:
        # For IOU, we get a C-dimensional tensor (C - number of classes)
        # so, we convert here to a numpy array
        return inp_tensor.cpu().numpy()
    elif hasattr(inp_tensor, "item"):
        return inp_tensor.item()
    elif isinstance(inp_tensor, (int, float)):
        return inp_tensor * 1.0
    else:
        raise NotImplementedError(
            "The data type is not supported yet in tensor_to_python_float function"
        )


def to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    # [0, 1] --> [0, 255]
    img_tensor = torch.mul(img_tensor, 255.0)
    # BCHW --> BHWC
    img_tensor = img_tensor.permute(0, 2, 3, 1)

    img_np = img_tensor.byte().cpu().numpy()
    return img_np
