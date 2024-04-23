#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import List, Optional, Tuple, Union

import numpy as np

from corenet.utils.math_utils import make_divisible


def image_batch_pairs(
    crop_size_w: int,
    crop_size_h: int,
    batch_size_gpu0: int,
    max_scales: Optional[float] = 5,
    check_scale_div_factor: Optional[int] = 32,
    min_crop_size_w: Optional[int] = 160,
    max_crop_size_w: Optional[int] = 320,
    min_crop_size_h: Optional[int] = 160,
    max_crop_size_h: Optional[int] = 320,
    *args,
    **kwargs
) -> List[Tuple[int, int, int]]:
    """This function creates batch and image size pairs.  For a given batch size and image size, different image sizes
        are generated and batch size is adjusted so that GPU memory can be utilized efficiently.

    Args:
        crop_size_w: Base Image width (e.g., 224)
        crop_size_h: Base Image height (e.g., 224)
        batch_size_gpu0: Batch size on GPU 0 for base image
        max_scales: Number of scales. How many image sizes that we want to generate between min and max scale factors. Default: 5
        check_scale_div_factor: Check if image scales are divisible by this factor. Default: 32
        min_crop_size_w: Min. crop size along width. Default: 160
        max_crop_size_w: Max. crop size along width. Default: 320
        min_crop_size_h: Min. crop size along height. Default: 160
        max_crop_size_h: Max. crop size along height. Default: 320

    Returns:
        a sorted list of tuples. Each index is of the form (h, w, batch_size)

    """
    width_dims = create_intervallic_integer_list(
        crop_size_w,
        min_crop_size_w,
        max_crop_size_w,
        max_scales,
        check_scale_div_factor,
    )
    height_dims = create_intervallic_integer_list(
        crop_size_h,
        min_crop_size_h,
        max_crop_size_h,
        max_scales,
        check_scale_div_factor,
    )
    img_batch_tuples = set()
    n_elements = crop_size_w * crop_size_h * batch_size_gpu0
    for crop_h, crop_y in zip(height_dims, width_dims):
        # compute the batch size for sampled image resolutions with respect to the base resolution
        _bsz = max(1, int(round(n_elements / (crop_h * crop_y), 2)))

        img_batch_tuples.add((crop_h, crop_y, _bsz))

    img_batch_tuples = list(img_batch_tuples)
    return sorted(img_batch_tuples)


def make_video_pairs(
    crop_size_h: int,
    crop_size_w: int,
    min_crop_size_h: int,
    max_crop_size_h: int,
    min_crop_size_w: int,
    max_crop_size_w: int,
    default_frames: int,
    max_scales: Optional[int] = 5,
    check_scale_div_factor: Optional[int] = 32,
    *args,
    **kwargs
) -> List[Tuple[int, int, int]]:
    """This function creates number of frames and spatial size pairs for videos.

    Args:
        crop_size_h: Base Image height (e.g., 224)
        crop_size_w: Base Image width (e.g., 224)
        min_crop_size_w: Min. crop size along width.
        max_crop_size_w: Max. crop size along width.
        min_crop_size_h: Min. crop size along height.
        max_crop_size_h: Max. crop size along height.
        default_frames: Default number of frames per clip in a video.
        max_scales: Number of scales. Default: 5
        check_scale_div_factor: Check if spatial scales are divisible by this factor.
            Default: 32.

    Returns:
        A sorted list of tuples. Each index is of the form (h, w, n_frames)
    """

    width_dims = create_intervallic_integer_list(
        crop_size_w,
        min_crop_size_w,
        max_crop_size_w,
        max_scales,
        check_scale_div_factor,
    )
    height_dims = create_intervallic_integer_list(
        crop_size_h,
        min_crop_size_h,
        max_crop_size_h,
        max_scales,
        check_scale_div_factor,
    )
    batch_pairs = set()
    n_elements = crop_size_w * crop_size_h * default_frames
    for h, w in zip(height_dims, width_dims):
        n_frames = max(1, int(round(n_elements / (h * w), 2)))
        batch_pairs.add((h, w, n_frames))
    return sorted(list(batch_pairs))


def create_intervallic_integer_list(
    base_val: Union[int, float],
    min_val: float,
    max_val: float,
    num_scales: Optional[int] = 5,
    scale_div_factor: Optional[int] = 1,
) -> List[int]:
    """This function creates a list of `n` integer values that scales `base_val` between
    `min_scale` and `max_scale`.

    Args:
        base_val: The base value to scale.
        min_val: The lower end of the value.
        max_val: The higher end of the value.
        n: Number of scaled values to generate.
        scale_div_factor: Check if scaled values are divisible by this factor.

    Returns:
        a sorted list of tuples. Each index is of the form (h, w, n_frames)
    """
    values = set(np.linspace(min_val, max_val, num_scales))
    values.add(base_val)
    values = [make_divisible(v, scale_div_factor) for v in values]
    return sorted(values)


def make_tuple_list(*val_list: List) -> List[Tuple]:
    """Make a list of values to a list of the tuples. Where ith element in each list
    is in the ith tuple of the returned list.

    For example: [[1, 2], [3, 4], [5, 6]] is converted to [(1, 3, 5), (2, 4, 6)].

    Args:
        val_list: A list of m list, where each element is a list of n values.

    Return:
        A list of size n, where each value is a tupe if m values.
    """
    return list(zip(*val_list))
