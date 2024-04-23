#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import operator
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytest
import torch

from corenet.data.transforms.base_transforms import BaseTransformation
from corenet.data.transforms.video import (
    CropByBoundingBox,
    SaveInputs,
    ShuffleAudios,
    _resize_fn,
)
from corenet.data.video_reader.pyav_reader import PyAVReader
from tests.configs import get_config


def test_resize_fn() -> None:
    bs = 2
    sz = (10, 8)
    c = 3
    t = 4
    new_sz = (6, 4)
    data = {
        "samples": {
            "video": torch.randn(bs, t, c, *sz),
            "mask": torch.rand(bs, t, *sz),
        }
    }
    new_data = _resize_fn(data, size=new_sz)
    video = data["samples"]["video"]
    mask = data["samples"]["mask"]
    assert video.shape == (bs, t, c, *new_sz)
    assert mask.shape == (bs, t, *new_sz)


def _non_zero_values_bounds(x: torch.Tensor) -> Tuple[float, float]:
    """
    Returns the min and max of the non-zero values in a tensor.
    """
    x = x.ravel()
    x = x[x != 0]
    return x.min().item(), x.max().item()


@pytest.mark.parametrize("is_training", [True, False])
@pytest.mark.parametrize("multiplier_range", [(1.3, 1.5), None])
def test_crop_by_bounding_box(
    is_training: bool, multiplier_range: Optional[Tuple[float, float]]
) -> None:
    opts = get_config()
    image_size = (10, 10)
    setattr(opts, "video_augmentation.crop_by_bounding_box.image_size", image_size)
    setattr(opts, "video_augmentation.crop_by_bounding_box.multiplier", 1.4)
    # Set interpolation to "nearest", so that we can compare some of the pixel values.
    setattr(opts, "video_augmentation.crop_by_bounding_box.interpolation", "nearest")
    setattr(
        opts,
        "video_augmentation.crop_by_bounding_box.multiplier_range",
        multiplier_range,
    )
    transform = CropByBoundingBox(opts, is_training=is_training)

    N, T, C, H, W = 2, 5, 3, 32, 32

    video = torch.rand(N, T, C, H, W)
    box_coordinates = torch.concat(
        [
            torch.rand(N, T, 2) * 0.5,
            torch.rand(N, T, 2) * 0.5 + 0.5,
        ],
        dim=2,
    )

    MAX_VAL = 255
    y, x = torch.meshgrid(
        torch.linspace(1, MAX_VAL, H), torch.linspace(1, MAX_VAL, W), indexing="ij"
    )
    video[0, 0, 0, :, :] = y
    video[0, 0, 1, :, :] = x

    # The bounding box is a small horizontal strip at the bottom of the image.
    # Later, we have some assertions for the original x and y values of the cropped
    # pixels. We assert that the expanded box exceeds horizontal boundaries of the
    # image, but not the vertical ones. Hence, the bounding box shouldn't be too close
    # to the bottom edge, so that the expanded box remains within the bottom boundary.
    box_coordinates[0, 0, :] = torch.tensor([0, 0.7, 1, 0.8])

    visible_frame_mask = torch.ones(N, T, dtype=torch.bool)
    # There are two kinds of invisible/invalid bounding boxes:
    # 1) When the bounding box has x0 == y0 == x1 == y1 == -1 coordinates.
    # 2) When the bounding box has valid coordinates but the area is too small (< 5 pixels).
    invisible_frame_1_index = (0, 3)  # An arbitrary [0<clip<N, 0<frame<T] index pair.
    box_coordinates[invisible_frame_1_index][:] = -1
    visible_frame_mask[invisible_frame_1_index] = 0
    invisible_frame_2_index = (1, 2)  # An arbitrary [0<clip<N, 0<frame<T] index pair.
    box_coordinates[invisible_frame_2_index][:] = torch.tensor([0, 0, 1e-5, 1e-5])
    visible_frame_mask[invisible_frame_2_index] = 0

    data = {
        "samples": {
            "video": video,
        },
        "targets": {
            "traces": {
                "a_uuid": {
                    "box_coordinates": box_coordinates,
                }
            }
        },
    }

    result = transform(data)
    assert isinstance(result, dict), f"{type(result)} != dict"
    result_video = result["samples"]["video"]
    assert result_video.shape == (N, T, C, *image_size)
    assert torch.all(
        result_video[~visible_frame_mask] == 0.0
    ), "Frames with invisible bounding boxes should be blacked out."

    result_box_coordinates = result["targets"]["traces"]["a_uuid"]["box_coordinates"]
    assert result_box_coordinates.shape == (N, T, 4)
    assert torch.all(result_box_coordinates[~visible_frame_mask] == -1)
    assert torch.all(0 < result_box_coordinates[:, :, :2][visible_frame_mask])
    assert torch.all(result_box_coordinates[:, :, 2:][visible_frame_mask] < 1)
    max_coord_threshold = (
        (1 - 1 / multiplier_range[1]) / 2
        if is_training and multiplier_range is not None
        else (transform.multiplier - 1) / 2
    )
    assert torch.all(
        result_box_coordinates[:, :, :2][visible_frame_mask] <= max_coord_threshold
    )
    assert torch.all(
        1 - max_coord_threshold < result_box_coordinates[:, :, 2:][visible_frame_mask]
    )

    # Since the bounding box was aligned with the right and left edges of the image,
    # we should observe 0 values in the right and left edges of the cropped image,
    # as a result of expansion.
    assert torch.all(result_video[0, 0, 0, :, 0] == 0)
    assert torch.all(result_video[0, 0, 0, :, -1] == 0)
    # but not on the top and bottom edges
    assert torch.all(result_video[0, 0, 0, 0, 2:-2] > 0)
    assert torch.all(result_video[0, 0, 0, -1, 2:-2] > 0)

    # Check the cropped pixels' original coordinates. In video[0,0,0] and video[0,0,1],
    # the x and y values were stored with uniform distributions of [1, W] and [1, H].
    min_y, max_y = _non_zero_values_bounds(result_video[0, 0, 0])
    min_x, max_x = _non_zero_values_bounds(result_video[0, 0, 1])
    # The values of 0.65, 0.75, and 0.85 are derived from y0=0.7 and y1=0.8 that are set
    # in box_coordinates. Maximum multiplier_range is 1.5, hence the expanded pixel
    # values will have y-coordinates between 0.65*MAX_VAL and 0.85*MAX_VAL.
    assert (
        0.65 * MAX_VAL <= min_y <= 0.75 * MAX_VAL <= max_y - 1 <= 0.85 * MAX_VAL
    ), "Y values are not within the expanded [0.7, 0.8] bbox."
    assert (
        0 * MAX_VAL <= min_x <= 0.1 * MAX_VAL <= 0.9 * MAX_VAL <= max_x <= MAX_VAL
    ), "X values are not within the expanded [0, 1] bbox."


@pytest.mark.parametrize("numel", [5, 6])
@pytest.mark.parametrize("is_training", [True, False])
def test_shuffle_audios_single_cycle_permutation(numel: int, is_training: bool) -> None:
    device = torch.device("cpu")
    prev_perm = ShuffleAudios._single_cycle_permutation(
        numel, device=device, is_training=is_training
    )
    identity = torch.arange(numel, device=device)
    is_random = False
    for _ in range(20):
        perm = ShuffleAudios._single_cycle_permutation(
            numel, device=device, is_training=is_training
        )
        if torch.any(perm != prev_perm):
            is_random = True
            prev_perm = perm

        assert torch.all(
            perm != identity
        ), f"Single cycle permutation should not have identity mapping: {perm}."

        sorted_perm, _ = perm.sort()
        assert torch.all(
            sorted_perm == identity
        ), f"Result is not a permutation: {perm}."

    assert is_random == is_training, "Outcomes should be random iff is_training."


@pytest.mark.parametrize(
    "N, mode, shuffle_ratio, generate_frame_level_targets, debug_mode",
    [
        (1000, "train", 0.2, True, False),
        (1000, "val", 0.3, False, False),
        (1000, "test", 0.7, True, True),
        (1, "train", 0.2, False, False),
        (1, "val", 0.2, False, False),
    ],
)
def test_shuffle_audios(
    N: int,
    mode: str,
    shuffle_ratio: float,
    generate_frame_level_targets: bool,
    debug_mode: bool,
) -> None:
    opts = get_config()
    setattr(
        opts,
        f"video_augmentation.shuffle_audios.shuffle_ratio_{mode}",
        shuffle_ratio,
    )
    setattr(
        opts,
        "video_augmentation.shuffle_audios.generate_frame_level_targets",
        generate_frame_level_targets,
    )
    setattr(
        opts,
        "video_augmentation.shuffle_audios.debug_mode",
        debug_mode,
    )

    C_v, C_a = 3, 2
    H, W = 8, 8
    num_video_frames = 3
    num_audio_frames = 5

    video = torch.rand(N, num_video_frames, C_v, H, W)
    # Generating unique audio elements (using torch.arange) so that we can compare them to check if they are shuffled.
    input_audio = torch.empty(N, num_audio_frames, C_a, dtype=torch.float)
    torch.arange(input_audio.numel(), dtype=torch.float, out=input_audio)

    data = {
        "samples": {
            "video": video,
            "audio": input_audio.clone(),
            "metadata": {},
        },
        "targets": {},
    }

    result = ShuffleAudios(
        opts=opts,
        is_training=mode == "train",
        is_evaluation=mode == "test",
        item_index=0,
    )(data)

    labels = data["targets"]["is_shuffled"]
    assert (
        labels.shape == (N, num_video_frames) if generate_frame_level_targets else (N,)
    )
    if generate_frame_level_targets:
        assert torch.all(
            labels == labels[:, :1].repeat(1, num_video_frames)
        ), "Labels should be identical among frames of the same clip."

    result_is_shuffled = (
        (~torch.isclose(input_audio, result["samples"]["audio"]))
        .float()  # shape: N x num_audio_frames x C_a
        .mean(axis=1)  # shape: N x C_a
        .mean(axis=1)  # shape: N
    )
    actual_participation_ratio = result_is_shuffled.float().mean()
    if N > 1:
        assert actual_participation_ratio == pytest.approx(shuffle_ratio, abs=0.05)
    else:
        assert actual_participation_ratio == pytest.approx(0.0, abs=0.05)

    assert torch.allclose(
        result_is_shuffled,
        (labels[:, 0] if generate_frame_level_targets else labels).float(),
    ), "Generated labels should match shuffled audios."

    if debug_mode:
        assert data["samples"]["metadata"]["shuffled_audio_permutation"].shape == (N,)
    else:
        assert "shuffled_audio_permutation" not in data["samples"]["metadata"]


@pytest.mark.parametrize(
    "t,expected",
    [
        (0, "00:00:0,000"),
        (0.5, "00:00:0,500"),
        (10.5, "00:00:10,500"),
        (70.5, "00:01:10,500"),
        (7270.5, "02:01:10,500"),
    ],
)
def test_save_inputs_srt_format_timestamp(t: float, expected: str) -> None:
    assert SaveInputs._srt_format_timestamp(t) == expected


@pytest.mark.parametrize(
    "params",
    [
        {
            "opts": {},
            "init_kwargs": {},
            "expected_output_count": 1,
            "video_only": False,
        },
        {
            "opts": {},
            "init_kwargs": {
                "get_frame_captions": (
                    lambda data: ["_"]
                    * operator.mul(*data["samples"]["video"].shape[:2])
                )
            },
            "expected_output_count": 1,
            "video_only": False,
        },
        {
            "opts": {"video_augmentation.save_inputs.symlink_to_original": True},
            "init_kwargs": {},
            "expected_output_count": 2,
            "video_only": True,
        },
    ],
)
def test_save_inputs(
    params: Tuple[Dict, Dict],
    tmp_path: Path,
) -> None:
    opts = get_config()
    setattr(opts, "video_augmentation.save_inputs.enable", True)
    setattr(opts, "video_augmentation.save_inputs.save_dir", str(tmp_path))
    for key, value in params["opts"].items():
        setattr(opts, key, value)

    data = {
        "samples": PyAVReader(opts).dummy_audio_video_clips(
            clips_per_video=2,
            num_frames_to_sample=3,
            height=24,
            width=24,
        )
    }
    if params["video_only"]:
        del data["samples"]["audio"]
    SaveInputs(opts, **params["init_kwargs"])(data)
    output_video_paths = list(tmp_path.glob("*/*.mp4"))
    assert len(output_video_paths) == params["expected_output_count"], (
        f"Expected {params['expected_output_count']} videos, but got:"
        f" {output_video_paths}."
    )
    for output_video_path in output_video_paths:
        if output_video_path.is_symlink():
            continue
        assert os.stat(output_video_path).st_size > 1000, (
            f"The generated file ({output_video_path}) is too small"
            f" ({os.stat(output_video_path).st_size})."
        )
