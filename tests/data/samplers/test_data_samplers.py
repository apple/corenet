#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Optional

import numpy as np
import pytest

from corenet.data.sampler import build_sampler
from tests.configs import get_config

N_DATA_SAMPLES = 1000
DEFAULT_BATCH_SIZE = 128


def set_common_defaults(
    opts: argparse.Namespace, batch_size: Optional[int] = DEFAULT_BATCH_SIZE
) -> None:
    """Set common default values for all samplers."""
    setattr(opts, "dataset.train_batch_size0", batch_size)


@pytest.mark.parametrize(
    "num_repeats, trunc_ra_sampler, crop_size_h, crop_size_w",
    [
        (1, True, 128, 320),
        (1, False, 128, 320),
        (4, True, 160, 384),
        (4, False, 160, 384),
    ],
)
def test_ssc_fbs_sampler(
    num_repeats: int, trunc_ra_sampler: bool, crop_size_h: int, crop_size_w: int
) -> None:
    """Test for single-scale fixed batch size sampler with and without repeated augmentation."""

    opts = get_config(config_file="tests/data/samplers/test_batch_sampler_config.yaml")

    set_common_defaults(opts)

    # set repeated augmentation related hyper-parameters
    setattr(opts, "sampler.num_repeats", num_repeats)
    setattr(opts, "sampler.truncated_repeat_aug_sampler", trunc_ra_sampler)

    # we over-ride the spatial sizes for testing different cases
    setattr(opts, "sampler.bs.crop_size_height", crop_size_h)
    setattr(opts, "sampler.bs.crop_size_width", crop_size_w)

    sampler = build_sampler(opts, n_data_samples=N_DATA_SAMPLES, is_training=True)
    assert hasattr(sampler, "crop_size_h")
    assert hasattr(sampler, "crop_size_w")
    assert sampler.crop_size_h == crop_size_h
    assert sampler.crop_size_w == crop_size_w

    # repeated sample test
    np.testing.assert_equal(
        len(sampler), N_DATA_SAMPLES * (1 if trunc_ra_sampler else num_repeats)
    )


@pytest.mark.parametrize(
    "num_repeats, trunc_ra_sampler, min_res_h, max_res_h, min_res_w, max_res_w,max_scales",
    [
        (1, True, 128, 320, 128, 320, 1),
        (1, False, 128, 320, 128, 320, 5),
        (4, True, 160, 384, 160, 384, 10),
        (4, False, 160, 384, 160, 384, 50),
    ],
)
def test_msc_fbs_batch_sampler(
    num_repeats: int,
    trunc_ra_sampler: bool,
    min_res_h: int,
    max_res_h: int,
    min_res_w: int,
    max_res_w: int,
    max_scales: int,
) -> None:
    """Test for multi-scale fixed batch size sampler with and without repeated augmentation."""
    opts = get_config(
        config_file="tests/data/samplers/test_multi_scale_sampler_config.yaml"
    )

    set_common_defaults(opts)

    setattr(opts, "sampler.msc.max_n_scales", max_scales)
    setattr(opts, "sampler.msc.min_crop_size_width", min_res_w)
    setattr(opts, "sampler.msc.max_crop_size_width", max_res_w)
    setattr(opts, "sampler.msc.min_crop_size_height", min_res_h)
    setattr(opts, "sampler.msc.max_crop_size_height", max_res_h)

    # set repeated augmentation related hyper-parameters
    setattr(opts, "sampler.num_repeats", num_repeats)
    setattr(opts, "sampler.truncated_repeat_aug_sampler", trunc_ra_sampler)

    sampler = build_sampler(opts, n_data_samples=1000, is_training=True)

    assert hasattr(sampler, "img_batch_tuples")

    scales_vbs = sampler.img_batch_tuples

    # check if number of sampled scales are <= max_scales + 1
    # + 1 here because if the base image size is not present in the sampled scales, then we add it
    assert len(scales_vbs) <= (max_scales + 1)

    for h, w, b_sz in scales_vbs:
        # Resolution needs to be within the specified intervals
        assert min_res_h <= h <= max_res_h
        assert min_res_w <= w <= max_res_w
        assert (
            b_sz == DEFAULT_BATCH_SIZE
        ), "We expect batch size to be the same for all scales in multi-scale sampler"

    # repeated sample test
    np.testing.assert_equal(
        len(sampler), N_DATA_SAMPLES * (1 if trunc_ra_sampler else num_repeats)
    )


@pytest.mark.parametrize(
    "num_repeats, trunc_ra_sampler, min_res_h, max_res_h, min_res_w, max_res_w,max_scales, batch_size",
    [
        (1, True, 128, 320, 128, 320, 1, 1),
        (1, False, 128, 320, 128, 320, 5, 2),
        (4, True, 160, 384, 160, 384, 10, DEFAULT_BATCH_SIZE),
        (4, False, 160, 384, 160, 384, 50, DEFAULT_BATCH_SIZE),
    ],
)
def test_msc_vbs_sampler(
    num_repeats: int,
    trunc_ra_sampler: bool,
    min_res_h: int,
    max_res_h: int,
    min_res_w: int,
    max_res_w: int,
    max_scales: int,
    batch_size: int,
) -> None:
    """Test for multi-scale variably-batch size sampler with and without repeated augmentation."""
    opts = get_config(
        config_file="tests/data/samplers/test_variable_batch_sampler_config.yaml"
    )

    set_common_defaults(opts, batch_size)

    setattr(opts, "sampler.vbs.max_n_scales", max_scales)
    setattr(opts, "sampler.vbs.min_crop_size_width", min_res_w)
    setattr(opts, "sampler.vbs.max_crop_size_width", max_res_w)
    setattr(opts, "sampler.vbs.min_crop_size_height", min_res_h)
    setattr(opts, "sampler.vbs.max_crop_size_height", max_res_h)
    setattr(opts, "sampler.num_repeats", num_repeats)
    setattr(opts, "sampler.truncated_repeat_aug_sampler", trunc_ra_sampler)

    sampler = build_sampler(opts, n_data_samples=N_DATA_SAMPLES, is_training=True)

    assert hasattr(sampler, "img_batch_tuples")

    scales_vbs = sampler.img_batch_tuples

    # check if number of sampled scales are <= max_scales + 1
    # + 1 here because if the base image size is not present in the sampled scales, then we add it
    assert len(scales_vbs) <= (max_scales + 1)

    for h, w, b_sz in scales_vbs:
        # Resolution needs to be within the specified intervals
        assert min_res_h <= h <= max_res_h
        assert min_res_w <= w <= max_res_w

        area_ratio = (h * w * 1.0) / (224 * 224)

        # For variable sampler, we expect larger batch sizes for smaller resolution as
        # compared to base batch size and vice-versa
        if area_ratio < 1.0:
            # When batch size=1 (or small), then MSc-VBS sampler behaves like MSc-FBS, i.e., any resolution slightly
            # smaller than base resolution have the same batch size of 1.
            assert b_sz >= batch_size, (
                "For spatial resolution smaller than the base resolution, "
                "we expect batch size >= base batch size"
            )
        elif area_ratio > 1.0:
            # When batch size=1 (or small), then MSc-VBS sampler behaves like MSc-FBS, i.e., any resolution higher
            # than base resolution have the same batch size of 1.
            assert b_sz <= batch_size, (
                "For spatial resolution larger than base resolution, "
                "we expect batch size to <= base batch size"
            )
        else:
            assert b_sz == batch_size, (
                "For spatial resolution equal to the base resolution, "
                "we expect batch size == base batch size"
            )

    # repeated sample test
    np.testing.assert_equal(
        len(sampler), N_DATA_SAMPLES * (1 if trunc_ra_sampler else num_repeats)
    )


@pytest.mark.parametrize(
    "num_repeats, trunc_ra_sampler, min_res_h, max_res_h, min_res_w, max_res_w,max_scales, fps_scales, batch_size, is_training",
    [
        (1, True, 128, 320, 128, 320, 1, 1, 1, True),
        (1, False, 128, 320, 128, 320, 5, 5, 2, False),
        (4, True, 160, 384, 160, 384, 10, 1, DEFAULT_BATCH_SIZE, True),
        (4, False, 160, 384, 160, 384, 50, 5, DEFAULT_BATCH_SIZE, False),
    ],
)
def test_video_clip_sampler(
    num_repeats: int,
    trunc_ra_sampler: bool,
    min_res_h: int,
    max_res_h: int,
    min_res_w: int,
    max_res_w: int,
    max_scales: int,
    fps_scales: int,
    batch_size: int,
    is_training: bool,
) -> None:
    """Test for multi-scale variably-batch size sampler with and without repeated augmentation."""
    opts = get_config(
        config_file="tests/data/samplers/test_video_clip_batch_sampler_config.yaml"
    )

    set_common_defaults(opts, batch_size)

    setattr(opts, "sampler.vbs.max_n_scales", max_scales)
    setattr(opts, "sampler.vbs.min_crop_size_width", min_res_w)
    setattr(opts, "sampler.vbs.max_crop_size_width", max_res_w)
    setattr(opts, "sampler.vbs.min_crop_size_height", min_res_h)
    setattr(opts, "sampler.vbs.max_crop_size_height", max_res_h)
    setattr(opts, "sampler.num_repeats", num_repeats)
    setattr(opts, "sampler.vcbs.video_fps_num_scales", fps_scales)
    setattr(opts, "sampler.truncated_repeat_aug_sampler", trunc_ra_sampler)

    metadata = {
        0: {
            "video_path": "tests/data/dummy_video.mov",
            "video_fps": 29,
            "total_video_frames": 114,
            "video_duration": 3.9,
        }
    }
    sampler = build_sampler(
        opts,
        n_data_samples=N_DATA_SAMPLES,
        is_training=is_training,
        get_item_metadata=metadata.__getitem__,
    )

    assert hasattr(sampler, "img_batch_tuples")
    assert hasattr(sampler, "frame_rate_scales")

    if is_training:
        # Check if number of sampled scales are <= max_scales + 1,
        # + 1 here because if the base image size is not present in the sampled scales,
        # then we add it.
        assert len(sampler.img_batch_tuples) <= max_scales + 1
        assert len(sampler.frame_rate_scales) <= fps_scales + 1
        expected_len = N_DATA_SAMPLES * (1 if trunc_ra_sampler else num_repeats)
    else:
        assert len(sampler.img_batch_tuples) == 1
        assert len(sampler.frame_rate_scales) == 1
        expected_len = N_DATA_SAMPLES
    # Repeated sample test.
    np.testing.assert_equal(len(sampler), expected_len)
