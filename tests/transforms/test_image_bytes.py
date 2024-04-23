#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Tuple

import pytest
import torch

from corenet.data.transforms import image_bytes


@pytest.mark.parametrize(
    "file_encoding,quality,expected_shape",
    [
        ("fCHW", None, 432),
        ("fHWC", None, 432),
        ("TIFF", None, 572),
        ("PNG", None, 512),
        ("JPEG", 100, 693),
        ("JPEG", 50, 644),
    ],
)
def test_pil_save(file_encoding, quality, expected_shape) -> None:
    opts = argparse.Namespace()
    setattr(opts, "image_augmentation.pil_save.file_encoding", file_encoding)
    setattr(opts, "image_augmentation.pil_save.quality", quality)

    t = image_bytes.PILSave(opts)

    C, H, W = 3, 12, 12
    x = {"samples": torch.arange(C * H * W).view(C, H, W) / (C * H * W)}
    out = t(x)["samples"]
    assert out.shape == (expected_shape,)


@pytest.mark.parametrize(
    "mode,stride,window_size",
    [
        ("reverse", 8, 4),
        ("random_shuffle", 8, 4),
        ("cyclic_half_length", 8, 4),
        ("stride", 16, 8),
        ("window_shuffle", 16, 8),
    ],
)
def test_shuffle_bytes(mode, stride, window_size) -> None:
    opts = argparse.Namespace()
    setattr(opts, "image_augmentation.shuffle_bytes.mode", mode)
    setattr(opts, "image_augmentation.shuffle_bytes.stride", stride)
    setattr(opts, "image_augmentation.shuffle_bytes.window_size", window_size)

    t = image_bytes.ShuffleBytes(opts)

    size = 32
    samples = torch.randint(255, [size])
    x = {"samples": samples.clone()}
    out = t(x)["samples"]
    assert torch.any(samples != out)
    assert samples.sum() == out.sum()  # checksum


@pytest.mark.parametrize("keep_frac", [1.0, 0.75, 0.50, 0.25])
def test_mask_positions(keep_frac) -> None:
    opts = argparse.Namespace()
    setattr(opts, "image_augmentation.mask_positions.keep_frac", keep_frac)
    setattr(opts, "dev.device", "cpu")

    t = image_bytes.MaskPositions(opts)

    size = 32
    samples = torch.randint(255, [size])
    x = {"samples": samples.clone()}
    out = t(x)["samples"]
    assert out.shape == (keep_frac * size,)


def test_byte_permutation() -> None:
    opts = argparse.Namespace()

    t = image_bytes.BytePermutation(opts)

    size = 32
    samples = torch.randint(255, [size])
    x = {"samples": samples.clone()}
    out = t(x)["samples"]
    assert out.shape == samples.shape
    assert samples.unique().shape == out.unique().shape


@pytest.mark.parametrize(
    "width_range",
    [
        (0, 0),
        (-1, 1),
        (-5, 5),
    ],
)
def test_byte_permutation(width_range: Tuple[int, int]) -> None:
    opts = argparse.Namespace()
    setattr(opts, "image_augmentation.random_uniform.width_range", width_range)

    t = image_bytes.RandomUniformNoise(opts)

    size = 32
    samples = torch.randint(255, [size])
    x = {"samples": samples.clone()}
    out = t(x)["samples"]
    assert samples.shape == out.shape

    if width_range == (0, 0):
        assert torch.all(samples == out)
    else:
        assert torch.any(samples != out)
