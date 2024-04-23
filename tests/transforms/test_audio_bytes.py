#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

import pytest
import torch

from corenet.data.transforms import audio_bytes


@pytest.mark.parametrize(
    "format,encoding_dtype,num_samples,expected_length",
    [
        ("wav", "float32", 4, 74),
        ("wav", "float32", 8, 90),
        ("wav", "int32", 8, 112),
        ("wav", "int16", 8, 60),
        ("wav", "uint8", 8, 52),
        ("mp3", None, 8, 216),
    ],
)
def test_audio_save(format, encoding_dtype, num_samples, expected_length) -> None:
    opts = argparse.Namespace()
    setattr(opts, "audio_augmentation.torchaudio_save.encoding_dtype", encoding_dtype)
    setattr(opts, "audio_augmentation.torchaudio_save.format", format)
    setattr(opts, "audio_augmentation.torchaudio_save.backend", "sox")
    t = audio_bytes.TorchaudioSave(opts)

    x = {
        "samples": {"audio": torch.randn([2, num_samples])},
        "metadata": {"audio_fps": 16},
    }

    outputs = t(x)["samples"]["audio"]
    assert torch.all(0 <= outputs)
    assert torch.all(outputs <= 255)
    assert outputs.shape == (expected_length,)
