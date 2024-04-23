#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import copy
import os
import shutil
import tempfile

import pytest
import scipy.io.wavfile as wav
import torch

from corenet.data.transforms.audio import (
    AudioResample,
    Gain,
    GaussianAudioNoise,
    LambdaAudio,
    MFCCs,
    Noise,
    Roll,
    SetFixedLength,
    StandardizeChannels,
)
from tests.configs import get_config


def test_mfccs():
    opts = get_config()

    transform = MFCCs(opts)
    num_frames = transform.num_frames
    assert num_frames == 8
    num_mfccs = transform.num_mfccs
    assert num_mfccs == 20
    N, C, fps = 10, 2, 41000
    audio_length = 1  # seconds

    audio = torch.rand(N, audio_length * fps, C)
    data = {
        "samples": {
            "audio": audio,
            "metadata": {
                "audio_fps": fps,
            },
        }
    }
    result = transform(data)
    expected_frame_length = 11  # depends on num_frames, fps, audio_length
    assert result["samples"]["audio"].shape == (
        N,
        C,
        num_mfccs,
        num_frames,
        expected_frame_length,
    )


def test_lambda_audio() -> None:
    opts = get_config()

    input = {
        "samples": {
            "audio": torch.rand(
                2, 5, 1, 7
            ),  # arbitrary number of dimensions of arbitrary lengths
            "xyz": torch.rand(3, 9),  # arbitrary key, value
        },
        "targets": {
            "xyz2": torch.rand(4, 8),  # arbitrary key, value
        },
    }

    func = lambda x: (x + 1) * 2
    output = LambdaAudio(opts, func)(copy.deepcopy(input))

    nested_key_names = lambda dict_: {key: value.keys() for key, value in dict_.items()}
    assert nested_key_names(input) == nested_key_names(output)

    assert torch.allclose(output["samples"]["audio"], func(input["samples"]["audio"]))
    assert torch.allclose(output["samples"]["xyz"], input["samples"]["xyz"])
    assert torch.allclose(output["targets"]["xyz2"], input["targets"]["xyz2"])


@pytest.mark.parametrize("length", [2, 3, 4])
def test_set_fixed_length(length: int) -> None:
    opts = get_config()
    setattr(opts, "audio_augmentation.set_fixed_length.length", length)
    transform = SetFixedLength(opts)

    C, N = 2, 8

    x = {"samples": {"audio": torch.rand(C, N)}}

    output = transform(x)
    assert output["samples"]["audio"].shape == (C, length)


@pytest.mark.parametrize("window", [0.5, 0.7])
def test_roll(window: float) -> None:
    torch.manual_seed(0)
    opts = get_config()
    setattr(opts, "audio_augmentation.roll.window", window)
    transform = Roll(opts)

    C, N = 2, 20

    audio = torch.arange(C * N).view(C, N)
    x = {"samples": {"audio": audio.clone()}}

    output = transform(x)["samples"]["audio"]
    # The first parameter won't change (unless we are unlucky and the "roll"
    # parameter is 0). That's why we fix the seed above.
    assert output[0, 0] != audio[0, 0]
    # Make sure the first channel is just a shifted version of the original audio.
    assert torch.all((output[0] - output[0, 0]) % output.shape[1] == audio[0])

    # Make sure the second channel has the same shift as the first.
    assert torch.all((output[0] - output[0, 0]) == (output[1] - (output[1, 0])))


def test_audio_resample() -> None:
    opts = get_config()
    num_clips = 2
    num_samples = 32
    original_fps = 32
    new_fps = 8
    num_channels = 2

    setattr(opts, "audio_augmentation.audio_resample.audio_fps", new_fps)
    resample = AudioResample(opts)
    samples = {
        "audio": torch.arange(num_clips * num_samples * num_channels)
        .float()
        .view(num_clips, num_samples, num_channels),
        "metadata": {"audio_fps": original_fps},
    }

    new_data = resample({"samples": samples})
    assert new_data["samples"]["audio"].shape == (
        num_clips,
        new_fps,
        num_channels,
    )
    assert new_data["samples"]["metadata"]["audio_fps"] == new_fps


@pytest.mark.parametrize(
    "in_channels, out_channels",
    [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ],
)
def test_standardize_channels(in_channels: int, out_channels: int):
    opts = get_config()
    setattr(opts, "audio_augmentation.standardize_channels.enable", True)
    setattr(
        opts,
        "audio_augmentation.standardize_channels.num_channels",
        out_channels,
    )

    N, T = 2, 5
    data = {"samples": {"audio": torch.rand(N, T, in_channels)}}
    data = StandardizeChannels(opts)(data)
    assert data["samples"]["audio"].shape == (N, T, out_channels)


@pytest.mark.parametrize("share_clip_params", [True, False])
def test_gain(share_clip_params):
    opts = get_config()
    gain_db = -20
    setattr(opts, "audio_augmentation.gain.levels", [gain_db])
    setattr(opts, "audio_augmentation.gain.share_clip_params", share_clip_params)
    augmenter = Gain(opts)
    # Dummy audio are between [-1, 1].
    dummy_audio = 2 * (torch.rand((1, 16000)) - 0.5)
    dummy_audio = dummy_audio.float()
    data = {
        "samples": {"audio": dummy_audio},
        "metadata": {"audio_fps": 16000},
    }
    augmented = augmenter(data)
    assert data["samples"]["audio"].shape == augmented["samples"]["audio"].shape
    assert (
        augmented["samples"]["audio"] != dummy_audio
    ).float().sum() > 0.99 * dummy_audio.numel()
    assert (
        torch.abs(
            (augmented["samples"]["audio"] / dummy_audio).float().sum()
            - 10 ** (gain_db / 20) * dummy_audio.numel()
        )
        < 0.001
    )


@pytest.mark.parametrize(
    "audio_channels, audio_length, noise_length",
    [
        (1, 16000, 16000),  # perfect training case
        (2, 16000, 16000),  # multichannel input audio
        (2, 10000, 16000),  # input audio length < noise length
        (2, 16000, 10000),  # input audio length > noise length
    ],
)
def test_noise_augmentation(audio_channels, audio_length, noise_length):
    try:
        opts = get_config()
        setattr(opts, "audio_augmentation.noise.levels", [-32])
        setattr(opts, "audio_augmentation.noise.cache_size", 1)
        setattr(opts, "audio_augmentation.noise.refresh_freq", 1)
        noise_file_dir = tempfile.mkdtemp()
        setattr(
            opts,
            "audio_augmentation.noise.files_dir",
            noise_file_dir,
        )
        dummy_noise = ((torch.rand((noise_length, 1)) - 0.5) * 2).numpy()
        wav.write(
            os.path.join(noise_file_dir, "noise_sample1.wav"),
            data=dummy_noise,
            rate=16000,
        )
        augmenter = Noise(opts)
        dummy_audio = (torch.rand((audio_channels, audio_length)) - 0.5) * 2
        data = {
            "samples": {"audio": dummy_audio},
            "metadata": {"audio_fps": 16000},
        }
        augmented = augmenter(data)
        assert data["samples"]["audio"].shape == augmented["samples"]["audio"].shape
        assert (
            augmented["samples"]["audio"] != dummy_audio
        ).float().sum() > 0.99 * dummy_audio.numel()
    finally:
        shutil.rmtree(noise_file_dir)


def test_no_noise_augmentation():
    # When noise.files_dir is None
    with pytest.raises(ValueError):
        opts = get_config()
        setattr(opts, "audio_augmentation.noise.levels", [-32])
        setattr(opts, "audio_augmentation.noise.cache_size", 1)
        setattr(opts, "audio_augmentation.noise.refresh_freq", 1)
        setattr(
            opts,
            "audio_augmentation.noise.files_dir",
            None,
        )
        augmenter = Noise(opts)
        dummy_audio = (torch.rand(-16000, 16000, (2, 32000)) - 0.5) * 2
        data = {
            "samples": {"audio": dummy_audio.clone()},
            "metadata": {"audio_fps": 16000},
        }
        output = augmenter(data)
        assert torch.all(output == dummy_audio)


def test_gaussian_noise_augmentation():
    opts = get_config()
    torch.random.manual_seed(1234)
    scale_range = (0.001, 0.005)
    setattr(
        opts,
        "audio_augmentation.gaussian_noise.audio_noise_scale_range",
        scale_range,
    )
    augmenter = GaussianAudioNoise(opts)
    dummy_audio = (
        2 * torch.rand((1, 16000, 1), dtype=torch.float32) - 1
    )  # Values between -1 and 1
    original_audio = copy.deepcopy(dummy_audio)
    data = {"samples": {"audio": dummy_audio}}
    augmented = augmenter(data)["samples"]["audio"]
    delta = augmented - original_audio
    assert scale_range[0] < delta.std() <= scale_range[1]
