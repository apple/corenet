#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

import torch

from corenet.data.collate_fns import byteformer_collate_functions


def test_byteformer_image_collate_fn() -> None:
    torch.manual_seed(1)
    C, H, W = 3, 8, 8
    batch = [
        {
            "samples": torch.rand(C, H, W),
        },
        {
            "samples": torch.rand(C, H, W),
        },
    ]

    padding_index = -1
    opts = argparse.Namespace()
    setattr(opts, "image_augmentation.pil_save.enable", True)
    setattr(opts, "image_augmentation.pil_save.file_encoding", "JPEG")
    setattr(opts, "image_augmentation.pil_save.quality", 100)
    setattr(opts, "image_augmentation.shuffle_bytes.enable", False)
    setattr(opts, "image_augmentation.mask_positions.enable", False)
    setattr(opts, "image_augmentation.random_uniform.enable", False)
    setattr(opts, "image_augmentation.byte_permutation.enable", False)
    setattr(opts, "image_augmentation.torchaudio_save.enable", False)
    setattr(opts, "model.classification.byteformer.padding_index", padding_index)

    collated_batch = byteformer_collate_functions.byteformer_image_collate_fn(
        batch, opts
    )

    assert list(collated_batch.keys()) == ["samples"]
    # Padding token should be used
    assert padding_index in collated_batch["samples"]


def test_wav_collate() -> None:
    C, N = 1, 16000
    batch = [
        {"samples": {"audio": torch.rand([C, N])}, "metadata": {"audio_fps": 16000}},
        {"samples": {"audio": torch.rand([C, N])}, "metadata": {"audio_fps": 16000}},
    ]

    padding_index = -1
    opts = argparse.Namespace()
    setattr(opts, "audio_augmentation.torchaudio_save.enable", True)
    setattr(opts, "audio_augmentation.torchaudio_save.encoding_dtype", "float32")
    setattr(opts, "audio_augmentation.torchaudio_save.format", "mp3")
    setattr(opts, "audio_augmentation.torchaudio_save.backend", "sox")
    setattr(opts, "model.classification.byteformer.padding_index", padding_index)

    collated_batch = byteformer_collate_functions.byteformer_audio_collate_fn(
        batch, opts
    )

    assert list(collated_batch.keys()) == ["samples"]
