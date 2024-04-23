#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Tuple
from unittest import mock

import pytest
import torch
from torch import Tensor

from corenet.data.collate_fns import collate_functions
from corenet.data.datasets.audio_classification.speech_commands_v2 import (
    SpeechCommandsv2Dataset,
)


def mock_load(filepath: str) -> Tuple[Tensor, int]:
    audio_rate = 1024
    outputs = torch.ones([1, audio_rate], dtype=torch.float) / 255
    for i, elem in enumerate(filepath):
        outputs[:, i] = int(elem) / 256
    return outputs, audio_rate


def mock_process_dataset_config(self) -> None:
    self.dataset_entries = [
        {"audio_filepath": "0", "command": "up"},
        {"audio_filepath": "1", "command": "down"},
        {"audio_filepath": "2", "command": "up"},
    ]

    self.noise_files = [
        "0",
        "1",
    ]

    self.label_to_index = {"up": 0, "down": 1}


@pytest.mark.parametrize(
    "is_training,is_evaluation,length,mixup",
    [(True, False, 16, False), (False, True, 8, True), (False, False, 4, True)],
)
@mock.patch("torchaudio.load", mock_load)
@mock.patch(
    "corenet.data.datasets.audio_classification.speech_commands_v2.SpeechCommandsv2Dataset._process_dataset_config",
    mock_process_dataset_config,
)
def test_getitem(
    is_training: bool, is_evaluation: bool, length: int, mixup: bool
) -> None:
    opts = argparse.Namespace()
    setattr(opts, "dataset.speech_commands_v2.mixup", mixup)
    setattr(opts, "dataset.root_train", "/tmp/train")
    setattr(opts, "dataset.root_val", "/tmp/val")
    setattr(opts, "dataset.root_test", "/tmp/test")
    setattr(opts, "audio_augmentation.set_fixed_length.enable", True)
    setattr(opts, "audio_augmentation.set_fixed_length.length", length)
    setattr(opts, "audio_augmentation.noise.enable", False)
    setattr(opts, "audio_augmentation.roll.enable", False)

    dataset = SpeechCommandsv2Dataset(
        opts, is_training=is_training, is_evaluation=is_evaluation
    )

    value1 = dataset[(None, None, 0)]
    value2 = dataset[(None, None, 1)]

    assert torch.any(value1["samples"]["audio"] != value2["samples"]["audio"])
    assert value1["samples"]["audio"].shape == (1, length)

    # Make sure the elements can be collated.
    collate_functions.pytorch_default_collate_fn([value1, value2], opts)
