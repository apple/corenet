#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import json
import os
import random
from typing import Dict, Tuple, Union

import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F

from corenet.data.datasets import DATASET_REGISTRY, dataset_base
from corenet.data.transforms.audio import Noise, Roll, SetFixedLength
from corenet.data.transforms.common import Compose
from corenet.data.transforms.image_pil import BaseTransformation


@DATASET_REGISTRY.register(name="speech_commands_v2", type="audio_classification")
class SpeechCommandsv2Dataset(dataset_base.BaseDataset):
    """
    Google's Speech Commands dataset for keyword spotting (https://arxiv.org/abs/1804.03209).

    This contains the "v2" version for 12-way classification (10 commands,
    plus unknown and background categories).

    Args:
        opts: Command-line arguments
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(opts=opts, *args, **kwargs)
        mode_str = self.mode
        if self.mode == "val":
            # This value is needed to calculate the correct annotation .json paths.
            mode_str = "validation"

        self.dataset_config = os.path.join(self.root, f"{mode_str}_manifest.json")

        self._process_dataset_config()
        self.mixup = getattr(opts, "dataset.speech_commands_v2.mixup")

    def _process_dataset_config(self) -> None:
        """
        Process the dataset .json files to set up the dataset.

        The .json configs contain:
        [
            {
                "audio_filepath": relative path to audio from the dataset root directory,
                "duration": floating point duration in seconds,
                "command": the label of the spoken command.
            },
            ...
        ]
        """
        with open(self.dataset_config) as f:
            lines = f.readlines()

        self.dataset_entries = []
        for line in lines:
            self.dataset_entries.append(json.loads(line))

        for elem in self.dataset_entries:
            audio_path = elem["audio_filepath"]
            new_path = os.path.join(self.root, audio_path)
            elem["audio_filepath"] = new_path

        all_labels = sorted(set([elem["command"] for elem in self.dataset_entries]))
        self.label_to_index = {l: i for i, l in enumerate(all_labels)}

        if getattr(self.opts, "audio_augmentation.noise.enable"):
            # Cache this, since it loads files on initialization.
            background_dir = os.path.join(self.root, "_background_noise_")
            self.noise = Noise(self.opts, noise_files_dir=background_dir)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--dataset.speech-commands-v2.mixup",
            action="store_true",
            help="If set, apply mixup inside the dataset.",
        )
        return parser

    def _training_transforms(self, *args, **kwargs) -> BaseTransformation:
        """
        Returns transformations applied to the input in training mode.
        """
        aug_list = [
            SetFixedLength(self.opts),
        ]

        if getattr(self.opts, "audio_augmentation.noise.enable"):
            aug_list.append(self.noise)
        if getattr(self.opts, "audio_augmentation.roll.enable"):
            aug_list.append(Roll(self.opts))
        return Compose(self.opts, aug_list)

    def _validation_transforms(self, *args, **kwargs) -> BaseTransformation:
        """
        Returns transformations applied to the input in validation mode.
        """
        aug_list = [SetFixedLength(self.opts)]
        return Compose(self.opts, aug_list)

    def get_sample(self, index: int) -> Tuple[Tensor, float, Tensor]:
        """
        Get the dataset sample at the given index.
        """
        dataset_entry = self.dataset_entries[index]
        waveform, audio_fps = torchaudio.load(dataset_entry["audio_filepath"])

        label = torch.tensor(
            self.label_to_index[dataset_entry["command"]], dtype=torch.long
        )
        return waveform, audio_fps, label

    def __getitem__(
        self, batch_indexes_tup: Tuple
    ) -> Dict[str, Union[Dict[str, Tensor], Tensor, int]]:
        """
        Returns the sample corresponding to the input sample index and applies
        transforms.

        If the class uses mixup, and is in training mode, this will additionally
        apply mixup.

        Args:
            batch_indexes_tup: Tuple of the form (crop_size_h, crop_size_w, sample_index).
                The first two parts are not needed, and are ignored by this function.

        Returns:
            A sample as a dictionary. It contains:
                {
                    "samples":
                        {
                            "audio": A [C, N] tensor, where C is the number of
                                channels, and N is the length.
                        }
                    "targets": an integer class label.
                    "sample_id": an integer giving the sample index.
                    "metadata":
                        {
                            "audio_fps": The sampling rate of the audio.
                        }
                }
        """
        _, _, index = batch_indexes_tup
        data = self.get_transformed_sample(index)

        if self.mixup and self.is_training:
            index = random.randint(0, len(self) - 1)
            data2 = self.get_transformed_sample(index)

            if data["metadata"]["audio_fps"] != data2["metadata"]["audio_fps"]:
                raise ValueError(
                    f"Inconsistent audio_fps ({data['metadata']['audio_fps']} and {data2['metadata']['audio_fps']})"
                )

            coefficient = torch.rand(1)
            data["samples"]["audio"] = data["samples"]["audio"] * coefficient + data2[
                "samples"
            ]["audio"] * (1.0 - coefficient)

            def to_onehot(targets: Tensor) -> Tensor:
                return F.one_hot(targets, num_classes=len(self.label_to_index))

            data["targets"] = to_onehot(data["targets"]) * coefficient + to_onehot(
                data2["targets"]
            ) * (1.0 - coefficient)

        return data

    def get_transformed_sample(
        self, index: int
    ) -> Dict[str, Union[Dict[str, Tensor], Tensor, int]]:
        """
        Get the sample at the index specified by @index.

        Args:
            index: The index of the sample.

        Returns:
            A sample as a dictionary. It contains:
                {
                    "samples":
                        {
                            "audio": A [C, N] tensor, where C is the number of
                                channels, and N is the length.
                        }
                    "targets": an integer class label.
                    "sample_id": an integer giving the sample index.
                    "metadata":
                        {
                            "audio_fps": The sampling rate of the audio.
                        }
                }
        """
        waveform, audio_fps, label = self.get_sample(index)

        data = {
            "samples": {"audio": waveform},
            "targets": label,
            "sample_id": index,
            "metadata": {"audio_fps": audio_fps},
        }

        transform_fn = self.get_augmentation_transforms()
        data = transform_fn(data)

        return data

    def __len__(self) -> int:
        return len(self.dataset_entries)
