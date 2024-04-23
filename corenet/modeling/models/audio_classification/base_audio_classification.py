#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.modeling.models import MODEL_REGISTRY, BaseAnyNNModel


@MODEL_REGISTRY.register(name="__base__", type="audio_classification")
class BaseAudioClassification(BaseAnyNNModel):
    """Base class for audio classification.

    Args:
        opts: Command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add model specific arguments"""
        if cls != BaseAudioClassification:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.audio-classification.name",
            type=str,
            default=None,
            help="Name of the audio classification model. Defaults to None.",
        )
        group.add_argument(
            "--model.audio-classification.pretrained",
            type=str,
            default=None,
            help="Path of the pretrained backbone. Defaults to None.",
        )
        return parser
