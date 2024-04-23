#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict

import torch
from torch import Tensor

from corenet.modeling.models import MODEL_REGISTRY, BaseAnyNNModel


@MODEL_REGISTRY.register(name="__base__", type="language_modeling")
class BaseLanguageModel(BaseAnyNNModel):
    """Base class for language modeling.

    Args:
        opts: Command-line arguments.
    """

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add LM model specific arguments"""
        if cls == BaseLanguageModel:
            group = parser.add_argument_group(title=cls.__name__)
            group.add_argument(
                "--model.language-modeling.name",
                type=str,
                default=None,
                help="Name of the language model. Defaults to None (i.e., user need to specify the model name).",
            )

            group.add_argument(
                "--model.language-modeling.pretrained",
                type=str,
                default=None,
                help="Path of the pre-trained model. Defaults to None (i.e., user needs to specify the path of pre-trained model).",
            )

        return parser

    def dummy_input_and_label(self, batch_size: int) -> Dict[str, Tensor]:
        """Create dummy input and labels for CI/CD purposes. Child classes must override it
        if functionality is different.
        """
        seq_length = 5
        return {
            "samples": torch.randint(
                low=0, high=1, size=(batch_size, seq_length)
            ).long(),
            "targets": torch.randint(
                low=0, high=1, size=(batch_size, seq_length)
            ).long(),
        }

    @classmethod
    def build_model(cls, opts: argparse.Namespace, *args, **kwargs) -> BaseAnyNNModel:
        """Helper function to build the language model.

        Args:
            opts: Command-line arguments.

        Returns:
            An instance of `corenet.modeling.models.BaseAnyNNModel`.
        """
        model = cls(opts, *args, **kwargs)
        return model
