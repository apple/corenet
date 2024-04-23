#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any

from torch import Tensor, nn

from corenet.utils import logger


class BaseTextTokenizer(nn.Module):
    """Base class for text tokenizers.

    Args:
        opts: Command-line arguments.
    """

    def __init__(self, opts: argparse.Namespace) -> None:
        super().__init__()
        self.opts = opts

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls == BaseTextTokenizer:
            group = parser.add_argument_group(title=cls.__name__)

            group.add_argument(
                "--text-tokenizer.name",
                type=str,
                default=None,
                help="Name of the text tokenizer (e.g., clip). Defaults to None.",
            )

            group.add_argument(
                "--text-tokenizer.sot-token",
                type=str,
                default=None,
                help=f"Start of the text token. Defaults to None (i.e., users must specify the value if it needs to be used.).",
            )
            group.add_argument(
                "--text-tokenizer.eot-token",
                type=str,
                default=None,
                help=f"End of the text token. Defaults to None (i.e., users must specify the value if it needs to be used.).",
            )
            group.add_argument(
                "--text-tokenizer.pad-token",
                type=str,
                default=None,
                help=f"Pad token. Defaults to None (i.e., users must specify the value if it needs to be used.).",
            )

        return parser

    @property
    def vocab_size(self) -> int:
        """Text vocabulary size."""
        raise NotImplementedError("Child classes must implement this method.")

    @property
    def eot_token(self) -> str:
        """End of text token."""
        eot = getattr(self.opts, "text_tokenizer.eot_token")
        if eot is None:
            logger.error(
                "EOT token can't be None. Please specify using 'text_tokenizer.eot_token' in config file."
            )
        return eot

    @property
    def eot_token_id(self) -> int:
        """Token index for EOT token."""
        raise NotImplementedError("Child classes must implement this method.")

    @property
    def sot_token(self) -> str:
        """Start of text token."""
        sot = getattr(self.opts, "text_tokenizer.sot_token")
        if sot is None:
            logger.error(
                "SOT token can't be None. Please specify using 'text_tokenizer.sot_token' in config file."
            )
        return sot

    @property
    def sot_token_id(self) -> int:
        """Start of token index."""
        raise NotImplementedError("Child classes must implement this method.")

    @property
    def pad_token(self) -> str:
        """Padding token."""
        pad = getattr(self.opts, "text_tokenizer.pad_token")
        if pad is None:
            logger.error(
                "Padding token can't be None. Please specify using 'text_tokenizer.pad_token' in config file."
            )
        return pad

    @property
    def pad_token_id(self) -> int:
        """Padding index."""
        raise NotImplementedError("Child classes must implement this method.")

    def tok_encode(self, input_sentence: str) -> Tensor:
        """Encodes a sentence into a tensor of token ids."""
        raise NotImplementedError("Child classes must implement this method.")

    def tok_decode(self, token_ids: Any) -> str:
        """Decodes token ids into a sentence."""
        raise NotImplementedError("Child classes must implement this method.")

    def forward(self, input_sentence: str) -> Tensor:
        """Tokenize the input sentence.

        Args:
            input_sentence: Pre-processed input sentence.

        Returns:
            Tensor containing tokenized sequence.

        ...note:
            Input sentence should be pre-processed (e.g., lower case).
        """
        tokenized_sentence = self.tok_encode(input_sentence)
        return tokenized_sentence
