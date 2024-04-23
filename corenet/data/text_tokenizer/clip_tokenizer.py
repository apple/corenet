#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

import torch
from torch import Tensor
from torchtext.transforms import CLIPTokenizer

from corenet.data.text_tokenizer import TOKENIZER_REGISTRY, BaseTextTokenizer
from corenet.utils import logger
from corenet.utils.download_utils import get_local_path


@TOKENIZER_REGISTRY.register(name="clip")
class ClipTokenizer(BaseTextTokenizer):
    """Tokenizer for CLIP.

    This tokenizer uses torchtext's CLIP Tokenizer to tokenize input sentence into token ids.
    Please see `torchtext documentation <https://pytorch.org/text/stable/transforms.html#torchtext.transforms.CLIPTokenizer>`_ for details.

    Args:
        opts: Command line arguments.
    """

    def __init__(self, opts: argparse.Namespace) -> None:
        merges_path = getattr(opts, "text_tokenizer.clip.merges_path")
        if merges_path is None:
            logger.error(
                "Please specify BPE merge file using --text-tokenizer.clip.merges-path argument"
            )

        # DDP case is handled internally
        merges_path = get_local_path(opts, path=merges_path, force_delete=False)

        encoder_json_path = getattr(opts, "text_tokenizer.clip.encoder_json_path")
        if encoder_json_path is None:
            logger.error(
                "Please specify Encoder JSON file using --text-tokenizer.clip.encoder-json-path argument"
            )

        encoder_json_path = get_local_path(
            opts, path=encoder_json_path, force_delete=False
        )

        super().__init__(opts)
        self.tokenizer = CLIPTokenizer(
            merges_path=merges_path, encoder_json_path=encoder_json_path
        )
        # BPE encodings is a dict, where  keys are tokens and values are token_ids
        self.bpe_encodings = self.tokenizer.bpe.bpe_encoder_

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls == ClipTokenizer:
            group = parser.add_argument_group(title=cls.__name__)

            group.add_argument(
                "--text-tokenizer.clip.merges-path",
                type=str,
                default=None,
                help="Path to bpe merges file. Defaults to None.",
            )

            group.add_argument(
                "--text-tokenizer.clip.encoder-json-path",
                type=str,
                default=None,
                help="Path to BPE encoder json file. This file is used to infer `num_merges`. Defaults to None.",
            )
        return parser

    @property
    def vocab_size(self) -> int:
        """Text vocabulary size."""
        return len(self.bpe_encodings)

    @property
    def eot_token(self) -> str:
        """End of text token."""
        return "<|endoftext|>"

    @property
    def eot_token_id(self) -> int:
        """Token index for EOT token."""
        return int(self.tokenizer(self.eot_token)[0])

    @property
    def sot_token(self) -> str:
        """Start of text token."""
        return "<|startoftext|>"

    @property
    def sot_token_id(self) -> int:
        """Start of token index."""
        return int(self.tokenizer(self.sot_token)[0])

    def tok_encode(self, input_sentence: str) -> Tensor:
        """Encodes a sentence into a tensor of token ids.

        ...note:
            SOT and EOT tokens are added to input sentence before tokenization.
        """
        input_sentence = f"{self.sot_token} {input_sentence} {self.eot_token}"
        # tokenizer returns indices as a string
        tokenized_sentence = self.tokenizer(input_sentence)
        # convert string to int and then create a tensor
        tokenized_sentence = torch.tensor(
            [int(cap) for cap in tokenized_sentence], dtype=torch.long
        )
        return tokenized_sentence
