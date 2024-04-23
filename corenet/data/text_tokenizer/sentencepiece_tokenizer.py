#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from argparse import Namespace
from typing import List, Union

import ftfy
import torch
from torch import Tensor

from corenet.data.text_tokenizer import TOKENIZER_REGISTRY, BaseTextTokenizer
from corenet.utils import logger
from corenet.utils.ddp_utils import is_rank_0_worker_0
from corenet.utils.download_utils import get_local_path


@TOKENIZER_REGISTRY.register(name="sentence_piece")
class SentencePieceTokenizer(BaseTextTokenizer):
    """Sentence piece tokenizer.

    Args:
        opts: Command-line arguments.

    ...note:
        Sentence piece library needs to be installed in order to use this tokenizer.
        It can be installed as:
        ```
            pip install -e '.[sentencepiece]'
        ```
    """

    def __init__(self, opts: Namespace) -> None:
        super().__init__(opts)

        try:
            from sentencepiece import SentencePieceProcessor
        except ModuleNotFoundError:
            raise ModuleNotFoundError("Please install sentencepiece library.")

        spm_model_path = getattr(self.opts, "text_tokenizer.sentence_piece.model_path")
        if spm_model_path is None:
            logger.error(
                f"Model path can't be None in {self.__class__.__name__}. Please specify using 'text_tokenizer.sentence_piece.model_path' in config file."
            )
        spm_model_local_path = get_local_path(
            opts=self.opts,
            path=spm_model_path,
            force_delete=False,
            use_start_rank=True,
            sync_ranks=False,
        )
        self.log_warning_once_on_rank0_worker0 = is_rank_0_worker_0(opts)

        self.sp_model = SentencePieceProcessor(model_file=spm_model_local_path)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add arguments related to sentence piece tokenizer."""
        if cls == SentencePieceTokenizer:
            group = parser.add_argument_group(cls.__name__)
            group.add_argument(
                "--text-tokenizer.sentence-piece.model-path",
                type=str,
                default=None,
                help="Sentence peice model path. Defaults to None (i.e., user need to supply the model path).",
            )
            group.add_argument(
                "--text-tokenizer.sentence-piece.enable-nfc-normalization",
                action="store_true",
                default=False,
                help="Normalize the text using NFC normalization. This is useful when pre-training. Defaults to False.",
            )
            group.add_argument(
                "--text-tokenizer.sentence-piece.append-sot-token",
                action="store_true",
                default=False,
                help="Append start of text token before tokenized text. Defaults to False.",
            )
            group.add_argument(
                "--text-tokenizer.sentence-piece.append-eot-token",
                action="store_true",
                default=False,
                help="Append end of text token after tokenized text. Defaults to False.",
            )

        return parser

    @property
    def vocab_size(self) -> int:
        """Vocabulary size."""
        v_size = self.sp_model.vocab_size()
        assert (
            isinstance(v_size, int) and v_size > 0
        ), f"Vocabulary size should be a positive integer. Got: {v_size}"
        return v_size

    @property
    def sot_token_id(self) -> int:
        """Start of text token index."""
        sot = self.sp_model.bos_id()
        assert (
            isinstance(sot, int) and sot > 0
        ), f"The start of text index should be a non-negative integer. Got: {sot}."
        return sot

    @property
    def eot_token_id(self) -> int:
        """End of text token index."""
        eot = self.sp_model.eos_id()
        assert (
            isinstance(eot, int) and eot > 0
        ), f"The end of text index should be a non-negative integer. Got: {eot}."
        return eot

    @property
    def pad_token_id(self) -> int:
        """Padding index.

        ...note:
            If the padding index is None or -1, we set it equal to the vocabulary size. Consequently, the range of indices
            in the vocabulary changes from '[0, vocab_size)' to '[0, vocab_size]'. This adjustment is primarily made to accommodate
            variable sequence lengths during LLM pre-training. Users should exclude the padding index from consideration in the loss
            function. They should also increase the size of embedding layer and classification layer in the model configuration to
            accommodate padding index if it is None or -1.
        """
        pad_id = self.sp_model.pad_id()

        assert pad_id is None or (
            isinstance(pad_id, int) and pad_id >= -1
        ), f"The padding index should be None or an integer greater than or equal to -1. Got: {pad_id}."
        if pad_id is None or pad_id == -1:
            pad_id = self.vocab_size
            if self.log_warning_once_on_rank0_worker0:
                logger.warning(
                    "Padding index is -1. Because -1 index does not work with embedding layer, we change it to vocab size."
                )
                self.log_warning_once_on_rank0_worker0 = False
        return pad_id

    def tok_encode(self, input_sentence: str) -> Tensor:
        """Encodes a sentence into a tensor of token ids.

        Args:
            input_sentence: Input sentence to be tokenized.

        Returns:
            A tensor containing token indices.
        """

        if getattr(self.opts, "text_tokenizer.sentence_piece.enable_nfc_normalization"):
            # normalize the text
            input_sentence = ftfy.fix_text(input_sentence, normalization="NFC")

        # tokenized sequence is returned as a list.
        tokenized_seq = self.sp_model.Encode(input_sentence)

        if getattr(self.opts, "text_tokenizer.sentence_piece.append_sot_token"):
            tokenized_seq = [self.sot_token_id] + tokenized_seq

        if getattr(self.opts, "text_tokenizer.sentence_piece.append_eot_token"):
            tokenized_seq = tokenized_seq + [self.eot_token_id]

        # convert a list into tensor
        tokenized_seq = torch.tensor(tokenized_seq, dtype=torch.long)
        return tokenized_seq

    def tok_decode(self, token_ids: Union[torch.Tensor, List[int]]) -> str:
        """Decodes token ids into a sentence.

        Args:
            token_ids: Token indices as a list of integers or a 1D integer tensor.

        Returns:
            A decoded sequence.
        """
        if isinstance(token_ids, torch.Tensor):
            assert token_ids.dim() == 1 and token_ids.dtype in [
                torch.int,
                torch.int64,
                torch.int32,
                torch.int8,
            ]
            token_ids = token_ids.numpy().tolist()

        assert isinstance(token_ids, list) and all(
            isinstance(x, int) for x in token_ids
        )
        return self.sp_model.Decode(token_ids)
