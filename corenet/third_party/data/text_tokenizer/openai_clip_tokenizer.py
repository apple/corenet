#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import gzip
from functools import lru_cache
from typing import Dict, List, Set, Tuple, Union

import regex as re
import torch
from torch import Tensor

from corenet.data.text_tokenizer import TOKENIZER_REGISTRY, BaseTextTokenizer
from corenet.utils import logger
from corenet.utils.download_utils import get_local_path


@lru_cache()
def bytes_to_unicode() -> Dict[int, str]:
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    This also avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


@TOKENIZER_REGISTRY.register(name="openai_clip")
class OpenAIClipTokenizer(BaseTextTokenizer):
    """OpenAI's CLIP tokenizer.

    The code is taken from https://github.com/openai/CLIP.

    Args:
        opts: Command-line arguments.

    Example:
        >>> tokenizer = OpenAIClipTokenizer(opts)
        >>> input_sentence = "the quick brown fox jumped over the lazy dog"
        >>> tokenized_sentence = tokenizer(input_sentence)
        >>> print(tokenized_sentence)
            tensor([49406,   518,  3712,  2866,  3240, 16901,   962,   518, 10753,  1929, 49407])
        >>> tokenizer.tok_decode(tokenized_sentence)
            '<|startoftext|>the quick brown fox jumped over the lazy dog <|endoftext|>'

    ...note:
        1. BPE file can be downloaded from OpenAI's CLIP github as:
        ```
            wget https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz
        ```
    """

    def __init__(self, opts: argparse.Namespace) -> None:
        super().__init__(opts)
        bpe_path = getattr(opts, "text_tokenizer.openai_clip.bpe_path")
        if bpe_path is None:
            logger.error(
                f"BPE path cannot be None in {self.__class__.__name__}. Please check."
            )

        bpe_path = get_local_path(
            opts,
            path=bpe_path,
            force_delete=False,
        )

        # merges contain pair of tokens that are frequently appearing in the corpora.
        # Example: ['i n', 't h', 'a n', 'r e', 'a r', 'e r', 'th e</w>', 'in g</w>', 'o u']
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")

        # Note 1: index 0 in merges file contain version information (e.g., 'bpe_simple_vocab_16e6.txt#version: 0.2').
        # Note 2: The OpenAI CLIP model operates on a lower-cased byte pair encoding (BPE) representation of the text with a 49,152 vocab size.
        # Note 3a: For each byte, we have ['a', 'a</w>']. so, 256 is subtracted.
        # Note 3b: We also add SOT and EOT tokens to the vocab, so we further subtract 2.
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]

        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend([self.sot_token, self.eot_token])

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.cache = {
            self.sot_token: self.sot_token,
            self.eot_token: self.eot_token,
        }
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """OpenAIClipTokenizer arguments."""
        if cls == OpenAIClipTokenizer:
            group = parser.add_argument_group(cls.__name__)

            openai_tokenizer_file_permanent_path = "https://github.com/openai/CLIP/raw/a1d071733d7111c9c014f024669f959182114e33/clip/bpe_simple_vocab_16e6.txt.gz"

            group.add_argument(
                "--text-tokenizer.openai-clip.bpe-path",
                type=str,
                default=openai_tokenizer_file_permanent_path,
                help=f"Path to BPE file. Defaults to '{openai_tokenizer_file_permanent_path}'.",
            )
        return parser

    @property
    def vocab_size(self) -> int:
        """Text vocabulary size."""
        return len(self.encoder)

    @property
    def eot_token(self) -> str:
        """End of text token."""
        return "<|endoftext|>"

    @property
    def eot_token_id(self) -> int:
        """Token index for EOT token."""
        return self.encoder[self.eot_token]

    @property
    def sot_token(self) -> str:
        """Start of text token."""
        return "<|startoftext|>"

    @property
    def sot_token_id(self) -> int:
        """Start of token index."""
        return self.encoder[self.sot_token]

    @property
    def pad_token_id(self) -> int:
        """Padding token index."""
        return self.encoder[self.pad_token]

    def _bpe(self, token: str) -> str:
        """Convert token to byte pair encoding (BPE).

        Args:
            token: Text token (e.g., the word 'the')

        Returns:
            Byte-pair encoding of the @token (e.g., 'the</w>').
        """
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def tok_encode(self, input_sentence: str) -> Tensor:
        """Encodes a sentence into a tensor of token IDs.

        The byte-pair encodings are obtained for text tokens, which are subsequently converted into
        token ids and returned as a tensor.

        Args:
            input_sentence: The input sentence to be tokenized and encoded.

        Returns:
            A 1D tensor containing token IDs representing the encoded input sentence. The length of
            tokenized sentence is dependent on 'input_sentence'.

        ...note:
            Special tokens SOT (Start of Text) and EOT (End of Text) are added to the input
            sentence before tokenization.
        """
        input_sentence = f"{self.sot_token} {input_sentence} {self.eot_token}"
        bpe_tokens = []
        for token in re.findall(self.pat, input_sentence):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self._bpe(token).split(" ")
            )
        bpe_tokens_tensor = torch.tensor(bpe_tokens, dtype=torch.long)
        return bpe_tokens_tensor

    def tok_decode(self, token_ids: Union[List[int], Tensor]) -> str:
        """Decodes list of token ids into a sentence.

        Args:
            token_ids: A list of token ids or a 1D tensor containing token ids.

        Returns:
            The decoded sentence.
        """
        if isinstance(token_ids, Tensor):
            token_ids = token_ids.numpy().tolist()
        text = "".join([self.decoder[token] for token in token_ids])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text
