#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from argparse import Namespace

import torch

from corenet.third_party.data.text_tokenizer.openai_clip_tokenizer import (
    OpenAIClipTokenizer,
)


def test_openai_clip_tokenizer():
    """Test for OpenAIClipTokenizer."""
    opts = Namespace()

    setattr(
        opts,
        "text_tokenizer.openai_clip.bpe_path",
        "https://github.com/openai/CLIP/raw/a1d071733d7111c9c014f024669f959182114e33/clip/bpe_simple_vocab_16e6.txt.gz",
    )
    tokenizer = OpenAIClipTokenizer(opts)
    out = tokenizer("the quick brown fox jumped over the lazy dog")

    expected_data = [
        49406,  # Start token id
        518,
        3712,
        2866,
        3240,
        16901,
        962,
        518,
        10753,
        1929,
        49407,  # end token id
    ]
    expected_out = torch.tensor(expected_data, dtype=out.dtype)
    torch.testing.assert_close(actual=out, expected=expected_out)
    assert tokenizer.sot_token == "<|startoftext|>"
    assert tokenizer.eot_token == "<|endoftext|>"
    assert tokenizer.sot_token_id == 49406
    assert tokenizer.eot_token_id == 49407
