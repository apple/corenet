#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from argparse import Namespace

import torch

from corenet.data.text_tokenizer.clip_tokenizer import ClipTokenizer


def test_clip_tokenizer():
    """Test for 'ClipTokenizer'."""
    opts = Namespace()

    setattr(
        opts,
        "text_tokenizer.clip.merges_path",
        "http://download.pytorch.org/models/text/clip_merges.bpe",
    )
    setattr(
        opts,
        "text_tokenizer.clip.encoder_json_path",
        "http://download.pytorch.org/models/text/clip_encoder.json",
    )

    tokenizer = ClipTokenizer(opts=opts)
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
