#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import sys

import numpy as np
import pytest

sys.path.append("..")

from corenet.modeling.layers.positional_embedding import PositionalEmbedding


@pytest.mark.parametrize(
    "is_learnable, input_seq_len, sequence_first, padding_idx",
    [
        (True, 34, True, None),
        (False, 128, False, 0),
        (False, 192, True, 0),
    ],
)
def test_pos_embedding(
    is_learnable: bool, input_seq_len: int, sequence_first: bool, padding_idx: int
):
    num_embeddings = 128
    pos_embedding = PositionalEmbedding(
        opts=None,
        num_embeddings=num_embeddings,
        embedding_dim=512,
        padding_idx=padding_idx,
        is_learnable=is_learnable,
        sequence_first=sequence_first,
    )
    seq_dim = 0 if sequence_first else 1

    out = pos_embedding(input_seq_len)
    np.testing.assert_equal(out.shape[seq_dim], input_seq_len)
