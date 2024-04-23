#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import pytest
import torch

import corenet.modeling.modules
from corenet.modeling.modules import WindowedTransformerEncoder
from tests.modeling.modules.test_transformer import ensure_equal_in_range, get_opts


@pytest.mark.parametrize(
    "N,window_size,window_shift", [(32, 16, 4), (32, 7, 2), (15, 9, 4)]
)
def test_get_windowed_attention_mask(N, window_size, window_shift) -> None:
    mask = corenet.modeling.modules.windowed_transformer.get_windows_shift_mask(
        N, window_size, window_shift, device=torch.device("cpu")
    )

    expected = torch.zeros(N // window_size, window_size, window_size)
    expected[-1].fill_(float("-inf"))
    expected[-1, : window_size - window_shift, : window_size - window_shift] = 0
    expected[-1, -window_shift:, -window_shift:] = 0

    assert torch.all(mask == expected)


def test_windowed_attention() -> None:
    opts = get_opts()

    B, N, C = 2, 66, 8
    window_size = 16
    window_shift = 0
    t = WindowedTransformerEncoder(
        opts,
        embed_dim=C,
        ffn_latent_dim=4 * C,
        window_size=window_size,
        window_shift=window_shift,
    )

    # These are the boundaries at which the tensor values change.
    chunk_bounds = [0, 16, 32, 48, 64, 66]
    x = torch.empty([B, N, C])
    for start, end in zip(chunk_bounds[0:-1], chunk_bounds[1:]):
        prototype = torch.randn([C])
        x[:, start:end] = prototype

    key_padding_mask = torch.zeros([B, N])
    key_padding_mask[0, 63:] = float("-inf")
    # Mask the @x values at the masked positions.
    x[0, 63:] = 0

    y = t(x, key_padding_mask=key_padding_mask)

    # We expect @y[0] values to change at the boundaries defined by chunk_bounds.
    ensure_equal_in_range(y[0], 0, 16)
    ensure_equal_in_range(y[0], 16, 32)
    ensure_equal_in_range(y[0], 32, 48)
    ensure_equal_in_range(y[0], 48, 63)
    # Masking will cause a change at index 63.
    ensure_equal_in_range(y[0], 63, 64)
    ensure_equal_in_range(y[0], 64, 66)

    # We expect @y[1] values to change at the boundaries defined by chunk_bounds.
    ensure_equal_in_range(y[1], 0, 16)
    ensure_equal_in_range(y[1], 16, 32)
    ensure_equal_in_range(y[1], 32, 48)
    ensure_equal_in_range(y[1], 48, 64)
    ensure_equal_in_range(y[1], 64, 66)


def test_windowed_attention_shift() -> None:
    opts = get_opts()

    B, N, C = 2, 66, 8
    window_size = 16
    window_shift = 8
    t = WindowedTransformerEncoder(
        opts,
        embed_dim=C,
        ffn_latent_dim=4 * C,
        window_size=window_size,
        window_shift=window_shift,
    )

    # These are the boundaries at which the tensor values change.
    chunk_bounds = [0, 8, 24, 40, 56, 66]
    x = torch.empty([B, N, C]).fill_(float("-inf"))
    for start, end in zip(chunk_bounds[0:-1], chunk_bounds[1:]):
        prototype = torch.randn([C])
        x[:, start:end] = prototype

    key_padding_mask = torch.zeros([B, N])
    key_padding_mask[0, 63:] = float("-inf")
    # Mask the @x values at the masked positions.
    x[0, 63:] = 0

    y = t(x, key_padding_mask=key_padding_mask)

    # We expect @y[0] values to change at the boundaries defined by chunk_bounds.
    # The values are offset by the shift of 8 as well, since windowing will
    # occur at indices congruent to 8 mod 16.
    ensure_equal_in_range(y[0], 0, 8)
    ensure_equal_in_range(y[0], 8, 24)
    ensure_equal_in_range(y[0], 24, 40)
    ensure_equal_in_range(y[0], 40, 56)
    ensure_equal_in_range(y[0], 56, 63)
    # Masking at index 63 changes the values.
    ensure_equal_in_range(y[0], 63, 66)

    # We expect @y[0] values to change at the boundaries defined by chunk_bounds.
    ensure_equal_in_range(y[1], 0, 8)
    ensure_equal_in_range(y[1], 8, 24)
    ensure_equal_in_range(y[1], 24, 40)
    ensure_equal_in_range(y[1], 40, 56)
    ensure_equal_in_range(y[1], 56, 66)
