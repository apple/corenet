#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Optional, Tuple

import torch
from torch import Tensor

from corenet.modeling.layers import token_merging
from corenet.modeling.modules import transformer


def window_partition(t: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partition tensor @t into chunks of size @window_size.

    @t's sequence length must be divisible by @window_size.

    Args:
        t: A tensor of shape [batch_size, sequence_length, embed_dim].
        window_size: The desired window size.

    Returns:
        A tensor of shape [batch_size * sequence_length // window_size,
        window_size, embed_dim].
    """
    B, N, C = t.shape

    if not N % window_size == 0:
        raise ValueError(
            f"sequence length {N} must be divisible by window size {window_size}"
        )

    t = t.reshape(B * N // window_size, window_size, C)
    return t


def window_partition_reverse(
    t: torch.Tensor, B: int, num_windows: int, C: int
) -> torch.Tensor:
    """
    Undo the @window_partition operation.

    Args:
        t: The input tensor of shape [batch_size * num_windows, window_size,
            embed_dim].
        B: The batch size.
        num_windows: The number of windows.
        C: The embedding dimension.

    Returns:
        A tensor of shape [batch_size, num_windows * window_size, embed_dim].
    """
    t = t.reshape(B, num_windows * t.shape[1], C)
    return t


def get_windows_shift_mask(
    N: int, window_size: int, window_shift: int, device: torch.device
) -> torch.Tensor:
    """
    Get the mask window required due to window shifting (needed for shifted
    window attention).

    This produces a tensor with mask values for each window. Most windows don't
    require masking, but windows that bleed across the beginning/end of the
    tensor (due to shifting) require it.

    Args:
        N: The sequence length.
        window_size: The window size.
        window_shift: The window shift.
        device: The device on which to create the tensor.

    Returns:
        A tensor of shape [N // window_size, window_size, window_size]
        containing mask values. The values are 0 (unmasked) or float("-inf")
        (masked).
    """
    ret = torch.zeros(N // window_size, window_size, window_size, device=device)
    ret[-1].fill_(float("-inf"))
    ret[-1, : window_size - window_shift, : window_size - window_shift] = 0
    ret[-1, -window_shift:, -window_shift:] = 0
    return ret


def window_x_and_key_padding_mask(
    x: torch.Tensor, key_padding_mask: torch.Tensor, window_size: int, window_shift: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform windowing on @x and @key_padding_mask in preparation for windowed
    attention.

    Args:
        x: The input tensor of shape [batch_size, sequence_length, num_channels].
        key_padding_mask: The mask, as a tensor of shape [batch_size, sequence_length].
        window_size: The window size to be used for windowed attention.
        window_shift: The window shift to be used for windowed attention.

    Returns:
        A tuple containing 3 tensors. The first is the windowed input. The second
        is the windowed mask. The third is the mask needed to perform shifted
        window attention (to avoid the first and last windows from bleeding
        into each other).
    """
    B, N = key_padding_mask.shape
    assert x.shape[:2] == (B, N)

    x, key_padding_mask = token_merging.pad_x_and_mask(x, key_padding_mask, window_size)

    # Now, perform the windowing.
    if window_shift > 0:
        x = torch.roll(x, shifts=(-window_shift), dims=1)
        key_padding_mask = torch.roll(key_padding_mask, shifts=(-window_shift), dims=1)

    x_windows = window_partition(x, window_size)
    token_mask_windows = key_padding_mask.reshape(
        B * x.shape[1] // window_size, window_size
    )
    window_mask = get_windows_shift_mask(
        x.shape[1], window_size, window_shift, x_windows.device
    ).expand(B, -1, -1, -1)
    window_mask = window_mask.reshape(
        window_mask.shape[0] * window_mask.shape[1],
        window_mask.shape[2],
        window_mask.shape[3],
    )

    return x_windows, token_mask_windows, window_mask


def unwindow_x(x_windows: torch.Tensor, B: int, N: int, C: int, window_shift: int):
    """
    Undoes the operation of @window_x_and_attention on the input tensor @x_windows.

    Args:
        x_windows: The input tensor to unwindow. Its shape is [batch_size *
              padded_sequence_length // window_size, window_size, embed_dim].
        B: The batch size. Referred to as batch_size in this docstring.
        N: The sequence length of the tensor before windowing. Referred to as
            sequence_length in this docstring.
        C: The number of channels. Referred to as embed_dim in this docstring.
        window_shift: The shift applied to the sequence before the windowing
            originally occurred.

    Returns:
        A tensor of shape [batch_size, sequence_length, embed_dim].
    """
    num_windows = x_windows.shape[0] // B
    x = window_partition_reverse(x_windows, B, num_windows, C)

    if window_shift > 0:
        x = torch.roll(x, shifts=window_shift, dims=1)
    x = x[:, :N]

    return x


class WindowedTransformerEncoder(transformer.TransformerEncoder):
    """
    This class defines the pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_
    with the addition of windowed attention.

    This class first partitions the input sequence into a series of windows (with
    an optional offset to use when defining windows). Then, it calls a
    TransformerEncoder module. Then, it undoes windowing.

    Args:
        opts: Command line arguments.
        embed_dim: :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`.
        ffn_latent_dim: Inner dimension of the FFN.
        num_heads: Number of heads in multi-head attention. Default: 8.
        attn_dropout: Dropout rate for attention in multi-head attention. Default: 0.0.
        dropout: Dropout rate. Default: 0.0.
        ffn_dropout: Dropout between FFN layers. Default: 0.0.
        transformer_norm_layer: Normalization layer. Default: layer_norm.
        stochastic_dropout: Stochastic dropout setting. Default: 0.0.
        window_size: The size of the window, if using windowed attention. Default: None.
        window_shift: The size of the shift, if using shifted windowed attention. Default: None.

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        embed_dim: int,
        ffn_latent_dim: int,
        num_heads: Optional[int] = 8,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        transformer_norm_layer: Optional[str] = "layer_norm",
        stochastic_dropout: Optional[float] = 0.0,
        window_size: Optional[int] = None,
        window_shift: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            opts=opts,
            embed_dim=embed_dim,
            ffn_latent_dim=ffn_latent_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            transformer_norm_layer=transformer_norm_layer,
            stochastic_dropout=stochastic_dropout,
        )
        if window_size is None:
            raise ValueError("Please specify window_size")
        if window_shift is None:
            raise ValueError("Please specify window_shift")
        self.window_size: int = window_size
        self.window_shift: int = window_shift

    def forward(
        self,
        x: Tensor,
        x_prev: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs,
    ) -> Tensor:
        """
        Compute the outputs of the WindowedTransformerEncoder on an input.

        Args:
            x: The input tensor, of shape [batch_size, sequence_length, embed_dim].
            x_prev: The context input, if using cross-attention. Its shape is
                [batch_size, sequence_length_2, embed_dim].
            key_padding_mask: An optional tensor of masks to be applied to the
                inputs @x. Its shape is [batch_size, sequence_length].
            attn_mask: An optional attention mask. Its shape is [batch_size,
                sequence_length, sequence_length_2]. (If using self-attention,
                the sequence lengths will be equal.)

        Returns:
            The WindowedTransformerEncoder output.
        """
        B, N, C = x.shape
        x, windowed_key_padding_mask, windows_mask = window_x_and_key_padding_mask(
            x, key_padding_mask, self.window_size, self.window_shift
        )
        total_mask = windowed_key_padding_mask.unsqueeze(1) + windows_mask

        if attn_mask is not None:
            total_mask += attn_mask

        # If an entire window is masked out, attention is computed across
        # only -inf values, which gives NaN. We instead set these masks to
        # 0 to avoid this.
        fully_masked_windows = total_mask.max(dim=-1).values == float("-inf")
        total_mask[fully_masked_windows] = 0

        x = super().forward(x, x_prev, attn_mask=attn_mask)

        # Undo windowing.
        x = unwindow_x(x, B, N, C, self.window_shift)
        return x

    def __repr__(self) -> str:
        # Remove closing parentheses from parent __repr__ call.
        ret = super().__repr__()[:-1]
        return f"{ret}, {self.window_size}, {self.window_shift})"
