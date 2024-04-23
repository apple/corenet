#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from corenet.modeling.layers import linear_layer
from corenet.modeling.layers.normalization import layer_norm


class TokenMerging(nn.Module):
    """
    Merge tokens from a [batch_size, sequence_length, num_channels] tensor
    using a linear projection.

    This function also updates masks and adds padding as needed to make the
    sequence length divisible by the window size before merging tokens.

    Args:
        dim: Number of input channels.
        window: The size of the window to merge into a single token.
    """

    def __init__(self, dim: int, window: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.reduction = linear_layer.LinearLayer(window * dim, dim, bias=False)
        self.norm = layer_norm.LayerNorm(dim)
        self.window = window

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform token merging.

        Args:
            x: A tensor of shape [batch_size, sequence_length, num_channels].
            key_padding_mask: A tensor of shape [batch_size, sequence_length]
                with "-inf" values at mask tokens, and "0" values at unmasked
                tokens.

        Returns:
            A tensor of shape [batch_size, math.ceil(sequence_length /
                self.window), num_channels], where @self.window is the window
                size.
        """
        if key_padding_mask is not None:
            # Zero out the masked portion of @x to make sure it doesn't
            # participate in linear projections after windowing.
            x[key_padding_mask == float("-inf")] = 0

        x, key_padding_mask = pad_x_and_mask(x, key_padding_mask, self.window)
        B, N, C = x.shape

        x = x.unfold(1, self.window, self.window)  # [B, N // window, C, window]
        x = x.reshape(B, N // self.window, C * self.window)
        x = self.reduction(x)  # [B, N // self.window, C]
        x = self.norm(x)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (B, N)
            key_padding_mask = key_padding_mask.unfold(
                1, self.window, self.window
            )  # [B, N // window, window].
            key_padding_mask = key_padding_mask.max(dim=-1).values  # [B, N // window].

        return x, key_padding_mask

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window={self.window}"


def pad_x_and_mask(
    x: torch.Tensor, key_padding_mask: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply padding to @x and @key_padding_mask to make their lengths divisible
    by @window_size.

    Args:
        x: The input tensor of shape [B, N, C].
        key_padding_mask: The mask of shape [B, N].
        window_size: the N dimension of @x and @key_padding_mask will be padded
            to make them divisble by this number.

    Returns:
        A tuple containing @x and @key_padding_mask, with padding applied.
    """
    B, N, _ = x.shape
    padding = (window_size - (N % window_size)) % window_size

    if key_padding_mask is not None:
        key_padding_mask = F.pad(key_padding_mask, (0, padding), value=float("-inf"))

    # Apply similar padding to x which is [B, N, C] before padding.
    x = F.pad(x, (0, 0, 0, padding), value=0)

    return x, key_padding_mask
