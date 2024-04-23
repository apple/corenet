#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from corenet.modeling.layers import (
    Dropout,
    LinearLayer,
    StochasticDepth,
    get_normalization_layer,
)
from corenet.modeling.layers.activation import build_activation_layer
from corenet.modeling.modules import BaseModule

"""
Most of the functions and classes below are heavily borrowed from torchvision https://github.com/pytorch/vision
"""


def _patch_merging_pad(x):
    H, W, _ = x.shape[-3:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
    return x


class Permute(BaseModule):
    """This module returns a view of the tensor input with its dimensions permuted.
    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(dims={self.dims})"
        return s


class PatchMerging(BaseModule):
    """Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (str): Normalization layer name.
        strided (Optional[bool]): Down-sample the input by a factor of 2. Default is True.
    """

    def __init__(self, opts, dim: int, norm_layer: str, strided: Optional[bool] = True):
        super().__init__()
        self.dim = dim
        self.reduction = LinearLayer(
            in_features=4 * dim, out_features=2 * dim, bias=False
        )
        self.norm = get_normalization_layer(
            opts=opts, norm_type=norm_layer, num_features=4 * dim
        )
        self.strided = strided

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Args:
            x (Tensor): input tensor with expected layout of [..., H, W, C]
        Returns:
            Tensor with layout of [..., H/2, W/2, 2*C]
        """
        x = _patch_merging_pad(x)

        if self.strided:
            x0 = x[..., 0::2, 0::2, :]  # ... H/s W/s C
            x1 = x[..., 1::2, 0::2, :]  # ... H/s W/s C
            x2 = x[..., 0::2, 1::2, :]  # ... H/s W/s C
            x3 = x[..., 1::2, 1::2, :]  # ... H/s W/s C
            x = torch.cat([x0, x1, x2, x3], -1)  # ... H/s W/s 4*C
        else:
            x = torch.cat([x, x, x, x], -1)  # H W 4*C

        x = self.norm(x)
        x = self.reduction(x)  # ... H/2 W/2 2*C
        return x

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(dim={self.dim})"
        return s


def shifted_window_attention(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    B, H, W, C = input.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(
        B,
        pad_H // window_size[0],
        window_size[0],
        pad_W // window_size[1],
        window_size[1],
        C,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(
        B * num_windows, window_size[0] * window_size[1], C
    )  # B*nW, Ws*Ws, C

    # multi-head attention
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(
        2, 0, 3, 1, 4
    )
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (C // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
        h_slices = (
            (0, -window_size[0]),
            (-window_size[0], -shift_size[0]),
            (-shift_size[0], None),
        )
        w_slices = (
            (0, -window_size[1]),
            (-window_size[1], -shift_size[1]),
            (-shift_size[1], None),
        )
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(
            pad_H // window_size[0],
            window_size[0],
            pad_W // window_size[1],
            window_size[1],
        )
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(
            num_windows, window_size[0] * window_size[1]
        )
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )
        attn = attn.view(
            x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1)
        )
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout)

    # reverse windows
    x = x.view(
        B,
        pad_H // window_size[0],
        pad_W // window_size[1],
        window_size[0],
        window_size[1],
        C,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x


class ShiftedWindowAttention(BaseModule):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).view(-1)  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.embed_dim = dim

    def __repr__(self) -> str:
        return "{}(embed_dim={}, window_size={}, shift_size={}, num_heads={}, dropout={}, attn_dropout={}, dropout={})".format(
            self.__class__.__name__,
            self.embed_dim,
            self.window_size,
            self.shift_size,
            self.num_heads,
            self.attention_dropout,
            self.dropout,
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """

        N = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index]  # type: ignore[index]
        relative_position_bias = relative_position_bias.view(N, N, -1)
        relative_position_bias = (
            relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
        )

        return shifted_window_attention(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
        )


class SwinTransformerBlock(BaseModule):
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
        self,
        opts,
        embed_dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Optional[str] = "layer_norm",
    ):
        super().__init__()

        attn_unit = ShiftedWindowAttention(
            embed_dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attn_dropout,
            dropout=dropout,
        )
        self.attn = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=norm_layer, num_features=embed_dim
            ),
            attn_unit,
            Dropout(p=dropout),
        )

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        ffn_latent_dim = int(embed_dim * mlp_ratio)
        act_name = build_activation_layer(opts, num_parameters=1)
        self.mlp = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=norm_layer, num_features=embed_dim
            ),
            LinearLayer(in_features=embed_dim, out_features=ffn_latent_dim, bias=True),
            act_name,
            Dropout(p=ffn_dropout),
            LinearLayer(in_features=ffn_latent_dim, out_features=embed_dim, bias=True),
            Dropout(p=dropout),
        )
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__class__.__name__
        self.act_fn_name = act_name.__class__.__name__
        self.norm_type = norm_layer

    def __repr__(self) -> str:
        return "{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, attn_fn={}, act_fn={}, norm_fn={})".format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_dim,
            self.std_dropout,
            self.ffn_dropout,
            self.attn_fn_name,
            self.act_fn_name,
            self.norm_type,
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = x + self.stochastic_depth(self.attn(x))
        x = x + self.stochastic_depth(self.mlp(x))
        return x
