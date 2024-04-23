#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from corenet.modeling.layers import ConvLayer2d, MultiHeadAttention, StochasticDepth
from corenet.modeling.layers.normalization.batch_norm import BatchNorm2d
from corenet.modeling.modules import BaseModule
from corenet.modeling.modules.mobileone_block import MobileOneBlock, RepLKBlock


def convolutional_stem(
    opts: argparse.Namespace, in_channels: int, out_channels: int
) -> nn.Sequential:
    """
    Build convolutional stem with MobileOne blocks.

    Args:
        opts: Command line arguments.
        in_channels: Number of input channels.
        out_channels: Number of output channels.

    Returns:
        nn.Sequential object with stem elements.
    """
    inference_mode = getattr(opts, "model.classification.fastvit.inference_mode")
    return nn.Sequential(
        MobileOneBlock(
            opts,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            opts,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            opts,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
    )


class PatchEmbed(BaseModule):
    """
    Convolutional Patch embedding layer.

    Args:
        opts: Command line arguments.
        patch_size: Patch size for embedding computation.
        stride: Stride for convolutional embedding layer.
        in_channels: Number of channels of input tensor.
        embed_dim: Number of embedding dimensions.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()

        inference_mode = getattr(opts, "model.classification.fastvit.inference_mode")
        block = list()
        block.append(
            RepLKBlock(
                opts,
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                groups=in_channels,
                small_kernel_size=3,
                inference_mode=inference_mode,
            )
        )
        block.append(
            MobileOneBlock(
                opts,
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            )
        )
        self.proj = nn.Sequential(*block)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor of shape :math:`(B, C, H//s, W//s)`,
            where `s` is the stride provide while instantiating the layer.
        """
        x = self.proj(x)
        return x


class RepMixer(BaseModule):
    """
    Reparameterizable token mixer

    For more details, please refer to our paper:
    `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization`

    Args:
        opts: Command line arguments.
        dim: Input feature map dimension. :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, H, W)`.
        kernel_size: Kernel size for spatial mixing. Default: 3
        use_layer_scale: If True, learnable layer scale is used. Default: ``True``
        layer_scale_init_value: Initial value for layer scale. Default: 1e-5
        inference_mode: If True, instantiates model in inference mode. Default: ``False``
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        dim: int,
        kernel_size: int = 3,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias=True,
            )
        else:
            self.norm = MobileOneBlock(
                opts,
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                opts,
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = nn.Parameter(
                    layer_scale_init_value * torch.ones((dim, 1, 1)),
                )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass implements inference logic for module
        before and after reparameterization.

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor of shape :math:`(B, C, H, W)`.
        """
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x

    def reparameterize(self) -> None:
        """
        Reparameterize mixer and norm into a single
        convolutional layer for efficient inference.
        """
        self.mixer.reparameterize()
        self.norm.reparameterize()

        if self.use_layer_scale:
            w = self.mixer.id_tensor + self.layer_scale.unsqueeze(-1) * (
                self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = torch.squeeze(self.layer_scale) * (
                self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = (
                self.mixer.id_tensor
                + self.mixer.reparam_conv.weight
                - self.norm.reparam_conv.weight
            )
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2d(
            in_channels=self.dim,
            out_channels=self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w
        self.reparam_conv.bias.data = b

        for para in self.parameters():
            para.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")


class ConvFFN(BaseModule):
    """
    Convolutional FFN Module.

    Args:
        opts: Command line arguments.
        in_channels: Number of input channels.
        hidden_channels: Number of channels after expansion. Default: None
        out_channels: Number of output channels. Default: None
        drop: Dropout rate. Default: ``0.0``.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = ConvLayer2d(
            opts,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=7,
            padding=3,
            groups=in_channels,
            bias=False,
            use_act=False,
        )
        self.fc1 = ConvLayer2d(
            opts, in_channels, hidden_channels, kernel_size=1, use_norm=False, bias=True
        )
        self.fc2 = ConvLayer2d(
            opts,
            hidden_channels,
            out_channels,
            kernel_size=1,
            use_norm=False,
            use_act=False,
            bias=True,
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor of shape :math:`(B, C, H, W)`.
        """
        x = self.conv(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepMixerBlock(BaseModule):
    """
    Implementation of Metaformer block with RepMixer as token mixer.
    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_

    Args:
        opts: Command line arguments.
        dim: Number of embedding dimensions.
        kernel_size: Kernel size for repmixer. Default: 3
        mlp_ratio: MLP expansion ratio. Default: 4.0
        drop: Dropout rate. Default: 0.0
        drop_path: Drop path rate. Default: 0.0
        use_layer_scale: Flag to turn on layer scale. Default: ``True``
        layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        inference_mode: Flag to instantiate block in inference mode. Default: ``False``
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ):

        super().__init__()
        self.token_mixer = RepMixer(
            opts,
            dim=dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            opts, in_channels=dim, hidden_channels=hidden_dim, drop=drop
        )

        self.drop_path = (
            StochasticDepth(drop_path, mode="row") if drop_path > 0.0 else nn.Identity()
        )
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)),
            )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor of shape :math:`(B, C, H, W)`.
        """
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x


class AttentionBlock(BaseModule):
    """
    Implementation of metaformer block with MHSA as token mixer.
    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_

    Args:
        opts: Command line arguments.
        dim: Number of embedding dimensions.
        mlp_ratio: MLP expansion ratio. Default: 4.0
        drop: Dropout rate. Default: 0.0
        drop_path: Drop path rate. Default: 0.0
        use_layer_scale: Flag to turn on layer scale. Default: ``True``
        layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        dim: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ):

        super().__init__()
        self.norm = BatchNorm2d(num_features=dim)
        self.head_dim = 32
        num_heads = dim // self.head_dim
        self.token_mixer = MultiHeadAttention(
            embed_dim=dim, num_heads=num_heads, bias=False
        )

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(
            mlp_ratio
        )
        hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            opts, in_channels=dim, hidden_channels=hidden_dim, drop=drop
        )

        self.drop_path = (
            StochasticDepth(drop_path, mode="row") if drop_path > 0.0 else nn.Identity()
        )
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)),
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim, 1, 1)),
            )

    def _apply_mhsa(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform appropriate reshaping before and after MHSA block.

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor of shape :math:`(B, C, H, W)`.
        """
        x_norm = self.norm(x)
        B, C, H, W = x_norm.shape
        x_norm_reshaped = torch.flatten(x_norm, start_dim=2).transpose(
            -2, -1
        )  # (B, N, C), where N = H * W
        out = self.token_mixer(x_norm_reshaped)
        out = out.transpose(-2, -1).reshape(B, C, H, W)
        return out

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor output from the attention block.
        """
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self._apply_mhsa(x))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self._apply_mhsa(x))
            x = x + self.drop_path(self.convffn(x))
        return x


class RepCPE(BaseModule):
    """
    Implementation of reparameterizable conditional positional encoding.
    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_

    Args:
        opts: Command line arguments.
        in_channels: Number of input channels.
        embed_dim: Number of embedding dimensions. Default: 768
        spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
        inference_mode: Flag to instantiate block in inference mode. Default: ``False``
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        in_channels: int,
        embed_dim: int = 768,
        spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
        inference_mode: bool = False,
    ):
        super(RepCPE, self).__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, Tuple), (
            f'"spatial_shape" must by a sequence or int, '
            f"get {type(spatial_shape)} instead."
        )
        assert len(spatial_shape) == 2, (
            f'Length of "spatial_shape" should be 2, '
            f"got {len(spatial_shape)} instead."
        )

        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias=True,
            )
        else:
            self.pe = ConvLayer2d(
                opts,
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=spatial_shape,
                stride=1,
                padding=int(spatial_shape[0] // 2),
                use_norm=False,
                use_act=False,
                bias=True,
                groups=embed_dim,
            )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass implements inference logic for module
        before and after reparameterization.

        Args:
            x: Input tensor of shape :math:`(B, C, H, W)`.

        Returns:
            torch.Tensor of shape :math:`(B, C, H, W)`.
        """
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            x = self.pe(x) + x
            return x

    def reparameterize(self) -> None:
        """Reparameterize linear branches."""
        # Build equivalent Id tensor
        input_dim = self.in_channels // self.groups
        kernel_value = torch.zeros(
            (
                self.in_channels,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ),
            dtype=self.pe.block.conv.weight.dtype,
            device=self.pe.block.conv.weight.device,
        )
        for i in range(self.in_channels):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = kernel_value

        # Reparameterize Id tensor and conv
        w_final = id_tensor + self.pe.block.conv.weight
        b_final = self.pe.block.conv.bias

        # Introduce reparam conv
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.embed_dim,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.embed_dim,
            bias=True,
        )
        self.reparam_conv.weight.data = w_final
        self.reparam_conv.bias.data = b_final

        for para in self.parameters():
            para.detach_()
        self.__delattr__("pe")
