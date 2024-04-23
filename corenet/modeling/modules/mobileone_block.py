#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Tuple, Union

import torch
import torch.nn as nn

from corenet.modeling.layers import ConvLayer2d, Identity
from corenet.modeling.layers.activation import build_activation_layer
from corenet.modeling.layers.normalization.batch_norm import BatchNorm2d
from corenet.modeling.modules import BaseModule, SqueezeExcitation


class MobileOneBlock(BaseModule):
    """
    MobileOne building block.

    For more details, please refer to our paper:
    `An Improved One millisecond Mobile Backbone <https://arxiv.org/pdf/2206.04040.pdf>`
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
    ) -> None:
        """
        Construct a MobileOneBlock.

        Args:
            opts: Command line arguments.
            in_channels: Number of channels in the input.
            out_channels: Number of channels produced by the block.
            kernel_size: Size of the convolution kernel.
            stride: Stride size. Default: 1
            padding: Zero-padding size. Default: 0
            dilation: Kernel dilation factor. Default: 1
            groups: Group number. Default: 1
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
            use_se: Whether to use SE-ReLU activations. Default: ``False``
            use_act: Whether to use activation. Default: ``True``
            use_scale_branch: Whether to use scale branch. Default: ``True``
            num_conv_branches: Number of linear conv branches. Default: 1
        """
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SqueezeExcitation(opts, out_channels, squeeze_factor=16)
        else:
            self.se = Identity()

        # Activation
        if use_act:
            self.activation = build_activation_layer(opts)
        else:
            self.activation = Identity()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
            )
        else:
            # Re-parameterizable skip connection
            self.rbr_skip = (
                BatchNorm2d(num_features=in_channels, affine=True)
                if out_channels == in_channels and stride == 1
                else None
            )

            # Re-parameterizable conv branches
            if num_conv_branches > 0:
                rbr_conv = list()
                for _ in range(self.num_conv_branches):
                    rbr_conv.append(
                        ConvLayer2d(
                            opts,
                            in_channels=self.in_channels,
                            out_channels=self.out_channels,
                            kernel_size=kernel_size,
                            stride=self.stride,
                            padding=padding,
                            groups=self.groups,
                            bias=False,
                            use_act=False,
                        )
                    )
                self.rbr_conv = nn.ModuleList(rbr_conv)
            else:
                self.rbr_conv = None

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if (kernel_size > 1) and use_scale_branch:
                self.rbr_scale = ConvLayer2d(
                    opts,
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=self.stride,
                    padding=0,
                    groups=self.groups,
                    bias=False,
                    use_act=False,
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
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self) -> None:
        """
        Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()

        if hasattr(self, "rbr_conv"):
            self.__delattr__("rbr_conv")
        if hasattr(self, "rbr_scale"):
            self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_branch_ops(self.rbr_scale.block)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale, [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_branch_ops(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_branch_ops(self.rbr_conv[ix].block)
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_branch_ops(
        self, branch: Union[nn.Sequential, nn.BatchNorm2d]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to fuse all linear ops in a branch.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        Args:
            branch: Sequence of ops to be fused.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            return self._fuse_conv_bn(kernel, branch.norm)
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            return self._fuse_conv_bn(kernel, branch)

    @staticmethod
    def _fuse_conv_bn(
        kernel: torch.Tensor, bn: nn.BatchNorm2d
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to fuse batchnorm layer with conv layer.

        Args:
            kernel: Convolutional kernel weights.
            bn: Batchnorm 2d layer.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        assert bn.affine, (
            "Expected BatchNorm layer to have affine parameters "
            "instead got BatchNorm layer without affine parameters."
        )
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class RepLKBlock(BaseModule):
    """
    This class defines overparameterized large kernel conv block in `RepLKNet <https://arxiv.org/abs/2203.06717>`_
    Reference: https://github.com/DingXiaoH/RepLKNet-pytorch

    Args:
        opts: Command-line arguments.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Kernel size of the large kernel conv branch.
        stride: Stride size. Default: 1
        dilation: Kernel dilation factor. Default: 1
        groups: Group number. Default: 1
        small_kernel_size: Kernel size of small kernel conv branch.
        inference_mode: If True, instantiates model in inference mode. Default: ``False``
        use_act: If True, activation is used. Default: ``True``
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        small_kernel_size: int = None,
        inference_mode: bool = False,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Activation
        if use_act:
            self.activation = build_activation_layer(opts)
        else:
            self.activation = Identity()

        self.kernel_size = kernel_size
        self.small_kernel_size = small_kernel_size
        self.padding = kernel_size // 2
        if inference_mode:
            self.lkb_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = ConvLayer2d(
                opts,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
                bias=False,
                use_act=False,
            )
            if small_kernel_size is not None:
                assert (
                    small_kernel_size <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel"
                self.small_conv = ConvLayer2d(
                    opts,
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.small_kernel_size,
                    stride=self.stride,
                    padding=self.small_kernel_size // 2,
                    groups=self.groups,
                    bias=False,
                    use_act=False,
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
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, "small_conv"):
                out += self.small_conv(x)

        self.activation(out)
        return out

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepLKNet-pytorch

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        lk_kernel, lk_bias = MobileOneBlock._fuse_conv_bn(
            self.lkb_origin.block.conv.weight, self.lkb_origin.block.norm
        )
        if hasattr(self, "small_conv"):
            sk_kernel, sk_bias = MobileOneBlock._fuse_conv_bn(
                self.small_conv.block.conv.weight, self.small_conv.block.norm
            )
            lk_bias += sk_bias
            #   add to the central part
            lk_kernel += nn.functional.pad(
                sk_kernel, [(self.kernel_size - self.small_kernel_size) // 2] * 4
            )
        return lk_kernel, lk_bias

    def reparameterize(self) -> None:
        """
        Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        kernel, bias = self._get_kernel_bias()
        self.lkb_reparam = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=True,
        )
        self.lkb_reparam.weight.data = kernel
        self.lkb_reparam.bias.data = bias
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")
