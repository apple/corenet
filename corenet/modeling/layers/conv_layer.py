#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Optional, Tuple, Type, Union

from torch import Tensor, nn

from corenet.modeling.layers.activation import build_activation_layer
from corenet.modeling.layers.base_layer import BaseLayer
from corenet.modeling.layers.normalization.layer_norm import LayerNorm, LayerNorm2D_NCHW
from corenet.modeling.layers.normalization_layers import get_normalization_layer
from corenet.utils import logger


class Conv2d(nn.Conv2d):
    """
    Applies a 2D convolution over an input.

    Args:
        in_channels: :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`.
        out_channels: :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out})`.
        kernel_size: Kernel size for convolution.
        stride: Stride for convolution. Default: 1.
        padding: Padding for convolution. Default: 0.
        dilation: Dilation rate for convolution. Default: 1.
        groups: Number of groups in convolution. Default: 1.
        bias: Use bias. Default: ``False``.
        padding_mode: Padding mode ('zeros', 'reflect', 'replicate' or 'circular'). Default: ``zeros``.
        use_norm: Use normalization layer after convolution. Default: ``True``.
        use_act: Use activation layer after convolution (or convolution and normalization).
                                Default: ``True``.
        act_name: Use specific activation function. Overrides the one specified in command line args.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = 1,
        padding: Optional[Union[int, Tuple[int, int]]] = 0,
        dilation: Optional[Union[int, Tuple[int, int]]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        padding_mode: Optional[str] = "zeros",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )


class _BaseConvNormActLayer(BaseLayer):
    """
    Applies an N-dimensional convolution over an input.

    Args:
        opts: Command line options.
        in_channels: :math:`C_{out}` from an expected output of size
            :math:`(bs, C_{in}, X_{1}, ..., X_{N})`.
        out_channels: :math:`C_{out}` from an expected output of size
            :math:`(bs, C_{out}, Y_{1}, ..., Y_{N})`.
        kernel_size: Kernel size for convolution. An integer, or tuple of length ``N``.
        stride: Stride for convolution. An integer, or tuple of length ``N``. Default: 1.
        dilation: Dilation rate for convolution. An integer, or tuple of length ``N``.
            Default: ``1``.
        padding: Padding for convolution. An integer, or tuple of length ``N``.
            If not specified, padding is automatically computed based on kernel size and
            dilation range. Default : ``None`` (equivalent to ``[
            int((kernel_size[i] - 1) / 2) * dilation[i] for i in range(N)]``).
        groups: Number of groups in convolution. Default: ``1``.
        bias: Use bias. Default: ``False``.
        padding_mode: Padding mode ('zeros', 'reflect', 'replicate' or 'circular').
            Default: ``zeros``.
        use_norm: Use normalization layer after convolution. Default: ``True``.
        use_act: Use activation layer after convolution (or convolution and normalization).
            Default: ``True``.
        norm_layer: If not None, the provided normalization layer object will be used.
            Otherwise, a normalization object will be created based on config
            ``model.normalization.*`` opts.
        act_layer: If not None, the provided activation function will be used.
            Otherwise, an activation function will be created based on config
            ``model.activation.*`` opts.

    Shape:
        - Input: :math:`(bs, C_{in}, X_{1}, ..., X_{N})`.
        - Output: :math:`(bs, C_{out}, Y_{1}, ..., Y_{N})`.

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    @property
    def ndim(self) -> int:
        raise NotImplementedError("subclasses should override ndim property")

    @property
    def module_cls(self) -> Type[nn.Module]:
        raise NotImplementedError("subclasses should override module_cls property")

    def __init__(
        self,
        opts: argparse.Namespace,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...]]] = None,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        use_norm: bool = True,
        use_act: bool = True,
        norm_layer: Optional[nn.Module] = None,
        act_layer: Optional[nn.Module] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        if norm_layer is None and use_norm:
            norm_type = getattr(opts, "model.normalization.name")
            if norm_type == "batch_norm":
                norm_type = f"batch_norm_{self.ndim}d"
            norm_layer = get_normalization_layer(
                opts=opts, num_features=out_channels, norm_type=norm_type
            )
        elif norm_layer is not None and not use_norm:
            logger.error(
                f"When use_norm is False, norm_layer should be None, but norm_layer={norm_layer} is provided."
            )

        if act_layer is None and use_act:
            act_layer = build_activation_layer(opts, num_parameters=out_channels)
        elif act_layer is not None and not use_act:
            logger.error(
                f"When use_act is False, act_layer should be None, but act_layer={act_layer} is provided."
            )

        if (
            use_norm
            and any(param[0] == "bias" for param in norm_layer.named_parameters())
            and bias
        ):
            assert (
                not bias
            ), "Do not use bias when using normalization layers with bias."

        if use_norm and isinstance(norm_layer, (LayerNorm, LayerNorm2D_NCHW)):
            bias = True

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * self.ndim

        if isinstance(stride, int):
            stride = (stride,) * self.ndim

        if isinstance(dilation, int):
            dilation = (dilation,) * self.ndim

        assert isinstance(kernel_size, Tuple)
        assert isinstance(stride, Tuple)
        assert isinstance(dilation, Tuple)

        if padding is None:
            padding = (
                int((kernel_size[i] - 1) / 2) * dilation[i] for i in range(self.ndim)
            )

        if in_channels % groups != 0:
            logger.error(
                "Input channels are not divisible by groups. {}%{} != 0 ".format(
                    in_channels, groups
                )
            )
        if out_channels % groups != 0:
            logger.error(
                "Output channels are not divisible by groups. {}%{} != 0 ".format(
                    out_channels, groups
                )
            )

        block = nn.Sequential()

        conv_layer = self.module_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,  # type: ignore
            stride=stride,  # type: ignore
            padding=padding,
            dilation=dilation,  # type: ignore
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        if use_act:
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias
        self.dilation = dilation

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        if cls != _BaseConvNormActLayer:
            return parser
        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--model.layer.conv-init",
            type=str,
            default="kaiming_normal",
            help="Init type for conv layers",
        )
        parser.add_argument(
            "--model.layer.conv-init-std-dev",
            type=float,
            default=None,
            help="Std deviation for conv layers",
        )
        return parser

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def __repr__(self):
        repr_str = self.block[0].__repr__()
        repr_str = repr_str[:-1]

        if self.norm_name is not None:
            repr_str += ", normalization={}".format(self.norm_name)

        if self.act_name is not None:
            repr_str += ", activation={}".format(self.act_name)
        repr_str += ")"
        return repr_str


class ConvLayer1d(_BaseConvNormActLayer):
    ndim = 1
    module_cls = nn.Conv1d


class ConvLayer2d(_BaseConvNormActLayer):
    ndim = 2
    module_cls = Conv2d


class ConvLayer3d(_BaseConvNormActLayer):
    ndim = 3
    module_cls = nn.Conv3d


class TransposeConvLayer2d(BaseLayer):
    """
    Applies a 2D Transpose convolution (aka as Deconvolution) over an input.

    Args:
        opts: Command line arguments.
        in_channels: :math:`C_{in}` from an expected input of size
          :math:`(N, C_{in}, H_{in}, W_{in})`.
        out_channels: :math:`C_{out}` from an expected output of size
          :math:`(N, C_{out}, H_{out}, W_{out})`.
        kernel_size: Kernel size for convolution.
        stride: Stride for convolution. Default: 1.
        dilation: Dilation rate for convolution. Default: 1.
        groups: Number of groups in convolution. Default: 1.
        bias: Use bias. Default: ``False``.
        padding_mode: Padding mode. Default: ``zeros``.
        use_norm: Use normalization layer after convolution. Default: ``True``.
        use_act: Use activation layer after convolution (or convolution and normalization).
        Default: ``True``.
        norm_layer: If not None, the provided normalization layer object will be used.
            Otherwise, a normalization object will be created based on config
            ``model.normalization.*`` opts.
        padding: Padding will be done on both sides of each dimension in the input.
        output_padding: Additional padding on the output tensor.
        auto_padding: Compute padding automatically. Default: ``True``.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Optional[Union[int, Tuple]] = 1,
        dilation: Optional[Union[int, Tuple]] = 1,
        groups: Optional[int] = 1,
        bias: Optional[bool] = False,
        padding_mode: Optional[str] = "zeros",
        use_norm: Optional[bool] = True,
        use_act: Optional[bool] = True,
        norm_layer: Optional[nn.Module] = None,
        padding: Optional[Union[int, Tuple]] = (0, 0),
        output_padding: Optional[Union[int, Tuple]] = None,
        auto_padding: Optional[bool] = True,
    ) -> None:
        super().__init__()

        if norm_layer is None and use_norm:
            norm_type = getattr(opts, "model.normalization.name")
            norm_layer = get_normalization_layer(
                opts=opts, num_features=out_channels, norm_type=norm_type
            )
        elif norm_layer is not None and not use_norm:
            logger.error(
                f"When use_norm is False, norm_layer should be None, but norm_layer={norm_layer} is provided."
            )

        if (
            use_norm
            and any(param[0] == "bias" for param in norm_layer.named_parameters())
            and bias
        ):
            assert (
                not bias
            ), "Do not use bias when using normalization layers with bias."

        if use_norm and isinstance(norm_layer, (LayerNorm, LayerNorm2D_NCHW)):
            bias = True

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        if isinstance(stride, int):
            stride = (stride, stride)

        if isinstance(dilation, int):
            dilation = (dilation, dilation)

        if output_padding is None:
            output_padding = (stride[0] - 1, stride[1] - 1)

        assert isinstance(kernel_size, (tuple, list))
        assert isinstance(stride, (tuple, list))
        assert isinstance(dilation, (tuple, list))

        if auto_padding:
            padding = (
                int((kernel_size[0] - 1) / 2) * dilation[0],
                int((kernel_size[1] - 1) / 2) * dilation[1],
            )

        if in_channels % groups != 0:
            logger.error(
                "Input channels are not divisible by groups. {}%{} != 0 ".format(
                    in_channels, groups
                )
            )
        if out_channels % groups != 0:
            logger.error(
                "Output channels are not divisible by groups. {}%{} != 0 ".format(
                    out_channels, groups
                )
            )

        block = nn.Sequential()
        conv_layer = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            output_padding=output_padding,
        )

        block.add_module(name="conv", module=conv_layer)

        self.norm_name = None
        if use_norm:
            block.add_module(name="norm", module=norm_layer)
            self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        act_type = getattr(opts, "model.activation.name", "relu")

        if act_type is not None and use_act:
            act_layer = build_activation_layer(
                opts,
                act_type=act_type,
                num_parameters=out_channels,
            )
            block.add_module(name="act", module=act_layer)
            self.act_name = act_layer.__class__.__name__

        self.block = block

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.kernel_size = conv_layer.kernel_size
        self.bias = bias

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def __repr__(self):
        repr_str = self.block[0].__repr__()
        repr_str = repr_str[:-1]

        if self.norm_name is not None:
            repr_str += ", normalization={}".format(self.norm_name)

        if self.act_name is not None:
            repr_str += ", activation={}".format(self.act_name)
        repr_str += ")"
        return repr_str


class NormActLayer(BaseLayer):
    """
    Applies a normalization layer followed by an activation layer.

    Args:
        opts: Command-line arguments.
        num_features: :math:`C` from an expected input of size :math:`(N, C, H, W)`.

    Shape:
        - Input: :math:`(N, C, H, W)`.
        - Output: :math:`(N, C, H, W)`.
    """

    def __init__(self, opts, num_features, *args, **kwargs):
        super().__init__()

        block = nn.Sequential()

        self.norm_name = None
        norm_layer = get_normalization_layer(opts=opts, num_features=num_features)
        block.add_module(name="norm", module=norm_layer)
        self.norm_name = norm_layer.__class__.__name__

        self.act_name = None
        act_layer = build_activation_layer(
            opts,
            num_parameters=num_features,
        )
        block.add_module(name="act", module=act_layer)
        self.act_name = act_layer.__class__.__name__

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def __repr__(self):
        repr_str = "{}(normalization={}, activation={})".format(
            self.__class__.__name__, self.norm_type, self.act_type
        )
        return repr_str


class _BaseSeparableConv(BaseLayer):
    """
    Applies an N-dimensional depth-wise separable convolution
        <https://arxiv.org/abs/1610.02357> over an N-dimensional input tensor.

    Args:
        opts: Command line arguments.
        in_channels: :math:`C_{in}` from an expected input of size
            :math:`(N, C_{in}, X_{1}, ..., X_{N})`.
        out_channels: :math:`C_{out}` from an expected output of size
            :math:`(N, C_{out}, Y_{1}, ..., Y_{N})`.
        kernel_size: Kernel size for convolution.
        stride: Stride for convolution. Default: 1.
        dilation: Dilation rate for convolution. Default: 1.
        use_norm: Use normalization layer after convolution. Default: ``True``.
        use_act: Use activation layer after convolution (or convolution and normalization).
            Default: ``True``.
        use_act_deptwise: Use activation layer after depthwise convolution (or
            convolution and normalization). Default: ``False``.
            NOTE: We recommend against using activation function in depth-wise convolution.
        bias: Use bias. Default: ``False``.
        padding_mode: Padding mode ('zeros', 'reflect', 'replicate' or 'circular').
            Default: ``zeros``.
        act_name: Use specific activation function. Overrides the one specified in
            command line args. Default: ``None``.

    Shape:
        - Input: :math:`(N, C_{in}, X_{1}, ..., X_{N})`.
        - Output: :math:`(N, C_{out}, Y_{1}, ..., Y_{N})`.

    .. note::
        For depth-wise convolution, `groups=C_{in}=C_{out}`.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        use_norm: bool = True,
        use_act: bool = True,
        use_act_depthwise: bool = False,
        bias: bool = False,
        padding_mode: str = "zeros",
        act_name: Optional[str] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dw_conv = self.conv_layer_cls(
            opts=opts,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=in_channels,
            bias=False,
            padding_mode=padding_mode,
            use_norm=True,
            # NOTE: We recommend against using activation function in depth-wise convolution.
            use_act=use_act_depthwise,
            act_name=act_name,
        )
        self.pw_conv = self.conv_layer_cls(
            opts=opts,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            groups=1,
            bias=bias,
            padding_mode=padding_mode,
            use_norm=use_norm,
            use_act=use_act,
            act_name=act_name,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.dilation = dilation

    @property
    def conv_layer_cls(self):
        raise NotImplementedError("Subclasses should override conv_layer_cls.")

    def __repr__(self):
        repr_str = "{}(in_channels={}, out_channels={}, kernel_size={}, stride={}, dilation={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.dilation,
        )
        return repr_str

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class SeparableConv1d(_BaseSeparableConv):
    conv_layer_cls = ConvLayer1d


class SeparableConv2d(_BaseSeparableConv):
    conv_layer_cls = ConvLayer2d


class SeparableConv3d(_BaseSeparableConv):
    conv_layer_cls = ConvLayer3d
