#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torchvision.ops.roi_align import RoIAlign

from corenet.modeling.layers import ConvLayer2d, SeparableConv2d, TransposeConvLayer2d
from corenet.modeling.misc.init_utils import initialize_conv_layer
from corenet.modeling.modules import BaseModule


class SSDHead(BaseModule):
    """
    This class defines the `SSD object detection Head <https://arxiv.org/abs/1512.02325>`_

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        n_anchors (int): Number of anchors
        n_classes (int): Number of classes in the dataset
        n_coordinates (Optional[int]): Number of coordinates. Default: 4 (x, y, w, h)
        proj_channels (Optional[int]): Number of projected channels. If `-1`, then projection layer is not used
        kernel_size (Optional[int]): Kernel size in convolutional layer. If kernel_size=1, then standard
            point-wise convolution is used. Otherwise, separable convolution is used
        stride (Optional[int]): stride for feature map. If stride > 1, then feature map is sampled at this rate
            and predictions are made on fewer pixels as compared to the input tensor. Default: 1
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        n_anchors: int,
        n_classes: int,
        n_coordinates: Optional[int] = 4,
        proj_channels: Optional[int] = -1,
        kernel_size: Optional[int] = 3,
        stride: Optional[int] = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        proj_layer = None
        self.proj_channels = None
        if proj_channels != -1 and proj_channels != in_channels and kernel_size > 1:
            proj_layer = ConvLayer2d(
                opts=opts,
                in_channels=in_channels,
                out_channels=proj_channels,
                kernel_size=1,
                stride=1,
                groups=1,
                bias=False,
                use_norm=True,
                use_act=True,
            )
            in_channels = proj_channels
            self.proj_channels = proj_channels

        self.proj_layer = proj_layer

        conv_fn = ConvLayer2d if kernel_size == 1 else SeparableConv2d
        if kernel_size > 1 and stride > 1:
            kernel_size = max(kernel_size, stride if stride % 2 != 0 else stride + 1)
        self.loc_cls_layer = conv_fn(
            opts=opts,
            in_channels=in_channels,
            out_channels=n_anchors * (n_coordinates + n_classes),
            kernel_size=kernel_size,
            stride=1,
            groups=1,
            bias=True,
            use_norm=False,
            use_act=False,
        )

        self.n_coordinates = n_coordinates
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.k_size = kernel_size
        self.stride = stride
        self.in_channel = in_channels

        self.reset_parameters()

    def __repr__(self) -> str:
        repr_str = "{}(in_channels={}, n_anchors={}, n_classes={}, n_coordinates={}, kernel_size={}, stride={}".format(
            self.__class__.__name__,
            self.in_channel,
            self.n_anchors,
            self.n_classes,
            self.n_coordinates,
            self.k_size,
            self.stride,
        )
        if self.proj_layer is not None:
            repr_str += ", proj=True, proj_channels={}".format(self.proj_channels)

        repr_str += ")"
        return repr_str

    def reset_parameters(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                initialize_conv_layer(module=layer, init_method="xavier_uniform")

    def _sample_fm(self, x: Tensor) -> Tensor:
        height, width = x.shape[-2:]
        device = x.device
        start_step = max(0, self.stride // 2)
        indices_h = torch.arange(
            start=start_step,
            end=height,
            step=self.stride,
            dtype=torch.int64,
            device=device,
        )
        indices_w = torch.arange(
            start=start_step,
            end=width,
            step=self.stride,
            dtype=torch.int64,
            device=device,
        )

        x_sampled = torch.index_select(x, dim=-1, index=indices_w)
        x_sampled = torch.index_select(x_sampled, dim=-2, index=indices_h)
        return x_sampled

    def forward(self, x: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        batch_size = x.shape[0]

        if self.proj_layer is not None:
            x = self.proj_layer(x)

        # [B x C x H x W] --> [B x Anchors * (coordinates + classes) x H x W]
        x = self.loc_cls_layer(x)

        if self.stride > 1:
            x = self._sample_fm(x)

        # [B x Anchors * (coordinates + classes) x H x W] --> [B x H x W x Anchors * (coordinates + classes)]
        x = x.permute(0, 2, 3, 1)
        # [B x H x W x Anchors * (coordinates + classes)] --> [B x H*W*Anchors X (coordinates + classes)]
        x = x.contiguous().view(batch_size, -1, self.n_coordinates + self.n_classes)

        # [B x H*W*Anchors X (coordinates + classes)] --> [B x H*W*Anchors X coordinates], [B x H*W*Anchors X classes]
        box_locations, box_classes = torch.split(
            x, [self.n_coordinates, self.n_classes], dim=-1
        )
        return box_locations, box_classes


class SSDInstanceHead(BaseModule):
    """
    Instance segmentation head for SSD model.
    """

    def __init__(
        self,
        opts,
        in_channels: int,
        n_classes: Optional[int] = 1,
        inner_dim: Optional[int] = 256,
        output_stride: Optional[int] = 1,
        output_size: Optional[int] = 8,
        *args,
        **kwargs
    ) -> None:
        """

        Args:
            opts: command-line arguments
            in_channels (int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
            n_classes (Optional[int]): Number of classes. Default: 1
            inner_dim: (Optional[int]): Inner dimension of the instance head. Default: 256
            output_stride (Optional[int]): Output stride of the feature map. Output stride is the ratio of input to
                the feature map size. Default: 1
            output_size (Optional[int]): Output size of the instances extracted from RoIAlign layer. Default: 8
        """
        super().__init__()
        self.roi_align = RoIAlign(
            output_size=output_size,
            spatial_scale=1.0 / output_stride,
            sampling_ratio=2,
            aligned=True,
        )

        self.seg_head = nn.Sequential(
            TransposeConvLayer2d(
                opts=opts,
                in_channels=in_channels,
                out_channels=inner_dim,
                kernel_size=2,
                stride=2,
                bias=True,
                use_norm=False,
                use_act=True,
                auto_padding=False,
                padding=0,
                output_padding=0,
            ),
            ConvLayer2d(
                opts=opts,
                in_channels=inner_dim,
                out_channels=n_classes,
                kernel_size=1,
                stride=1,
                use_norm=False,
                use_act=False,
                bias=True,
            ),
        )
        self.inner_channels = inner_dim
        self.in_channels = in_channels
        self.mask_classes = n_classes
        self.reset_parameters()

    def __repr__(self) -> str:
        return "{}(in_channels={}, up_out_channels={}, n_classes={})".format(
            self.__class__.__name__,
            self.in_channels,
            self.inner_channels,
            self.mask_classes,
        )

    def reset_parameters(self) -> None:
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                initialize_conv_layer(module=layer, init_method="kaiming_normal")

    def forward(self, x: Tensor, boxes: Tensor, *args, **kwargs) -> Tensor:
        rois = self.roi_align(x, boxes)
        rois = self.seg_head(rois)
        return rois
