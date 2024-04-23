#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import List, Tuple

import torch
from torch import Tensor, nn

from corenet.modeling.layers import (
    ConvLayer2d,
    LinearLayer,
    TransposeConvLayer2d,
    get_normalization_layer,
)
from corenet.modeling.misc.init_utils import initialize_conv_layer, initialize_fc_layer

# Below classes are adapted from Torchvision version=0.12 to make the code compatible with previous torch versions.


class FastRCNNConvFCHead(nn.Sequential):
    def __init__(
        self,
        opts,
        input_size: Tuple[int, int, int],
        conv_layers: List[int],
        fc_layers: List[int],
        *args,
        **kwargs,
    ):
        """
        Args:
            input_size (Tuple[int, int, int]): the input size in CHW format.
            conv_layers (list): feature dimensions of each Convolution layer
            fc_layers (list): feature dimensions of each FCN layer
        """
        in_channels, in_height, in_width = input_size

        blocks = []
        previous_channels = in_channels
        for current_channels in conv_layers:
            blocks.extend(
                [
                    ConvLayer2d(
                        opts,
                        in_channels=previous_channels,
                        out_channels=current_channels,
                        kernel_size=3,
                        stride=1,
                        use_norm=False,
                        use_act=False,
                    ),
                    replace_syncbn_with_syncbnfp32(opts, num_features=current_channels),
                    nn.ReLU(inplace=False),
                ]
            )
            previous_channels = current_channels
        blocks.append(nn.Flatten())
        previous_channels = previous_channels * in_height * in_width

        for current_channels in fc_layers:
            blocks.append(LinearLayer(previous_channels, current_channels, bias=True))
            blocks.append(nn.ReLU(inplace=True))
            previous_channels = current_channels

        super().__init__(*blocks)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                initialize_conv_layer(module=layer, init_method="kaiming_normal")
            elif isinstance(layer, LinearLayer):
                initialize_fc_layer(module=layer, init_method="kaiming_uniform")


class RPNHead(nn.Module):
    """
    Adds a simple RPN Head with classification and regression heads

    Args:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        conv_depth (int, optional): number of convolutions
    """

    def __init__(self, opts, in_channels: int, num_anchors: int, conv_depth=1) -> None:
        super().__init__()
        convs = []
        for _ in range(conv_depth):
            convs.extend(
                [
                    ConvLayer2d(
                        opts,
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=1,
                        use_norm=False,
                        use_act=False,
                        bias=False,
                    ),
                    replace_syncbn_with_syncbnfp32(opts, num_features=in_channels),
                    nn.ReLU(inplace=False),
                ]
            )
        self.conv = nn.Sequential(*convs)
        self.cls_logits = ConvLayer2d(
            opts,
            in_channels=in_channels,
            out_channels=num_anchors,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
            bias=True,
        )
        self.bbox_pred = ConvLayer2d(
            opts,
            in_channels=in_channels,
            out_channels=num_anchors * 4,
            kernel_size=1,
            stride=1,
            use_act=False,
            use_norm=False,
            bias=True,
        )

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                initialize_conv_layer(module=layer, init_method="normal", std_val=0.01)

    def forward(self, x: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits = []
        bbox_reg = []
        for feature in x:
            t = self.conv(feature)
            logits.append(self.cls_logits(t))
            bbox_reg.append(self.bbox_pred(t))
        return logits, bbox_reg


class MaskRCNNHeads(nn.Sequential):
    def __init__(self, opts, in_channels: int, layers: List, dilation: int):
        """
        Args:
            in_channels (int): number of input channels
            layers (list): feature dimensions of each FCN layer
            dilation (int): dilation rate of kernel
            norm_layer (callable, optional): Module specifying the normalization layer to use. Default: None
        """
        blocks = []
        next_feature = in_channels
        for layer_features in layers:
            blocks.extend(
                [
                    ConvLayer2d(
                        opts=opts,
                        in_channels=next_feature,
                        out_channels=layer_features,
                        kernel_size=3,
                        stride=1,
                        dilation=dilation,
                        use_norm=False,
                        use_act=False,
                        bias=False,
                    ),
                    replace_syncbn_with_syncbnfp32(
                        opts=opts, num_features=layer_features
                    ),
                    nn.ReLU(inplace=False),
                ]
            )
            next_feature = layer_features

        super().__init__(*blocks)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                initialize_conv_layer(module=layer, init_method="kaiming_normal")


class MaskRCNNPredictor(nn.Sequential):
    def __init__(
        self, opts, in_channels: int, dim_reduced: int, num_classes: int
    ) -> None:
        super().__init__(
            *[
                TransposeConvLayer2d(
                    opts,
                    in_channels=in_channels,
                    out_channels=dim_reduced,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    output_padding=0,
                    use_norm=False,
                    use_act=False,
                    bias=False,
                    groups=1,
                ),
                replace_syncbn_with_syncbnfp32(opts, num_features=dim_reduced),
                nn.ReLU(inplace=False),
                ConvLayer2d(
                    opts,
                    in_channels=dim_reduced,
                    out_channels=num_classes,
                    kernel_size=1,
                    stride=1,
                    bias=True,
                    use_norm=False,
                    use_act=False,
                ),
            ]
        )

        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                initialize_conv_layer(module=layer, init_method="kaiming_normal")


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.cls_score = LinearLayer(in_channels, num_classes, bias=True)
        self.bbox_pred = LinearLayer(in_channels, num_classes * 4, bias=True)

        for layer in self.modules():
            if isinstance(layer, LinearLayer):
                initialize_fc_layer(module=layer, init_method="kaiming_uniform")

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, bbox_deltas


def replace_syncbn_with_syncbnfp32(opts, num_features: int) -> nn.Module:
    # Sync-BN with 0 batch size does not work well with AMP. To avoid that,
    # we replace all sync_bn in mask rcnn head with FP32 ones.
    norm_layer = getattr(opts, "model.normalization.name", None)

    if norm_layer.find("sync") > -1:
        return get_normalization_layer(
            opts, num_features=num_features, norm_type="sync_batch_norm_fp32"
        )
    else:
        return get_normalization_layer(opts=opts, num_features=num_features)
