#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from corenet.modeling.layers import ConvLayer2d
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.segmentation.heads.base_seg_head import BaseSegHead
from corenet.modeling.modules import ASPP
from corenet.options.parse_args import JsonValidator
from corenet.utils import logger


@MODEL_REGISTRY.register(name="deeplabv3", type="segmentation_head")
class DeeplabV3(BaseSegHead):
    """
    `DeepLabv3 <https://arxiv.org/abs/1706.05587>`_ segmentation head.

    Args:
        opts: Command-line arguments.
        enc_conf: Image encoder's input-output configuration at each spatial level.
        use_l5_exp: Deeplabv3 segmentation head uses features from level5 of the image encoder. However,
            some of the models (e.g., MobileNetv3) have expansion layers at level5 of the image encoder.
            Features from such layers can be used by setting 'use_l5_exp=True'.
        aspp_in_channels: The number of channels in the input to the ASPP module. The default
            behavior is None, i.e., determine automatically from image encoder's configuration.

    ...note:
        Image encoder's configuration is a mapping of the form
        {
            "conv_1": {"in": in_features_1, "out": out_features_1},
            "layer1": {"in": out_features_1, "out": out_features_2},
            "layer2": {"in": out_features_2, "out": out_features_3},
            "layer3": {"in": out_features_3, "out": out_features_4},
            "layer4": {"in": out_features_4, "out": out_features_5},
            "layer5": {"in": out_features_5, "out": out_features_6},
            "exp_before_cls": {"in": out_features_6, "out": out_features_7},
        }

        When 'use_l5_exp' is enabled, then output of expansion layer before classification (i.e., exp_before_cls)
        is used. Otherwise, the output of 'layer5' is used.
    """

    def __init__(
        self,
        opts,
        enc_conf: Dict,
        use_l5_exp: bool = False,
        aspp_in_channels: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        atrous_rates = getattr(opts, "model.segmentation.deeplabv3.aspp_rates")
        out_channels = getattr(opts, "model.segmentation.deeplabv3.aspp_out_channels")
        is_sep_conv = getattr(opts, "model.segmentation.deeplabv3.aspp_sep_conv")
        dropout = getattr(opts, "model.segmentation.deeplabv3.aspp_dropout")

        super().__init__(opts=opts, enc_conf=enc_conf, use_l5_exp=use_l5_exp)

        self.aspp = nn.Sequential()
        if aspp_in_channels is None:
            aspp_in_channels = (
                self.enc_l5_channels
                if not self.use_l5_exp
                else self.enc_l5_exp_channels
            )
        self.aspp.add_module(
            name="aspp_layer",
            module=ASPP(
                opts=opts,
                in_channels=aspp_in_channels,
                out_channels=out_channels,
                atrous_rates=atrous_rates,
                is_sep_conv=is_sep_conv,
                dropout=dropout,
            ),
        )

        self.classifier = ConvLayer2d(
            opts=opts,
            in_channels=out_channels,
            out_channels=self.n_seg_classes,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
            bias=True,
        )
        self.encoder_level5_output_key = "out_l5_exp" if self.use_l5_exp else "out_l5"

        self.reset_head_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """DeepLabv3 specific arguments"""
        if cls == DeeplabV3:
            group = parser.add_argument_group(title=cls.__name__)
            group.add_argument(
                "--model.segmentation.deeplabv3.aspp-rates",
                type=JsonValidator(Tuple[int, int, int]),
                default=(6, 12, 18),
                help=f"Atrous rates to be used in the ASPP module in {cls.__name__} segmentation head. \
                    Defaults to (6, 12, 18).",
            )
            group.add_argument(
                "--model.segmentation.deeplabv3.aspp-out-channels",
                type=int,
                default=256,
                help=f"Output channels of ASPP module in {cls.__name__} segmentation head. \
                    Defaults to 256.",
            )
            group.add_argument(
                "--model.segmentation.deeplabv3.aspp-sep-conv",
                action="store_true",
                default=False,
                help=f"Use separable convolution in the ASPP module in {cls.__name__} segmentation head. \
                    Defaults to False.",
            )
            group.add_argument(
                "--model.segmentation.deeplabv3.aspp-dropout",
                type=float,
                default=0.1,
                help=f"Dropout value in the ASPP module in {cls.__name__} segmentation head. \
                    Defaults to 0.1.",
            )
        return parser

    def forward_seg_head(self, enc_out: Dict) -> Tensor:
        """Forward method for DeeplabV3 segmentation head.

        Args:
            enc_out: A dictionary containing the feature maps from different spatial levels in the image encoder.

        Returns:
            A 4D tensor whose spatial size is 1/output of the input image.

        ...note:
            The input to the ASPP module is a low-resolution feature map from image encoder's level 5, optionally
            with adjusted strides and dilation rates to (1) increase the effective receptive field, and (2) adjust
            the feature map size to produce high-quality segmentation outputs.
        """
        x = enc_out[self.encoder_level5_output_key]
        x = self.aspp(x)
        x = self.classifier(x)
        return x


@MODEL_REGISTRY.register(name="msc_deeplabv3", type="segmentation_head")
class MultiScaleDeeplabV3(DeeplabV3):
    """Multi-scale DeepLabv3.

    The feature maps with different spatial levels are brought to the same spatial size using
    pixel shuffle and unshuffle operations. They are then concatenated and projected
    using a point-wise convolution.

    Note that CNN-based models typically down-sample the input by a factor of 32, but output stride of
    level 4 and level 5 are adjusted to obtain feature maps at higher resolution. On the other hand,
    ViT-based models down-sample the input by a factor of 16. Below is an example shown for multi-scale
    feature aggregation for image encoder backbone (ViT) whose expected output spatial dimension (before classification layer)
    is expected to be 1/16th of the the input image's spatial dimension.

    input (HxW) --> L1 --> L2 (H/2 * W/2) --> L3 (H/4 * W/4) --> L4 (H/8 * W/8) --> L5 (H/16 * W/16)
                                |x0.5      | x1.              | 2x              | 4x
                                |------> Concat <-------------------------------|

    ...note:
        Below is the naming convention that is used in CoreNet for layer and output key name mapping.
                                    ------------------------------
                                    Model Layer  | Output key name
                                    -------------|----------------
                                    layer_1      |    out_l1
                                    layer_2      |    out_l2
                                    layer_3      |    out_l3
                                    layer_4      |    out_l4
                                    layer_5      |    out_l5
                                    conv_1x1_exp |    out_l5_exp
                                    ------------------------------
    """

    def __init__(self, opts, enc_conf: Dict, *args, **kwargs) -> None:

        expected_encoder_output_keys = ["out_l2", "out_l3", "out_l4", "out_l5"]
        encoder_level5_output_key = "out_l5"
        if "use_l5_exp" in kwargs and kwargs["use_l5_exp"]:
            # some of the models (e.g., MobileNetv3) uses expansion layer in the classification backbone
            # When enabled, the output of expansion layer is used.
            expected_encoder_output_keys = ["out_l2", "out_l3", "out_l4", "out_l5_exp"]
            encoder_level5_output_key = "out_l5_exp"

        aspp_in_channels = getattr(
            opts, "model.segmentation.deeplabv3.aspp_in_channels"
        )

        super().__init__(
            opts, enc_conf, aspp_in_channels=aspp_in_channels, *args, **kwargs
        )

        proj_in_channels = (
            # Increase in latent dimension due to pixel shuffle.
            (self.enc_l2_channels * 4)
            + (self.enc_l3_channels)
            # decrease in latent dimension due to pixel unshuffle.
            + (self.enc_l4_channels // 4)
            + (
                (self.enc_l5_channels // 16)
                if encoder_level5_output_key == "out_l5"
                else (self.enc_l5_exp_channels // 16)
            )
        )

        self.msc_fusion = ConvLayer2d(
            opts,
            in_channels=proj_in_channels,
            # The output of multi-scale fusion block is fed to ASPP. Therefore,
            # the output channels of this block is equal to input channels in the ASPP block.
            out_channels=aspp_in_channels,
            bias=False,
            kernel_size=1,
            use_norm=True,
            use_act=True,
        )
        if self.upsample_seg_out is not None:
            # The feature maps from different levels are fused at spatial level 3 (or L3) whose spatial size is
            # 1/4th of the original input image. Therefore, the output of this block should be upsampled by 4x
            # inside the 'forward' method.
            output_upsample_factor = getattr(
                opts, "model.segmentation.deeplabv3.output_upsample_factor"
            )
            if output_upsample_factor is None:
                logger.error(
                    f"Please specify --model.segmentation.deeplabv3.output-upsample-factor in {self.__class__.__name__}."
                )
            self.upsample_seg_out.scale_factor = output_upsample_factor
        self.expected_encoder_output_keys = expected_encoder_output_keys
        self.encoder_level5_output_key = encoder_level5_output_key

        self.reset_head_parameters(opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """MultiScaleDeeplabV3 specific arguments"""
        if cls == MultiScaleDeeplabV3:
            group = parser.add_argument_group(title=cls.__name__)
            group.add_argument(
                "--model.segmentation.deeplabv3.aspp-in-channels",
                type=int,
                default=512,
                help=f"Input channels of the ASPP module. This is only used in {cls.__name__}. \
                    Defaults to 512.",
            )
            group.add_argument(
                "--model.segmentation.deeplabv3.output-upsample-factor",
                type=int,
                default=None,
                help=f"Output stide of the image encoder. This argument is used on {cls.__name__}. \
                    Default value is None.",
            )
        return parser

    def forward_seg_head(self, enc_out: Dict) -> Tensor:
        """Forward method for MultiScaleDeeplabV3 segmentation head.

        Args:
            enc_out: A dictionary containing the feature maps from different spatial levels in the image encoder.

        Returns:
            A 4D tensor whose spatial size is 1/4th of the input image.

        ...note:
            Often times, strides of a model are adjusted to produce a high resolution feature map. In such cases,
            the spatial dimensions from different levels may be different than the expected size. To address this,
            we dynamically check the output size of a given level is equal to output of level 3 or not. If not, feature
            map is interpolated using bilinear interpolation to the same size as level 3.
        """

        if not set(self.expected_encoder_output_keys).issubset(enc_out.keys()):
            logger.error(
                f"{self.__class__.__name__} requires featuers from {self.expected_encoder_output_keys} levels \
                for multi-scale fusion, but got only following levels: {enc_out.keys()}."
            )

        # encoder outputs are in [batch, channel, height, width] format
        level3_spatial_dims = enc_out["out_l3"].shape[2:]

        out_l2_to_l3 = F.pixel_unshuffle(enc_out["out_l2"], downscale_factor=2)
        if out_l2_to_l3.shape[2:] != level3_spatial_dims:
            out_l2_to_l3 = F.interpolate(
                out_l2_to_l3,
                size=list(level3_spatial_dims),
                mode="bilinear",
                align_corners=True,
            )

        out_l3 = enc_out["out_l3"]

        out_l4_to_l3 = F.pixel_shuffle(enc_out["out_l4"], upscale_factor=2)
        if out_l4_to_l3.shape[2:] != level3_spatial_dims:
            out_l4_to_l3 = F.interpolate(
                out_l4_to_l3,
                size=list(level3_spatial_dims),
                mode="bilinear",
                align_corners=True,
            )

        out_l5_to_l3 = F.pixel_shuffle(
            enc_out[self.encoder_level5_output_key], upscale_factor=4
        )
        if out_l5_to_l3.shape[2:] != level3_spatial_dims:
            out_l5_to_l3 = F.interpolate(
                out_l5_to_l3,
                size=list(level3_spatial_dims),
                mode="bilinear",
                align_corners=True,
            )

        out = torch.cat([out_l2_to_l3, out_l3, out_l4_to_l3, out_l5_to_l3], dim=1)
        out = self.msc_fusion(out)

        out = super().forward_seg_head({"out_l5": out})
        return out
