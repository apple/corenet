#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
# isort: skip_file
from corenet.modeling.modules.base_module import BaseModule
from corenet.modeling.modules.squeeze_excitation import SqueezeExcitation
from corenet.modeling.modules.mobilenetv2 import InvertedResidual, InvertedResidualSE
from corenet.modeling.modules.resnet_modules import (
    BasicResNetBlock,
    BottleneckResNetBlock,
)
from corenet.modeling.modules.aspp_block import ASPP
from corenet.modeling.modules.transformer import TransformerEncoder
from corenet.modeling.modules.windowed_transformer import WindowedTransformerEncoder
from corenet.modeling.modules.pspnet_module import PSP
from corenet.modeling.modules.mobilevit_block import MobileViTBlock, MobileViTBlockv2
from corenet.modeling.modules.feature_pyramid import FeaturePyramidNetwork
from corenet.modeling.modules.ssd_heads import SSDHead, SSDInstanceHead
from corenet.modeling.modules.efficientnet import EfficientNetBlock
from corenet.modeling.modules.mobileone_block import MobileOneBlock, RepLKBlock
from corenet.modeling.modules.swin_transformer_block import (
    SwinTransformerBlock,
    PatchMerging,
    Permute,
)
from corenet.modeling.modules.regnet_modules import XRegNetBlock, AnyRegNetStage
from corenet.modeling.modules.flash_transformer import FlashTransformerEncoder


__all__ = [
    "InvertedResidual",
    "InvertedResidualSE",
    "BasicResNetBlock",
    "BottleneckResNetBlock",
    "ASPP",
    "TransformerEncoder",
    "WindowedTransformerEncoder",
    "SqueezeExcitation",
    "PSP",
    "MobileViTBlock",
    "MobileViTBlockv2",
    "MobileOneBlock",
    "RepLKBlock",
    "FeaturePyramidNetwork",
    "SSDHead",
    "SSDInstanceHead",
    "EfficientNetBlock",
    "SwinTransformerBlock",
    "PatchMerging",
    "Permute",
    "XRegNetBlock",
    "AnyRegNetStage",
    "FlashTransformerEncoder",
]
