#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, Optional

from torch import Tensor

from corenet.modeling.layers import ConvLayer2d
from corenet.modeling.misc.init_utils import initialize_weights
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.segmentation.heads.base_seg_head import BaseSegHead
from corenet.modeling.modules import PSP


@MODEL_REGISTRY.register(name="pspnet", type="segmentation_head")
class PSPNet(BaseSegHead):
    """
    This class defines the segmentation head in `PSPNet architecture <https://arxiv.org/abs/1612.01105>`_
    Args:
        opts: command-line arguments
        enc_conf (Dict): Encoder input-output configuration at each spatial level
        use_l5_exp (Optional[bool]): Use features from expansion layer in Level5 in the encoder
    """

    def __init__(
        self, opts, enc_conf: dict, use_l5_exp: Optional[bool] = False, *args, **kwargs
    ) -> None:
        psp_out_channels = getattr(
            opts, "model.segmentation.pspnet.psp_out_channels", 512
        )
        psp_pool_sizes = getattr(
            opts, "model.segmentation.pspnet.psp_pool_sizes", [1, 2, 3, 6]
        )
        psp_dropout = getattr(opts, "model.segmentation.pspnet.psp_dropout", 0.1)

        super().__init__(opts=opts, enc_conf=enc_conf, use_l5_exp=use_l5_exp)

        psp_in_channels = (
            self.enc_l5_channels if not self.use_l5_exp else self.enc_l5_exp_channels
        )
        self.psp_layer = PSP(
            opts=opts,
            in_channels=psp_in_channels,
            out_channels=psp_out_channels,
            pool_sizes=psp_pool_sizes,
            dropout=psp_dropout,
        )
        self.classifier = ConvLayer2d(
            opts=opts,
            in_channels=psp_out_channels,
            out_channels=self.n_seg_classes,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
            bias=True,
        )
        self.reset_head_parameters(opts=opts)

    def update_classifier(self, opts, n_classes: int) -> None:
        """
        This function updates the classification layer in a model. Useful for finetuning purposes.
        """
        in_channels = self.classifier.in_channels
        conv_layer = ConvLayer2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=n_classes,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
            bias=True,
        )

        initialize_weights(opts, modules=conv_layer)
        self.classifier = conv_layer

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.segmentation.pspnet.psp-pool-sizes",
            type=int,
            nargs="+",
            default=[1, 2, 3, 6],
            help="Pool sizes in the PSPNet module",
        )
        group.add_argument(
            "--model.segmentation.pspnet.psp-out-channels",
            type=int,
            default=512,
            help="Output channels of PSPNet module",
        )
        group.add_argument(
            "--model.segmentation.pspnet.psp-dropout",
            type=float,
            default=0.1,
            help="Dropout in the PSPNet module",
        )
        return parser

    def forward_seg_head(self, enc_out: Dict) -> Tensor:
        # low resolution features
        x = enc_out["out_l5_exp"] if self.use_l5_exp else enc_out["out_l5"]

        # Apply PSP layer
        x = self.psp_layer(x)

        out = self.classifier(x)

        return out
