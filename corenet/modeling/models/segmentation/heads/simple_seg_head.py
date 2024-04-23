#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, Optional

from torch import Tensor

from corenet.modeling.layers import ConvLayer2d
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.segmentation.heads.base_seg_head import BaseSegHead


@MODEL_REGISTRY.register(name="simple_seg_head", type="segmentation_head")
class SimpleSegHead(BaseSegHead):
    """
    This class defines the simple segmentation head with merely a classification layer. This is useful for performing
    linear probling on segmentation task.
    Args:
        opts: command-line arguments
        enc_conf (Dict): Encoder input-output configuration at each spatial level
        use_l5_exp (Optional[bool]): Use features from expansion layer in Level5 in the encoder
    """

    def __init__(
        self, opts, enc_conf: Dict, use_l5_exp: Optional[bool] = False, *args, **kwargs
    ) -> None:

        super().__init__(opts=opts, enc_conf=enc_conf, use_l5_exp=use_l5_exp)

        in_channels = (
            self.enc_l5_channels if not self.use_l5_exp else self.enc_l5_exp_channels
        )

        self.classifier = ConvLayer2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=self.n_seg_classes,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
            bias=True,
        )

        self.reset_head_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def forward_seg_head(self, enc_out: Dict) -> Tensor:
        x = enc_out["out_l5_exp"] if self.use_l5_exp else enc_out["out_l5"]
        # classify
        x = self.classifier(x)
        return x
