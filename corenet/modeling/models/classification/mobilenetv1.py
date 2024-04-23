#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, List, Optional, Tuple

from torch import nn

from corenet.modeling.layers import (
    ConvLayer2d,
    Dropout,
    GlobalPool,
    Identity,
    LinearLayer,
    SeparableConv2d,
)
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.modeling.models.classification.config.mobilenetv1 import get_configuration
from corenet.utils.math_utils import bound_fn


@MODEL_REGISTRY.register(name="mobilenetv1", type="classification")
class MobileNetv1(BaseImageEncoder):
    """
    This class defines the `MobileNet architecture <https://arxiv.org/abs/1704.04861>`_
    """

    def __init__(self, opts, *args, **kwargs) -> None:

        image_channels = 3
        classifier_dropout = getattr(
            opts, "model.classification.classifier_dropout", 0.0
        )
        if classifier_dropout == 0.0:
            width_mult = getattr(
                opts, "model.classification.mobilenetv1.width_multiplier", 1.0
            )
            val = round(0.1 * width_mult, 3)
            classifier_dropout = bound_fn(min_val=0.0, max_val=0.1, value=val)

        super().__init__(opts, *args, **kwargs)

        cfg = get_configuration(opts=opts)

        self.model_conf_dict = dict()
        input_channels = cfg["conv1_out"]
        self.conv_1 = ConvLayer2d(
            opts=opts,
            in_channels=image_channels,
            out_channels=input_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )
        self.model_conf_dict["conv1"] = {"in": image_channels, "out": input_channels}

        self.layer_1, out_channels = self._make_layer(
            opts=opts, mv1_config=cfg["layer1"], input_channel=input_channels
        )
        self.model_conf_dict["layer1"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.layer_2, out_channels = self._make_layer(
            opts=opts, mv1_config=cfg["layer2"], input_channel=input_channels
        )
        self.model_conf_dict["layer2"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.layer_3, out_channels = self._make_layer(
            opts=opts, mv1_config=cfg["layer3"], input_channel=input_channels
        )
        self.model_conf_dict["layer3"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.layer_4, out_channels = self._make_layer(
            opts=opts,
            mv1_config=cfg["layer4"],
            input_channel=input_channels,
            dilate=self.dilate_l4,
        )

        self.model_conf_dict["layer4"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.layer_5, out_channels = self._make_layer(
            opts=opts,
            mv1_config=cfg["layer5"],
            input_channel=input_channels,
            dilate=self.dilate_l5,
        )
        self.model_conf_dict["layer5"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": input_channels,
            "out": input_channels,
        }

        pool_type = getattr(opts, "model.layer.global_pool", "mean")

        self.classifier = nn.Sequential()
        self.classifier.add_module(
            name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False)
        )
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(
                name="classifier_dropout", module=Dropout(p=classifier_dropout)
            )
        self.classifier.add_module(
            name="classifier_fc",
            module=LinearLayer(
                in_features=input_channels, out_features=self.n_classes, bias=True
            ),
        )

        self.model_conf_dict["cls"] = {"in": input_channels, "out": self.n_classes}

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add model specific arguments"""
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.mobilenetv1.width-multiplier",
            type=float,
            default=1.0,
            help="Width multiplier for MobileNetv1. Default: 1.0",
        )

        return parser

    def _make_layer(
        self,
        opts,
        mv1_config: Dict or List,
        input_channel: int,
        dilate: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:
        prev_dilation = self.dilation
        mv1_block = []

        out_channels = mv1_config.get("out_channels")
        stride = mv1_config.get("stride", 1)
        n_repeat = mv1_config.get("repeat", 0)

        if stride == 2:
            if dilate:
                self.dilation *= stride
                stride = 1

            mv1_block.append(
                SeparableConv2d(
                    opts=opts,
                    in_channels=input_channel,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    use_norm=True,
                    use_act=True,
                    dilation=prev_dilation,
                ),
            )
            input_channel = out_channels

        for i in range(n_repeat):
            mv1_block.append(
                SeparableConv2d(
                    opts=opts,
                    in_channels=input_channel,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    use_norm=True,
                    use_act=True,
                    dilation=self.dilation,
                ),
            )
            input_channel = out_channels

        return nn.Sequential(*mv1_block), input_channel
