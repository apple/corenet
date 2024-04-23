#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, List, Optional, Tuple

from torch import nn

from corenet.modeling.layers import ConvLayer2d, Dropout, GlobalPool, LinearLayer
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.modeling.models.classification.config.mobilenetv2 import get_configuration
from corenet.modeling.modules import InvertedResidual
from corenet.utils.math_utils import bound_fn, make_divisible


@MODEL_REGISTRY.register(name="mobilenetv2", type="classification")
class MobileNetV2(BaseImageEncoder):
    """
    This class defines the `MobileNetv2 architecture <https://arxiv.org/abs/1801.04381>`_
    """

    def __init__(self, opts, *args, **kwargs) -> None:

        width_mult = getattr(
            opts, "model.classification.mobilenetv2.width_multiplier", 1.0
        )

        cfg = get_configuration(opts=opts)

        image_channels = 3
        input_channels = 32
        last_channel = 1280
        classifier_dropout = getattr(
            opts, "model.classification.classifier_dropout", 0.0
        )
        if classifier_dropout == 0.0 or classifier_dropout is None:
            val = round(0.2 * width_mult, 3)
            classifier_dropout = bound_fn(min_val=0.0, max_val=0.2, value=val)

        super().__init__(opts, *args, **kwargs)

        last_channel = make_divisible(
            last_channel * max(1.0, width_mult), self.round_nearest
        )
        self.model_conf_dict = dict()

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
            opts=opts,
            mv2_config=cfg["layer1"],
            width_mult=width_mult,
            input_channel=input_channels,
        )
        self.model_conf_dict["layer1"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.layer_2, out_channels = self._make_layer(
            opts=opts,
            mv2_config=cfg["layer2"],
            width_mult=width_mult,
            input_channel=input_channels,
        )
        self.model_conf_dict["layer2"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.layer_3, out_channels = self._make_layer(
            opts=opts,
            mv2_config=cfg["layer3"],
            width_mult=width_mult,
            input_channel=input_channels,
        )
        self.model_conf_dict["layer3"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.layer_4, out_channels = self._make_layer(
            opts=opts,
            mv2_config=[cfg["layer4"], cfg["layer4_a"]],
            width_mult=width_mult,
            input_channel=input_channels,
            dilate=self.dilate_l4,
        )
        self.model_conf_dict["layer4"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.layer_5, out_channels = self._make_layer(
            opts=opts,
            mv2_config=[cfg["layer5"], cfg["layer5_a"]],
            width_mult=width_mult,
            input_channel=input_channels,
            dilate=self.dilate_l5,
        )
        self.model_conf_dict["layer5"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.conv_1x1_exp = ConvLayer2d(
            opts=opts,
            in_channels=input_channels,
            out_channels=last_channel,
            kernel_size=1,
            stride=1,
            use_act=True,
            use_norm=True,
        )
        self.model_conf_dict["exp_before_cls"] = {
            "in": input_channels,
            "out": last_channel,
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
                in_features=last_channel, out_features=self.n_classes, bias=True
            ),
        )

        self.model_conf_dict["cls"] = {"in": last_channel, "out": self.n_classes}

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.mobilenetv2.width-multiplier",
            type=float,
            default=1.0,
            help="Width multiplier for MobileNetv2. Default: 1.0",
        )
        return parser

    def _make_layer(
        self,
        opts,
        mv2_config: Dict or List,
        width_mult: float,
        input_channel: int,
        dilate: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:
        prev_dilation = self.dilation
        mv2_block = nn.Sequential()
        count = 0

        if isinstance(mv2_config, Dict):
            mv2_config = [mv2_config]

        for cfg in mv2_config:
            t = cfg.get("expansion_ratio")
            c = cfg.get("out_channels")
            n = cfg.get("num_blocks")
            s = cfg.get("stride")

            output_channel = make_divisible(c * width_mult, self.round_nearest)

            for block_idx in range(n):
                stride = s if block_idx == 0 else 1
                block_name = "mv2_block_{}".format(count)
                if dilate and count == 0:
                    self.dilation *= stride
                    stride = 1

                layer = InvertedResidual(
                    opts=opts,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    stride=stride,
                    expand_ratio=t,
                    dilation=prev_dilation if count == 0 else self.dilation,
                )
                mv2_block.add_module(name=block_name, module=layer)
                count += 1
                input_channel = output_channel
        return mv2_block, input_channel
