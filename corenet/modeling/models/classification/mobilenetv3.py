#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Optional, Tuple

from torch import nn

from corenet.modeling.layers import ConvLayer2d, Dropout, GlobalPool, LinearLayer
from corenet.modeling.layers.activation import build_activation_layer
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.modeling.models.classification.config.mobilenetv3 import get_configuration
from corenet.modeling.modules import InvertedResidualSE
from corenet.utils.math_utils import bound_fn, make_divisible


@MODEL_REGISTRY.register(name="mobilenetv3", type="classification")
class MobileNetV3(BaseImageEncoder):
    """
    This class implements the `MobileNetv3 architecture <https://arxiv.org/abs/1905.02244>`_
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        width_mult = getattr(
            opts, "model.classification.mobilenetv3.width_multiplier", 1.0
        )
        classifier_dropout = getattr(
            opts, "model.classification.classifier_dropout", 0.0
        )
        if classifier_dropout == 0.0 or classifier_dropout is None:
            val = round(0.2 * width_mult, 3)
            classifier_dropout = bound_fn(min_val=0.0, max_val=0.2, value=val)

        image_channels = 3
        input_channels = make_divisible(16 * width_mult, 8)

        mv3_config = get_configuration(opts)

        super().__init__(opts, *args, **kwargs)

        self.conv_1 = nn.Sequential()
        self.conv_1.add_module(
            name="conv_3x3_bn",
            module=ConvLayer2d(
                opts=opts,
                in_channels=image_channels,
                out_channels=input_channels,
                kernel_size=3,
                stride=2,
                use_norm=True,
                use_act=False,
            ),
        )
        self.conv_1.add_module(
            name="act",
            module=build_activation_layer(opts, act_type="hard_swish", inplace=True),
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": input_channels}

        self.layer_1, out_channels = self._make_layer(
            opts=opts,
            mv3_config=mv3_config["layer_1"],
            width_mult=width_mult,
            input_channel=input_channels,
        )
        self.model_conf_dict["layer1"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.layer_2, out_channels = self._make_layer(
            opts=opts,
            mv3_config=mv3_config["layer_2"],
            width_mult=width_mult,
            input_channel=input_channels,
        )
        self.model_conf_dict["layer2"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.layer_3, out_channels = self._make_layer(
            opts=opts,
            mv3_config=mv3_config["layer_3"],
            width_mult=width_mult,
            input_channel=input_channels,
        )
        self.model_conf_dict["layer3"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.layer_4, out_channels = self._make_layer(
            opts=opts,
            mv3_config=mv3_config["layer_4"],
            width_mult=width_mult,
            input_channel=input_channels,
            dilate=self.dilate_l4,
        )
        self.model_conf_dict["layer4"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.layer_5, out_channels = self._make_layer(
            opts=opts,
            mv3_config=mv3_config["layer_5"],
            width_mult=width_mult,
            input_channel=input_channels,
            dilate=self.dilate_l5,
        )
        self.model_conf_dict["layer5"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.conv_1x1_exp = nn.Sequential()
        out_channels = 6 * input_channels
        self.conv_1x1_exp.add_module(
            name="conv_1x1",
            module=ConvLayer2d(
                opts=opts,
                in_channels=input_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                use_act=False,
                use_norm=True,
            ),
        )
        self.conv_1x1_exp.add_module(
            name="act",
            module=build_activation_layer(opts, act_type="hard_swish", inplace=True),
        )
        self.model_conf_dict["exp_before_cls"] = {
            "in": input_channels,
            "out": out_channels,
        }

        pool_type = getattr(opts, "model.layer.global_pool", "mean")

        last_channels = mv3_config["last_channels"]
        self.classifier = nn.Sequential()
        self.classifier.add_module(
            name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False)
        )
        self.classifier.add_module(
            name="fc1",
            module=LinearLayer(
                in_features=out_channels, out_features=last_channels, bias=True
            ),
        )
        self.classifier.add_module(
            name="act",
            module=build_activation_layer(opts, act_type="hard_swish", inplace=True),
        )
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(
                name="classifier_dropout", module=Dropout(p=classifier_dropout)
            )
        self.classifier.add_module(
            name="classifier_fc",
            module=LinearLayer(
                in_features=last_channels, out_features=self.n_classes, bias=True
            ),
        )

        self.model_conf_dict["cls"] = {"in": 6 * input_channels, "out": self.n_classes}

    def _make_layer(
        self,
        opts,
        mv3_config,
        width_mult: float,
        input_channel: int,
        dilate: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:
        prev_dilation = self.dilation
        mv3_block = nn.Sequential()
        count = 0

        for i in range(len(mv3_config)):
            for kernel_size, expansion_factor, in_channels, use_se, use_hs, stride in [
                mv3_config[i]
            ]:
                block_name = "mv3_s_{}_idx_{}".format(stride, count)
                output_channel = make_divisible(
                    in_channels * width_mult, self.round_nearest
                )

                if dilate and count == 0:
                    self.dilation *= stride
                    stride = 1

                layer = InvertedResidualSE(
                    opts=opts,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    stride=stride,
                    expand_ratio=expansion_factor,
                    dilation=prev_dilation if count == 0 else self.dilation,
                    act_fn_name="hard_swish" if use_hs else "relu",
                    use_se=use_se,
                )
                mv3_block.add_module(name=block_name, module=layer)
                count += 1
                input_channel = output_channel
        return mv3_block, input_channel

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.mobilenetv3.mode",
            type=str,
            default="large",
            help="Configuration for mobilenetv3. Default: large",
            choices=("small", "large"),
        )
        group.add_argument(
            "--model.classification.mobilenetv3.width-multiplier",
            type=float,
            default=1.0,
            help="Width multiplier for mobilenetv3. Default: 1.0",
        )
        return parser
