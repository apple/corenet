#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any, Callable, Optional, Tuple

import torch
from torch import nn

from corenet.modeling.layers import ConvLayer2d, Dropout, GlobalPool, LinearLayer
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.modeling.models.classification.config.efficientnet import (
    EfficientNetBlockConfig,
    get_configuration,
)
from corenet.modeling.modules import EfficientNetBlock


@MODEL_REGISTRY.register(name="efficientnet", type="classification")
class EfficientNet(BaseImageEncoder):
    """
    This class defines the `EfficientNet architecture <https://arxiv.org/abs/1905.11946>`_
    """

    def __init__(
        self,
        opts,
        *args,
        **kwargs: Any,
    ) -> None:
        super().__init__(opts, *args, **kwargs)

        classifier_dropout = getattr(opts, "model.classification.classifier_dropout")

        network_config = get_configuration(opts)
        last_channels = network_config["last_channels"]
        total_layers = network_config["total_layers"]
        stochastic_depth_prob = getattr(
            opts, "model.classification.efficientnet.stochastic_depth_prob", 0.2
        )

        # building first layer
        image_channels = 3
        in_channels = network_config["layer_1"][0].in_channels
        self.conv_1 = ConvLayer2d(
            opts=opts,
            in_channels=image_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": in_channels}

        # building inverted residual blocks
        prev_layers_cnt = 0  # counts the number of layers added so far
        for layer_name in ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5"]:
            dilation = False
            if layer_name == "layer_4":
                dilation = self.dilate_l4
            elif layer_name == "layer_5":
                dilation = self.dilate_l5
            layer, prev_layers_cnt = self._make_layer(
                opts=opts,
                block_config=network_config[layer_name],
                stochastic_depth_prob=stochastic_depth_prob,
                prev_layers_cnt=prev_layers_cnt,
                total_layers=total_layers,
                dilate=dilation,
            )

            setattr(self, layer_name, layer)
            # we have saved mappings without underscore in layer_name, so removing it
            self.model_conf_dict[layer_name.replace("_", "")] = {
                "in": network_config[layer_name][0].in_channels,
                "out": network_config[layer_name][-1].out_channels,
            }

        # building last several layers
        in_channels = network_config["layer_5"][-1].out_channels
        out_channels = last_channels
        self.conv_1x1_exp = ConvLayer2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_act=True,
            use_norm=True,
        )

        self.model_conf_dict["exp_before_cls"] = {
            "in": in_channels,
            "out": out_channels,
        }
        pool_type = getattr(opts, "model.layer.global_pool", "mean")

        self.classifier = nn.Sequential()
        self.classifier.add_module(
            name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False)
        )
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(
                name="classifier_dropout",
                module=Dropout(p=classifier_dropout, inplace=True),
            )
        self.classifier.add_module(
            name="classifier_fc",
            module=LinearLayer(
                in_features=out_channels, out_features=self.n_classes, bias=True
            ),
        )

    def _make_layer(
        self,
        opts,
        block_config,
        stochastic_depth_prob: float,
        prev_layers_cnt: int,  # number of layers before calling this function
        total_layers: int,  # Total number of layers in the network
        dilate: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Tuple[nn.Module, int]:
        # This is to accommodate segmentation architectures modifying strides of the backbone network.
        prev_dilation = self.dilation
        # For classification, dilation here should always be 1.
        block = []
        count = 0

        for layer_config in block_config:
            assert isinstance(layer_config, EfficientNetBlockConfig)
            in_channels = layer_config.in_channels
            out_channels = layer_config.out_channels
            for layer_idx in range(layer_config.num_layers):
                stride = layer_config.stride if layer_idx == 0 else 1
                if dilate and stride == 2:
                    self.dilation *= stride
                    stride = 1
                    dilate = False

                sd_prob = (
                    stochastic_depth_prob
                    * float(prev_layers_cnt + count)
                    / total_layers
                )
                sd_prob = round(sd_prob, 4)

                efficient_net_layer = EfficientNetBlock(
                    stochastic_depth_prob=sd_prob,
                    opts=opts,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=layer_config.kernel,
                    stride=stride,
                    expand_ratio=layer_config.expand_ratio,
                    dilation=prev_dilation if count == 0 else self.dilation,
                    use_hs=False,
                    use_se=True,
                    use_input_as_se_dim=True,
                    squeeze_factor=layer_config.expand_ratio * 4,
                    act_fn_name="swish",
                    se_scale_fn_name="sigmoid",
                )
                block.append(efficient_net_layer)
                count += 1
                in_channels = out_channels
        prev_layers_cnt += count
        return nn.Sequential(*block), prev_layers_cnt

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.efficientnet.mode",
            type=str,
            choices=[f"b{i}" for i in range(8)],
        )
        group.add_argument(
            "--model.classification.efficientnet.stochastic-depth-prob",
            type=float,
            default=0.0,
        )
        return parser

    def get_activation_checkpoint_submodule_class(self) -> Callable:
        return EfficientNetBlock
