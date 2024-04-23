#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from functools import partial
from typing import Dict, List, Tuple

import numpy as np
from torch import nn

from corenet.modeling.layers import (
    ConvLayer2d,
    Dropout,
    GlobalPool,
    Identity,
    LinearLayer,
)
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.modeling.models.classification.config.resnet import get_configuration
from corenet.modeling.modules import BasicResNetBlock, BottleneckResNetBlock


@MODEL_REGISTRY.register(name="resnet", type="classification")
class ResNet(BaseImageEncoder):
    """
    This class implements the `ResNet architecture <https://arxiv.org/pdf/1512.03385.pdf>`_

    .. note::
        Our ResNet implementation is different from the original implementation in two ways:
        1. First 7x7 strided conv is replaced with 3x3 strided conv
        2. MaxPool operation is replaced with another 3x3 strided depth-wise conv
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        image_channels = 3
        input_channels = 64
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout")

        stochastic_depth_prob = getattr(
            opts, "model.classification.resnet.stochastic_depth_prob"
        )

        pool_type = getattr(opts, "model.layer.global_pool")

        cfg = get_configuration(opts=opts)

        super().__init__(opts, *args, **kwargs)
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

        self.layer_1 = ConvLayer2d(
            opts=opts,
            in_channels=input_channels,
            out_channels=input_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
            groups=input_channels,
        )
        self.model_conf_dict["layer1"] = {"in": input_channels, "out": input_channels}

        # Stochastic depth variables
        block_repeats = [cfg[f"layer{i}"].get("num_blocks", 2) for i in range(2, 6)]
        block_start_indices = np.cumsum([0] + block_repeats[:-1])
        net_num_blocks = sum(block_repeats)
        stochastic_depth_fn = partial(
            self._block_stochastic_depth_prob,
            stochastic_depth_prob=stochastic_depth_prob,
            net_num_blocks=net_num_blocks,
        )

        start_idx = block_start_indices[0]
        num_blocks = cfg["layer2"]["num_blocks"]
        self.layer_2, out_channels = self._make_layer(
            opts=opts,
            in_channels=input_channels,
            layer_config=cfg["layer2"],
            stochastic_depth_probs=[
                stochastic_depth_fn(start_idx=start_idx, idx=idx)
                for idx in range(num_blocks)
            ],
        )
        self.model_conf_dict["layer2"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        start_idx = block_start_indices[1]
        num_blocks = cfg["layer3"]["num_blocks"]
        self.layer_3, out_channels = self._make_layer(
            opts=opts,
            in_channels=input_channels,
            layer_config=cfg["layer3"],
            stochastic_depth_probs=[
                stochastic_depth_fn(start_idx=start_idx, idx=idx)
                for idx in range(num_blocks)
            ],
        )
        self.model_conf_dict["layer3"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        start_idx = block_start_indices[2]
        num_blocks = cfg["layer4"]["num_blocks"]
        self.layer_4, out_channels = self._make_layer(
            opts=opts,
            in_channels=input_channels,
            layer_config=cfg["layer4"],
            stochastic_depth_probs=[
                stochastic_depth_fn(start_idx=start_idx, idx=idx)
                for idx in range(num_blocks)
            ],
            dilate=self.dilate_l4,
        )
        self.model_conf_dict["layer4"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        start_idx = block_start_indices[3]
        num_blocks = cfg["layer5"]["num_blocks"]
        self.layer_5, out_channels = self._make_layer(
            opts=opts,
            in_channels=input_channels,
            layer_config=cfg["layer5"],
            stochastic_depth_probs=[
                stochastic_depth_fn(start_idx=start_idx, idx=idx)
                for idx in range(num_blocks)
            ],
            dilate=self.dilate_l5,
        )
        self.model_conf_dict["layer5"] = {"in": input_channels, "out": out_channels}
        input_channels = out_channels

        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": input_channels,
            "out": input_channels,
        }

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

    def _block_stochastic_depth_prob(
        self,
        stochastic_depth_prob: float,
        idx: int,
        start_idx: int,
        net_num_blocks: int,
    ):
        """Computes the stochastic depth probability for a particular block in the network"""
        return round(
            stochastic_depth_prob * (idx + start_idx) / (net_num_blocks - 1), 4
        )

    def _make_layer(
        self,
        opts: argparse.Namespace,
        in_channels: int,
        layer_config: Dict,
        stochastic_depth_probs: List[float],
        dilate: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[nn.Sequential, int]:
        block_type = (
            BottleneckResNetBlock
            if layer_config.get("block_type", "bottleneck").lower() == "bottleneck"
            else BasicResNetBlock
        )
        mid_channels = layer_config.get("mid_channels")
        num_blocks = layer_config.get("num_blocks", 2)
        stride = layer_config.get("stride", 1)

        squeeze_channels = layer_config.get("squeeze_channels", None)

        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        out_channels = block_type.expansion * mid_channels
        dropout = getattr(opts, "model.classification.resnet.dropout")

        block = nn.Sequential()
        block.add_module(
            name="block_0",
            module=block_type(
                opts=opts,
                in_channels=in_channels,
                mid_channels=mid_channels,
                out_channels=out_channels,
                stride=stride,
                dilation=previous_dilation,
                dropout=dropout,
                stochastic_depth_prob=stochastic_depth_probs[0],
                squeeze_channels=squeeze_channels,
            ),
        )

        for block_idx in range(1, num_blocks):
            block.add_module(
                name="block_{}".format(block_idx),
                module=block_type(
                    opts=opts,
                    in_channels=out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    stride=1,
                    dilation=self.dilation,
                    dropout=dropout,
                    stochastic_depth_prob=stochastic_depth_probs[block_idx],
                    squeeze_channels=squeeze_channels,
                ),
            )

        return block, out_channels

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument("--model.classification.resnet.depth", type=int, default=50)
        group.add_argument(
            "--model.classification.resnet.dropout",
            type=float,
            default=0.0,
            help="Dropout in Resnet blocks. Defaults to 0.",
        )

        group.add_argument(
            "--model.classification.resnet.stochastic-depth-prob",
            type=float,
            default=0.0,
            help="Stochastic depth drop probability in Resnet blocks. Defaults to 0.",
        )

        group.add_argument(
            "--model.classification.resnet.se-resnet",
            action="store_true",
            default=False,
            help="Whether to use SE block to construct SE-ResNet model. Defaults to False.",
        )
        return parser
