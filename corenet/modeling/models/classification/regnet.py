#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, List, Tuple

from torch import nn

from corenet.modeling.layers import (
    AdaptiveAvgPool2d,
    ConvLayer2d,
    Dropout,
    Flatten,
    Identity,
    LinearLayer,
)
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.modeling.models.classification.config.regnet import (
    get_configuration,
    supported_modes,
)
from corenet.modeling.modules import AnyRegNetStage


@MODEL_REGISTRY.register(name="regnet", type="classification")
class RegNet(BaseImageEncoder):
    """
    This class implements the `RegNet architecture <https://arxiv.org/pdf/2003.13678.pdf>`_
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        image_channels = 3
        classifier_dropout = getattr(opts, "model.classification.classifier_dropout")

        cfg = get_configuration(opts=opts)

        # Output channels of first conv layer
        stem_width = getattr(opts, "model.classification.regnet.stem_width")

        stochastic_depth_prob = getattr(
            opts, "model.classification.regnet.stochastic_depth_prob"
        )
        stage_depths = [cfg[f"layer{i}"]["depth"] for i in range(1, 5)]

        super().__init__(opts, *args, **kwargs)
        self.model_conf_dict = dict()

        # Stem
        self.conv_1 = ConvLayer2d(
            opts=opts,
            in_channels=image_channels,
            out_channels=stem_width,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )
        self.model_conf_dict["conv1"] = {
            "in": image_channels,
            "out": stem_width,
        }

        # Body/stages
        in_channels = stem_width
        net_num_blocks = sum(stage_depths)
        for stage_index in range(1, 5):
            # Set stochastic depths for each block in the stage
            stage_depth = stage_depths[stage_index - 1]
            start_index = sum(stage_depths[: stage_index - 1])
            stochastic_depth_probs = [
                round(
                    stochastic_depth_prob * (i + start_index) / (net_num_blocks - 1), 4
                )
                for i in range(stage_depth)
            ]

            layer, out_channels = self._make_stage(
                opts=opts,
                width_in=in_channels,
                stage_config=cfg[f"layer{stage_index}"],
                stage_index=stage_index,
                stochastic_depth_probs=stochastic_depth_probs,
            )

            setattr(self, f"layer_{stage_index}", layer)

            self.model_conf_dict[f"layer{stage_index}"] = {
                "in": in_channels,
                "out": out_channels,
            }

            in_channels = out_channels

        self.layer_5 = Identity()
        self.model_conf_dict["layer5"] = {
            "in": in_channels,
            "out": in_channels,
        }

        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": in_channels,
            "out": in_channels,
        }

        # Head
        self.classifier = nn.Sequential()
        self.classifier.add_module(
            name="avg_pool",
            module=AdaptiveAvgPool2d(output_size=(1, 1), keep_dim=False),
        )
        self.classifier.add_module(name="flatten", module=Flatten())

        if classifier_dropout > 0:
            self.classifier.add_module(
                name="classifier_dropout", module=Dropout(p=classifier_dropout)
            )

        self.classifier.add_module(
            name="classifier_fc",
            module=LinearLayer(
                in_features=in_channels, out_features=self.n_classes, bias=True
            ),
        )

        self.model_conf_dict["cls"] = {"in": in_channels, "out": self.n_classes}

        self.check_model()
        self.reset_parameters(opts=opts)

    def _make_stage(
        self,
        opts: argparse.Namespace,
        width_in: int,
        stage_config: Dict,
        stage_index: int,
        stochastic_depth_probs: List[float],
        *args,
        **kwargs,
    ) -> Tuple[nn.Sequential, int]:
        stage_depth = stage_config["depth"]
        stage_width = stage_config["width"]
        groups = stage_config["groups"]
        stride = stage_config["stride"]
        bottleneck_multiplier = stage_config["bottleneck_multiplier"]
        se_ratio = stage_config["se_ratio"]

        stage = AnyRegNetStage(
            opts=opts,
            depth=stage_depth,
            width_in=width_in,
            width_out=stage_width,
            stride=stride,
            groups=groups,
            bottleneck_multiplier=bottleneck_multiplier,
            se_ratio=se_ratio,
            stage_index=stage_index,
            stochastic_depth_probs=stochastic_depth_probs,
        )

        return stage, stage_width

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != RegNet:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--model.classification.regnet.mode",
            type=str,
            default="y_4.0gf",
            help=f"The RegNet<mode> to use. Must be one of {', '.join(supported_modes)}. Defaults to y_4.0gf.",
        )
        group.add_argument(
            "--model.classification.regnet.stochastic-depth-prob",
            type=float,
            default=0.0,
            help="Stochastic depth drop probability in RegNet blocks. Defaults to 0.",
        )
        group.add_argument(
            "--model.classification.regnet.stem-width",
            type=int,
            default=32,
            help="The number of output channels of the first conv layer. Defaults to 32",
        )
        return parser
