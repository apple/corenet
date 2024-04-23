#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import copy

import torch.nn as nn

from corenet.modeling.layers import GlobalPool, Identity, LinearLayer
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.modeling.models.classification.config.mobileone import get_configuration
from corenet.modeling.modules.mobileone_block import MobileOneBlock


@MODEL_REGISTRY.register(name="mobileone", type="classification")
class MobileOne(BaseImageEncoder):
    """
    This class implements `MobileOne architecture <https://arxiv.org/pdf/2206.04040.pdf>`_
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        cfg = get_configuration(opts=opts)

        super().__init__(opts, *args, **kwargs)

        self.opts = opts
        image_channels = 3
        num_blocks_per_stage = cfg["num_blocks_per_stage"]
        width_multipliers = cfg["width_multipliers"]
        use_se = cfg["use_se"]
        self.num_conv_branches = cfg["num_conv_branches"]
        self.inference_mode = getattr(
            opts, "model.classification.mobileone.inference_mode"
        )

        assert len(width_multipliers) == 4
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.model_conf_dict = dict()

        # First dense conv
        self.conv_1 = MobileOneBlock(
            opts,
            in_channels=image_channels,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            inference_mode=self.inference_mode,
        )
        self.model_conf_dict["conv1"] = {"in": image_channels, "out": self.in_planes}

        self.layer_1 = Identity()
        self.model_conf_dict["layer1"] = {"in": self.in_planes, "out": self.in_planes}

        # Build stages
        self.cur_layer_idx = 1
        self.model_conf_dict["layer2"] = {
            "in": self.in_planes,
            "out": int(64 * width_multipliers[0]),
        }
        self.layer_2 = self._make_stage(
            int(64 * width_multipliers[0]), num_blocks_per_stage[0], num_se_blocks=0
        )
        self.model_conf_dict["layer3"] = {
            "in": self.in_planes,
            "out": int(128 * width_multipliers[1]),
        }
        self.layer_3 = self._make_stage(
            int(128 * width_multipliers[1]), num_blocks_per_stage[1], num_se_blocks=0
        )
        self.model_conf_dict["layer4"] = {
            "in": self.in_planes,
            "out": int(256 * width_multipliers[2]),
        }
        self.layer_4 = self._make_stage(
            int(256 * width_multipliers[2]),
            num_blocks_per_stage[2],
            num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0,
        )
        self.model_conf_dict["layer5"] = {
            "in": self.in_planes,
            "out": int(512 * width_multipliers[3]),
        }
        self.layer_5 = self._make_stage(
            int(512 * width_multipliers[3]),
            num_blocks_per_stage[3],
            num_se_blocks=num_blocks_per_stage[3] if use_se else 0,
        )

        # No extra 1x1 conv before classifier
        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": int(512 * width_multipliers[3]),
            "out": int(512 * width_multipliers[3]),
        }

        # Build classifier
        pool_type = getattr(opts, "model.layer.global_pool")
        self.classifier = nn.Sequential()
        self.classifier.add_module(
            name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False)
        )
        self.classifier.add_module(
            name="classifier_fc",
            module=LinearLayer(
                in_features=int(512 * width_multipliers[3]),
                out_features=self.n_classes,
                bias=True,
            ),
        )
        self.model_conf_dict["cls"] = {
            "in": int(512 * width_multipliers[3]),
            "out": self.n_classes,
        }

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add model specific arguments"""
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.mobileone.variant",
            type=str,
            default="s1",
            help="Variant string for MobileOne. Default: s1",
        )
        group.add_argument(
            "--model.classification.mobileone.inference-mode",
            type=bool,
            default=False,
            help="Flag to instantiate inference mode architecture. Default: False",
        )
        return parser

    def _make_stage(
        self, planes: int, num_blocks: int, num_se_blocks: int
    ) -> nn.Sequential:
        """Build a stage of MobileOne model.

        Args:
            planes: Number of output channels.
            num_blocks: Number of blocks in this stage.
            num_se_blocks: Number of SE blocks in this stage.

        Returns:
            A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError(
                    f"Number of SE blocks ({num_se_blocks}) cannot exceed number of layers ({num_blocks})."
                )
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # MobileOne block with depthwise conv
            blocks.append(
                MobileOneBlock(
                    self.opts,
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.in_planes,
                    inference_mode=self.inference_mode,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches,
                )
            )
            # MobileOne block with pointwise conv
            blocks.append(
                MobileOneBlock(
                    self.opts,
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    inference_mode=self.inference_mode,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches,
                )
            )
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def get_exportable_model(self) -> nn.Module:
        """Method returns a model where a multi-branched structure
           used in training is re-parameterized into a single branch
            for inference.

        Returns:
            Reparametrized MobileOne model for faster inference.
        """
        # Avoid editing original graph
        model = copy.deepcopy(self)
        for module in model.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()
        return model
