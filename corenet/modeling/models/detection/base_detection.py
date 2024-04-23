#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict

from torch import nn

from corenet.modeling.misc.init_utils import initialize_weights
from corenet.modeling.models import MODEL_REGISTRY, BaseAnyNNModel, get_model
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.utils import logger


@MODEL_REGISTRY.register(name="__base__", type="detection")
class BaseDetection(BaseAnyNNModel):
    """Base class for the task of object detection

    Args:
        opts: Command-line arguments
        encoder: Image-encoder model (e.g., MobileNet or ResNet)
    """

    def __init__(
        self, opts: argparse.Namespace, encoder: BaseImageEncoder, *args, **kwargs
    ) -> None:
        super().__init__(opts, *args, **kwargs)
        assert isinstance(encoder, BaseImageEncoder)
        self.encoder: BaseImageEncoder = encoder
        self.n_detection_classes = getattr(opts, "model.detection.n_classes")

        enc_conf = self.encoder.model_conf_dict

        enc_ch_l5_out_proj = check_feature_map_output_channels(
            enc_conf, "exp_before_cls"
        )
        enc_ch_l5_out = check_feature_map_output_channels(enc_conf, "layer5")
        enc_ch_l4_out = check_feature_map_output_channels(enc_conf, "layer4")
        enc_ch_l3_out = check_feature_map_output_channels(enc_conf, "layer3")
        enc_ch_l2_out = check_feature_map_output_channels(enc_conf, "layer2")
        enc_ch_l1_out = check_feature_map_output_channels(enc_conf, "layer1")

        self.enc_l5_channels = enc_ch_l5_out
        self.enc_l5_channels_exp = enc_ch_l5_out_proj
        self.enc_l4_channels = enc_ch_l4_out
        self.enc_l3_channels = enc_ch_l3_out
        self.enc_l2_channels = enc_ch_l2_out
        self.enc_l1_channels = enc_ch_l1_out

        self.opts = opts

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add model specific arguments"""
        if cls != BaseDetection:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)

        group.add_argument(
            "--model.detection.name",
            type=str,
            default=None,
            help="Detection model name",
        )
        group.add_argument(
            "--model.detection.n-classes",
            type=int,
            default=80,
            help="Number of classes in the dataset. Defaults to 80.",
        )
        group.add_argument(
            "--model.detection.pretrained",
            type=str,
            default=None,
            help="Path of the pretrained detection model. Defaults to None.",
        )
        group.add_argument(
            "--model.detection.output-stride",
            type=int,
            default=None,
            help="Output stride of the classification network. Defaults to None.",
        )
        group.add_argument(
            "--model.detection.replace-stride-with-dilation",
            action="store_true",
            default=False,
            help="Replace stride with dilation",
        )
        group.add_argument(
            "--model.detection.freeze-batch-norm",
            action="store_true",
            default=False,
            help="Freeze batch norm layers in detection model. Defaults to False.",
        )

        return parser

    @staticmethod
    def reset_layer_parameters(layer: nn.Module, opts: argparse.Namespace) -> None:
        """Initialize weights of a given layer"""
        initialize_weights(opts=opts, modules=layer.modules())

    @classmethod
    def build_model(cls, opts: argparse.Namespace, *args, **kwargs) -> BaseAnyNNModel:
        output_stride = getattr(opts, "model.detection.output_stride", None)

        image_encoder = get_model(
            opts=opts,
            category="classification",
            output_stride=output_stride,
            *args,
            **kwargs
        )

        detection_model = cls(opts=opts, encoder=image_encoder, *args, **kwargs)

        if getattr(opts, "model.detection.freeze_batch_norm"):
            cls.freeze_norm_layers(opts, model=detection_model)
        return detection_model


def check_feature_map_output_channels(config: Dict, layer_name: str) -> int:
    enc_ch_l: Dict = config.get(layer_name, None)
    if enc_ch_l is None or not enc_ch_l:
        logger.error(
            "Encoder does not define input-output mapping for {}: Got: {}".format(
                layer_name, config
            )
        )

    enc_ch_l_out = enc_ch_l.get("out", None)
    if enc_ch_l_out is None or not enc_ch_l_out:
        logger.error(
            "Output channels are not defined in {} of the encoder. Got: {}".format(
                layer_name, enc_ch_l
            )
        )

    return enc_ch_l_out
