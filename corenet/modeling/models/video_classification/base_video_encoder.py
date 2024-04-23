#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.constants import SUPPORTED_VIDEO_CLIP_VOTING_FN
from corenet.modeling.misc.init_utils import initialize_weights
from corenet.modeling.models import MODEL_REGISTRY, BaseAnyNNModel


@MODEL_REGISTRY.register(name="__base__", type="video_classification")
class BaseVideoEncoder(BaseAnyNNModel):
    """Base class for the video backbones

    Args:
        opts: Command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        self.round_nearest = 8
        self.model_conf_dict = dict()
        self.inference_mode = getattr(
            opts, "model.video_classification.inference_mode", False
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BaseVideoEncoder:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--model.video-classification.classifier-dropout",
            type=float,
            default=0.0,
            help="Dropout rate in classifier",
        )

        group.add_argument(
            "--model.video-classification.name",
            type=str,
            default="mobilevit",
            help="Model name",
        )
        group.add_argument(
            "--model.video-classification.n-classes",
            type=int,
            default=1000,
            help="Number of classes in the dataset",
        )
        group.add_argument(
            "--model.video-classification.pretrained",
            type=str,
            default=None,
            help="Path of the pretrained backbone",
        )
        group.add_argument(
            "--model.video-classification.freeze-batch-norm",
            action="store_true",
            help="Freeze batch norm layers",
        )

        group.add_argument(
            "--model.video-classification.activation.name",
            default=None,
            type=str,
            help="Non-linear function type",
        )
        group.add_argument(
            "--model.video-classification.activation.inplace",
            action="store_true",
            help="Inplace non-linear functions",
        )
        group.add_argument(
            "--model.video-classification.activation.neg-slope",
            default=0.1,
            type=float,
            help="Negative slope in leaky relu",
        )
        group.add_argument(
            "--model.video-classification.clip-out-voting-fn",
            type=str,
            default="sum",
            choices=SUPPORTED_VIDEO_CLIP_VOTING_FN,
            help="How to fuse the outputs of different clips in a video",
        )

        group.add_argument(
            "--model.video-classification.inference-mode",
            action="store_true",
            help="Inference mode",
        )

        return parser

    @staticmethod
    def reset_module_parameters(opts, module) -> None:
        """Reset parameters for a specific module in the network"""
        initialize_weights(opts=opts, modules=module)
