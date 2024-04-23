#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from corenet.modeling.models import MODEL_REGISTRY, BaseAnyNNModel


@MODEL_REGISTRY.register(name="__base__", type="multi_modal_image_text")
class BaseMultiModalImageText(BaseAnyNNModel):
    """Base class for multi-modal image-text data

    Args:
        opts: Command-line arguments
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        self.lr_multiplier_img_encoder = getattr(
            opts, "model.multi_modal_image_text.lr_multiplier_img_encoder"
        )
        self.lr_multiplier_text_encoder = getattr(
            opts, "model.multi_modal_image_text.lr_multiplier_text_encoder"
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add model specific arguments"""
        if cls != BaseMultiModalImageText:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.multi-modal-image-text.name",
            type=str,
            default=None,
            help="Name of the multi-modal image-text model",
        )

        group.add_argument(
            "--model.multi-modal-image-text.lr-multiplier-img-encoder",
            type=float,
            default=1.0,
            help="LR multiplier for the image encoder in {}".format(cls.__name__),
        )
        group.add_argument(
            "--model.multi-modal-image-text.lr-multiplier-text-encoder",
            type=float,
            default=1.0,
            help="LR multiplier for the text encoder in {}".format(cls.__name__),
        )

        group.add_argument(
            "--model.multi-modal-image-text.pretrained",
            type=str,
            default=None,
            help="Path of the pretrained backbone",
        )
        group.add_argument(
            "--model.multi-modal-image-text.freeze-batch-norm",
            action="store_true",
            help="Freeze batch norm layers",
        )

        return parser
