#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any, Dict

import torch

from corenet.modeling.models import MODEL_REGISTRY, BaseAnyNNModel
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder


@MODEL_REGISTRY.register(name="__base__", type="segmentation")
class BaseSegmentation(BaseAnyNNModel):
    """Base class for segmentation networks.

    Args:
        opts: Command-line arguments
        encoder: Image classification network
    """

    def __init__(self, opts, encoder: BaseImageEncoder, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        self.lr_multiplier = getattr(opts, "model.segmentation.lr_multiplier")
        assert isinstance(
            encoder, BaseImageEncoder
        ), "encoder should be an instance of BaseEncoder"
        self.encoder: BaseImageEncoder = encoder
        self.default_norm = getattr(opts, "model.normalization.name")
        self.opts = opts

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add segmentation model specific arguments"""
        if cls != BaseSegmentation:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--model.segmentation.name",
            type=str,
            default=None,
            help="Segmentation model name. Defaults to None.",
        )
        group.add_argument(
            "--model.segmentation.n-classes",
            type=int,
            # FIXME: In another PR make this default value to None and update configs.
            default=21,
            help="Number of classes in the dataset. Defaults to 21.",
        )
        group.add_argument(
            "--model.segmentation.pretrained",
            type=str,
            default=None,
            help="Path of the pretrained segmentation model. Useful for evaluation",
        )
        group.add_argument(
            "--model.segmentation.lr-multiplier",
            type=float,
            default=1.0,
            help="Multiply the learning rate in segmentation network (e.g., decoder) by this factor."
            "Defaults to 1.0.",
        )
        group.add_argument(
            "--model.segmentation.classifier-dropout",
            type=float,
            default=0.1,
            help="Dropout rate in classifier",
        )
        group.add_argument(
            "--model.segmentation.use-aux-head",
            action="store_true",
            help="Use auxiliary output",
        )
        group.add_argument(
            "--model.segmentation.aux-dropout",
            default=0.1,
            type=float,
            help="Dropout in auxiliary branch",
        )

        group.add_argument(
            "--model.segmentation.output-stride",
            type=int,
            default=None,
            help="Output stride in classification network",
        )
        group.add_argument(
            "--model.segmentation.replace-stride-with-dilation",
            action="store_true",
            help="Replace stride with dilation",
        )

        group.add_argument(
            "--model.segmentation.activation.name",
            default=None,
            type=str,
            help="Non-linear function type",
        )
        group.add_argument(
            "--model.segmentation.activation.inplace",
            action="store_true",
            help="Inplace non-linear functions",
        )
        group.add_argument(
            "--model.segmentation.activation.neg-slope",
            default=0.1,
            type=float,
            help="Negative slope in leaky relu",
        )
        group.add_argument(
            "--model.segmentation.freeze-batch-norm",
            action="store_true",
            help="Freeze batch norm layers",
        )

        group.add_argument(
            "--model.segmentation.use-level5-exp",
            action="store_true",
            default=False,
            help="Use output of Level 5 expansion layer in base feature extractor",
        )

        group.add_argument(
            "--model.segmentation.finetune-pretrained-model",
            action="store_true",
            default=False,
            help="Finetune a pretrained segmentation model. Defaults to False.",
        )
        group.add_argument(
            "--model.segmentation.n-pretrained-classes",
            type=int,
            default=None,
            help="Number of classes in the pre-trained segmentation model. "
            "Defaults to None.",
        )

        group.add_argument(
            "--model.segmentation.norm-layer",
            type=str,
            default="batch_norm",
            help="Normalization layer for segmentation. Defaults to batch_norm.",
        )
        return parser

    def maybe_seg_norm_layer(self):
        seg_norm_layer = getattr(self.opts, "model.segmentation.norm_layer")
        if seg_norm_layer is not None:
            # update the default norm layer
            setattr(self.opts, "model.normalization.name", seg_norm_layer)

    def set_default_norm_layer(self):
        setattr(self.opts, "model.normalization.name", self.default_norm)

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        """Create dummy input and labels for CI/CD purposes. Child classes must override it
        if functionality is different.
        """
        img_channels = 3
        height = 224
        width = 224
        n_classes = 10
        img_tensor = torch.randn(
            batch_size, img_channels, height, width, dtype=torch.float
        )
        label_tensor = torch.randint(
            low=0, high=n_classes, size=(batch_size, height, width)
        ).long()
        return {"samples": img_tensor, "targets": label_tensor}

    def update_classifier(self, opts, n_classes: int) -> None:
        """This function updates the classification layer in a model. Useful for finetuning purposes."""
        raise NotImplementedError

    @classmethod
    def set_model_specific_opts_before_model_building(
        cls, opts: argparse.Namespace, *args, **kwargs
    ) -> Dict[str, Any]:
        seg_act_fn = getattr(opts, "model.segmentation.activation.name")
        if seg_act_fn is not None:
            # Override the general activation arguments
            default_act_fn = getattr(opts, "model.activation.name", "relu")
            default_act_inplace = getattr(opts, "model.activation.inplace", False)
            default_act_neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)

            setattr(opts, "model.activation.name", seg_act_fn)
            setattr(
                opts,
                "model.activation.inplace",
                getattr(opts, "model.segmentation.activation.inplace", False),
            )
            setattr(
                opts,
                "model.activation.neg_slope",
                getattr(opts, "model.segmentation.activation.neg_slope", 0.1),
            )
            return {
                "model.activation.name": default_act_fn,
                "model.activation.inplace": default_act_inplace,
                "model.activation.neg_slope": default_act_neg_slope,
            }
        return {}


# TODO: Find models and configurations that uses `set_model_specific_opts_before_model_building` and
#  `unset_model_specific_opts_after_model_building` functions. Find a more explicit way of satisfying this requirement,
#  such as namespacing config entries in a more composable way so that we no longer have conflicting config entries.


def set_model_specific_opts_before_model_building(
    opts: argparse.Namespace,
) -> Dict[str, Any]:
    """Override library-level defaults with model-specific default values.

    Args:
        opts: Command-line arguments

    Returns:
        A dictionary containing the name of arguments that are updated along with their original values.
        This dictionary is used in `unset_model_specific_opts_after_model_building` function to unset the
        model-specific to library-specific defaults.
    """
    seg_act_fn = getattr(opts, "model.segmentation.activation.name")
    if seg_act_fn is not None:
        # Override the general activation arguments
        default_act_fn = getattr(opts, "model.activation.name", "relu")
        default_act_inplace = getattr(opts, "model.activation.inplace", False)
        default_act_neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)

        setattr(opts, "model.activation.name", seg_act_fn)
        setattr(
            opts,
            "model.activation.inplace",
            getattr(opts, "model.segmentation.activation.inplace", False),
        )
        setattr(
            opts,
            "model.activation.neg_slope",
            getattr(opts, "model.segmentation.activation.neg_slope", 0.1),
        )
        return {
            "model.activation.name": default_act_fn,
            "model.activation.inplace": default_act_inplace,
            "model.activation.neg_slope": default_act_neg_slope,
        }
    return {}


def unset_model_specific_opts_after_model_building(
    opts: argparse.Namespace, default_opts_info: Dict[str, Any], *ars, **kwargs
) -> None:
    """Given command-line arguments and a mapping of opts that needs to be unset, this function
    unsets the library-level defaults that were over-ridden previously
    in `set_model_specific_opts_before_model_building`.
    """
    assert isinstance(default_opts_info, dict), (
        f"Please ensure set_model_specific_opts_before_model_building() "
        f"returns a dict."
    )
    for k, v in default_opts_info.items():
        setattr(opts, k, v)
