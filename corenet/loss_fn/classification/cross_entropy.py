#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

from torch import Tensor
from torch.nn import functional as F

from corenet.loss_fn import LOSS_REGISTRY
from corenet.loss_fn.classification.base_classification_criteria import (
    BaseClassificationCriteria,
)
from corenet.loss_fn.utils.class_weighting import compute_class_weights


@LOSS_REGISTRY.register(name="cross_entropy", type="classification")
class CrossEntropy(BaseClassificationCriteria):
    """Cross entropy loss function for image classification tasks

    Args:
        opts: command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

        self.ignore_idx = getattr(
            opts, "loss.classification.cross_entropy.ignore_index"
        )
        self.use_class_wts = getattr(
            opts, "loss.classification.cross_entropy.class_weights"
        )
        self.label_smoothing = getattr(
            opts, "loss.classification.cross_entropy.label_smoothing"
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add cross-entropy criteria-specific arguments to the parser."""
        if cls != CrossEntropy:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.classification.cross-entropy.class-weights",
            action="store_true",
            default=False,
            help=f"Use class weights in {cls.__name__}. Defaults to False.",
        )
        group.add_argument(
            "--loss.classification.cross-entropy.ignore-index",
            type=int,
            default=-1,
            help=f"Target value that is ignored and does not contribute to "
            f"the input gradient in {cls.__name__}. Defaults to -1.",
        )
        group.add_argument(
            "--loss.classification.cross-entropy.label-smoothing",
            type=float,
            default=0.0,
            help=f"Specifies the amount of smoothing when computing the loss in {cls.__name__}, "
            f"where 0.0 means no smoothing. Defaults to 0.0.",
        )
        return parser

    def _compute_loss(
        self, prediction: Tensor, target: Tensor, *args, **kwargs
    ) -> Tensor:
        """Computes cross-entropy loss between prediction and target tensors.

        Args:
            prediction: Predicted tensor of shape [N, C]
            target: Target label tensor of shape [N] containing values between [0, C),

            Here, :math:`C` is the number of classes and :math:`N` is the batch size

        Returns:
            A scalar loss value
        """
        weight = None
        if self.use_class_wts and self.training:
            n_classes = prediction.shape[1]
            weight = compute_class_weights(target=target, n_classes=n_classes)

        # for validation, we compute standard CE loss
        label_smoothing_val = self.label_smoothing if self.training else 0.0
        return F.cross_entropy(
            input=prediction,
            target=target,
            weight=weight,
            ignore_index=self.ignore_idx,
            label_smoothing=label_smoothing_val,
        )

    def extra_repr(self) -> str:
        return (
            f"\n\t ignore_idx={self.ignore_idx}"
            f"\n\t class_weighting={self.use_class_wts}"
            f"\n\t label_smoothing={self.label_smoothing}"
        )
