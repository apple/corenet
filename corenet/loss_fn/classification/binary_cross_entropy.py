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


@LOSS_REGISTRY.register(name="binary_cross_entropy", type="classification")
class BinaryCrossEntropy(BaseClassificationCriteria):
    """Binary cross-entropy loss for classification tasks

    Args:
        opts: command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        self.reduction = getattr(
            opts,
            "loss.classification.binary_cross_entropy.reduction",
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BinaryCrossEntropy:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--loss.classification.binary-cross-entropy.reduction",
            type=str,
            default="mean",
            choices=["sum", "mean", "none", "batch_mean"],
            help="Specifies the reduction to apply to the output (default='mean')."
            " 'batch_mean' divides the sum of the loss only by the first dimension.",
        )
        return parser

    def _compute_loss(
        self, prediction: Tensor, target: Tensor, *args, **kwargs
    ) -> Tensor:
        """The binary cross-entropy loss with logits for binary classification.
        The probability for class one is the Sigmoid on the logit.
        For multi-class problems with multiple valid labels, the loss penalizes by
        the given target probability of the same shape as predictions.

        Args:
            prediction: Logits of class 1
            target: Ground-truth class index or probability.

        Shapes:
            prediction: [Batch size, ...]
            target: A tensor of similar shape to prediction if the target
                probability for each output is known. Or a tensor of ground-truth labels
                missing the last dimension of size `num_classes`.

        Returns:
            If reduction is none, then tensor of the same shape as prediction is returned.
            Otherwise, a scalar loss value is returned.
        """
        if target.dim() == (prediction.dim() - 1):
            # Ground truth labels are hard labels. Convert to one-hot labels.
            target = F.one_hot(target, num_classes=prediction.shape[-1])

        div_by = 1.0
        if self.reduction == "batch_mean":
            div_by = target.shape[0]
        reduction = self.reduction if self.reduction != "batch_mean" else "sum"

        bce_loss = F.binary_cross_entropy_with_logits(
            input=prediction,
            target=target.to(prediction.dtype),
            reduction=reduction,
        )

        return bce_loss / div_by

    def extra_repr(self) -> str:
        return f"\n\t reduction={self.reduction}"
