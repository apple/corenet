#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from corenet.loss_fn import LOSS_REGISTRY
from corenet.loss_fn.classification.base_classification_criteria import (
    BaseClassificationCriteria,
)


@LOSS_REGISTRY.register(name="focal_loss", type="classification")
class FocalLoss(BaseClassificationCriteria):
    """Add focal loss, as introduced in RetinaNet
    (https://arxiv.org/pdf/1708.02002.pdf).

    This loss is similar to Cross Entropy, but downweights the loss from
    examples that are correctly classified with high confidence. This helps the
    classifier focus on hard examples. The weighting term is (1 - p)^gamma,
    where gamma is a hyperparameter.

    Arguments:
        gamma: The loss-weighting hyperparameter.
        weights: The class-specific loss weights, as a vector.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        self.gamma = getattr(opts, "loss.classification.focal_loss.gamma")
        self.weights = getattr(opts, "loss.classification.focal_loss.weights")
        if self.weights is not None:
            self.weights = torch.tensor(self.weights)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add Focal loss criteria-specific arguments to the parser."""
        if cls != FocalLoss:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.classification.focal-loss.gamma",
            type=float,
            default=0,
            help=f"Gamma of focal loss. Defaults to 0 and it's equvilent to CE loss.",
        )
        group.add_argument(
            "--loss.classification.focal-loss.weights",
            nargs="*",
            default=None,
            type=float,
            help=f"Weights for {cls.__name__}. Defaults to None.",
        )
        return parser

    def _compute_loss(
        self, prediction: Tensor, target: Tensor, *args, **kwargs
    ) -> Tensor:
        """Calculate the focal loss.

        Arguments:
            prediction: A tensor of shape [batch_size, num_classes] containing logits.
            targets: Either (1) a tensor of shape [batch_size] containing class ids, or
                (2) a tensor of shape [batch_size, num_classes] containing soft targets.

        Returns:
            A scalar loss value.
        """
        if prediction.dim() != 2:
            raise ValueError(f"Expected 2 dimensions, got {prediction.dim()}")
        if target.dim() not in (1, 2):
            raise ValueError(f"Expected 1 or 2 dimensions, got {target.dim()}")

        if target.dim() == 1:
            target = F.one_hot(target, num_classes=prediction.shape[-1])

        log_softmax_probabilities = torch.log_softmax(prediction, dim=-1)
        loss = -target * log_softmax_probabilities

        # Add the focal loss term.
        focal_term = (1 - torch.exp(log_softmax_probabilities)).pow(self.gamma)
        loss *= focal_term

        # Add the weights.
        if self.weights is not None:
            loss *= self.weights.to(loss.device)

        return loss.sum(dim=-1).mean()

    def extra_repr(self) -> str:
        return f"\n\t gamma={self.gamma}" f"\n\t weights={self.weights}"
