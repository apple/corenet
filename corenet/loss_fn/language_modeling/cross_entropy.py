#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, Union

from torch import Tensor
from torch.nn import functional as F

from corenet.loss_fn import LOSS_REGISTRY
from corenet.loss_fn.language_modeling.base_lm import BaseLanguageModelingCriteria


@LOSS_REGISTRY.register(name="cross_entropy", type="language_modeling")
class CrossEntropyLM(BaseLanguageModelingCriteria):
    """Cross entropy loss function for language modeling tasks.

    Args:
        opts: Command-line arguments.
    """

    def __init__(self, opts: argparse.Namespace) -> None:
        super().__init__(opts)

        self.ignore_idx = getattr(
            opts, "loss.language_modeling.cross_entropy.ignore_index"
        )
        if self.training:
            self.label_smoothing = getattr(
                opts, "loss.language_modeling.cross_entropy.label_smoothing"
            )
        else:
            # for validation/test sets, we compute standard CE loss
            self.label_smoothing = 0.0

        self.use_z_loss = getattr(
            opts, "loss.language_modeling.cross_entropy.use_z_loss"
        )
        self.z_loss_eps = getattr(
            opts, "loss.language_modeling.cross_entropy.z_loss_eps"
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add cross-entropy criteria-specific arguments to the parser."""
        if cls == CrossEntropyLM:
            group = parser.add_argument_group(title=cls.__name__)
            group.add_argument(
                "--loss.language-modeling.cross-entropy.ignore-index",
                type=int,
                default=-1,
                help=f"Target value that is ignored and does not contribute to "
                f"the input gradient in {cls.__name__}. Defaults to -1.",
            )
            group.add_argument(
                "--loss.language-modeling.cross-entropy.label-smoothing",
                type=float,
                default=0.0,
                help=f"Specifies the amount of smoothing when computing the loss in {cls.__name__}, "
                f"where 0.0 means no smoothing. Defaults to 0.0.",
            )
            group.add_argument(
                "--loss.language-modeling.cross-entropy.use-z-loss",
                action="store_true",
                default=False,
                help="Use z-loss with cross-entropy loss. Defaults to False.",
            )
            group.add_argument(
                "--loss.language-modeling.cross-entropy.z-loss-eps",
                default=1.0e-4,
                type=float,
                help="Epsilon value for z-loss. Defaults to 0.0001.",
            )
        return parser

    def _compute_loss(
        self, prediction: Tensor, target: Tensor
    ) -> Union[Dict[str, Tensor], Tensor]:
        """Computes cross-entropy loss between prediction and target tensors.

        Args:
            prediction: Predicted tensor of shape [batch size, sequence length, vocabulary size].
            target: Target label tensor containing values in the range `[0, vocabulary size)`. The shape of tensor is
                [batch size, sequence length].

        Returns:
            1. If z-loss is disabled, then a 0-dimensional tensor containing scalar loss value is returned.
            2. If z-loss is enabled, then a dictionary of the form (string: 0-dimensional tensor) is returned
                with three keys: 'total_loss', 'ce_loss', and 'z_loss'.
        """

        batch_size, seq_length, vocab_size = prediction.shape
        prediction = prediction.reshape(batch_size * seq_length, vocab_size)
        target = target.reshape(batch_size * seq_length)
        ce_loss = F.cross_entropy(
            input=prediction,
            target=target,
            ignore_index=self.ignore_idx,
            label_smoothing=self.label_smoothing,
        )

        if self.use_z_loss:
            # Adaption of Eq. (5) for z-loss computation in https://arxiv.org/pdf/2202.08906.pdf (non-router use-case).
            valid_tokens = (target != self.ignore_idx).type_as(prediction)
            # do not compute z_loss for ignored indices
            z_loss = (prediction * valid_tokens[:, None]).logsumexp(-1).pow(2).sum()
            z_loss *= self.z_loss_eps / valid_tokens.sum()
            return {
                "total_loss": ce_loss + z_loss,
                "ce_loss": ce_loss,
                "z_loss": z_loss,
            }
        return ce_loss

    def extra_repr(self) -> str:
        loss_info_str = (
            f"\n\t ignore_idx={self.ignore_idx}"
            f"\n\t label_smoothing={self.label_smoothing}"
        )
        if self.use_z_loss:
            loss_info_str += (
                f"\n\t use_z_loss={self.use_z_loss}"
                f"\n\t z_loss_eps={self.z_loss_eps}"
            )

        return loss_info_str
