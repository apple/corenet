#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

import torch
from torch import Tensor
from torch.nn import functional as F

from corenet.loss_fn import LOSS_REGISTRY
from corenet.loss_fn.distillation.base_distillation import BaseDistillationCriteria
from corenet.utils import logger


@LOSS_REGISTRY.register(name="soft_kl_loss", type="distillation")
class SoftKLLoss(BaseDistillationCriteria):
    """Soft KL Loss for classification tasks. Given an input sample, soft-labels (or probabilities)
    are generated from a teacher and KL loss is computed between soft-labels and student model's output.

    Args:
        opts: command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        temperature = getattr(opts, "loss.distillation.soft_kl_loss.temperature")
        if temperature <= 0.0:
            logger.error(
                f"The value of temperature in {self.__class__.__name__} should be positive."
            )

        super().__init__(opts, *args, **kwargs)

        self.temperature = temperature

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != SoftKLLoss:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.distillation.soft-kl-loss.temperature",
            type=float,
            default=1.0,
            help=f"Temperature for KL divergence loss in {cls.__name__}. Defaults to 1.",
        )
        return parser

    def _forward_distill(
        self, input_sample: Tensor, prediction: Tensor, *args, **kwargs
    ) -> Tensor:
        """Computes distillation loss.

        Args:
            input_sample: Input image tensor
            prediction: Student model's output.

        Shapes:
            input_sample: Shape is [Batch size, 3, height, width]
            prediction: Shape is [Batch size, number of classes]

        Returns:
            A scalar loss value.
        """
        with torch.no_grad():
            teacher_logits = self._logits_from_teacher(input_sample)
            teacher_lprobs = F.log_softmax(
                teacher_logits / self.temperature, dim=1
            ).detach()

        student_lprobs = F.log_softmax(prediction / self.temperature, dim=-1)
        kl_loss = F.kl_div(
            student_lprobs, teacher_lprobs, reduction="batchmean", log_target=True
        )
        return kl_loss * (self.temperature**2)

    def extra_repr(self) -> str:
        return f"\n\t temperature={self.temperature}"
