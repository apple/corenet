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


@LOSS_REGISTRY.register(name="hard_distillation", type="distillation")
class HardDistillationLoss(BaseDistillationCriteria):
    """Hard distillation using cross-entropy for classification tasks. Given an input sample, hard-labels
    are generated from a teacher and cross-entropy loss is computed between hard-labels and student model's output.

    Args:
        opts: command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        top_k = getattr(opts, "loss.distillation.hard_distillation.topk")
        if top_k < 1:
            logger.error(f"The value of top-k should be greater than 0. Got: {top_k}")

        label_smoothing = getattr(
            opts, "loss.distillation.hard_distillation.label_smoothing"
        )
        if not (0.0 <= label_smoothing < 1.0):
            logger.error(
                f"The value of label smoothing should be between 0 and 1. Got: {label_smoothing}"
            )

        super().__init__(opts, *args, **kwargs)

        self.topk = top_k
        self.label_smoothing = label_smoothing

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != HardDistillationLoss:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.distillation.hard-distillation.topk",
            type=int,
            default=1,
            help=f"Distill top-k labels from teacher when in {cls.__name__}. Defaults to 1.",
        )
        group.add_argument(
            "--loss.distillation.hard-distillation.label-smoothing",
            type=float,
            default=0.0,
            help=f"Specifies the amount of smoothing when computing the classification loss in {cls.__name__}, "
            f"where 0.0 means no smoothing. Defaults to 0.0.",
        )
        return parser

    def _forward_distill(
        self, input_sample: Tensor, prediction: Tensor, *args, **kwargs
    ) -> Tensor:
        """
        Computes cross entropy loss between students and hard labels generated from teacher.

        Args:
            input_sample: Input image tensor
            prediction: Output of student model

        Shapes:
            input_sample: Shape is [Batch size, 3, height, width]
            prediction: Shape is [Batch size, Number of classes]

        Returns:
            A scalar loss value

        ...note:
            When top-k labels extracted from teacher are used for distillation, binary cross entropy loss is used.
        """

        with torch.no_grad():
            teacher_logits = self._logits_from_teacher(input_sample)
            teacher_probs = F.softmax(teacher_logits, dim=-1).detach()
            _, teacher_topk_labels = torch.topk(
                teacher_probs, k=self.topk, dim=-1, largest=True, sorted=True
            )

        if self.topk > 1:
            num_classes = prediction.shape[-1]
            teacher_topk_labels = F.one_hot(
                teacher_topk_labels, num_classes=num_classes
            )
            teacher_topk_labels = teacher_topk_labels.sum(1)
            teacher_topk_labels = teacher_topk_labels.to(dtype=prediction.dtype)

            # smooth labels corresponding to multiple classes
            smooth_class_p = (1.0 - self.label_smoothing) / self.topk
            # distribute the mass over remaining classes
            smooth_non_class_p = self.label_smoothing / (num_classes - self.topk)

            teacher_topk_labels = torch.where(
                teacher_topk_labels == 1.0, smooth_class_p, smooth_non_class_p
            )

            # scale by number of classes. Otherwise, the contribution is small
            loss = (
                F.binary_cross_entropy_with_logits(
                    input=prediction, target=teacher_topk_labels, reduction="mean"
                )
                * num_classes
            )
        else:
            teacher_topk_labels = teacher_topk_labels.reshape(-1)
            loss = F.cross_entropy(
                input=prediction,
                target=teacher_topk_labels,
                reduction="mean",
                label_smoothing=self.label_smoothing,
            )
        return loss

    def extra_repr(self) -> str:
        return f"\n\t topk={self.topk}" f"\n\tlabel_smoothing={self.label_smoothing}"
