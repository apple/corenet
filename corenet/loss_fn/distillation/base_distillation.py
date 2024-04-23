#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse
from typing import Mapping, Union

import torch
from torch import Tensor

from corenet.loss_fn import LOSS_REGISTRY, BaseCriteria
from corenet.loss_fn.utils.build_helper import build_cls_teacher_from_opts
from corenet.utils import logger


@LOSS_REGISTRY.register(name="__base__", type="distillation")
class BaseDistillationCriteria(BaseCriteria):
    """Base class for defining distillation loss functions. Sub-classes must implement `_forward_distill` function.

    Args:
        opts: command line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

        self.teacher = build_cls_teacher_from_opts(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != BaseDistillationCriteria:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--loss.distillation.name",
            type=str,
            default=None,
            help="Name of the loss function. Defaults to None.",
        )
        return parser

    @torch.no_grad()
    def _logits_from_teacher(self, input_sample: Tensor) -> Tensor:
        """Compute logits from teacher given input image tensor.

        Args:
            input_sample: Input image tensor

        Shape:
            input_sample: Shape is [Batch size, 3, height, width]
            teacher_output or teacher_output["logits"]: Shape is [Batch size, number of classes]

        Returns:
            Teacher output tensor (without softmax)

        ...note:
            The output of teacher can be Tensor or Dict[str, Tensor]. In case
            of dictionary, logits is a mandatory key.
        """
        self.teacher.eval()
        teacher_output: Union[Tensor, Mapping[str, Tensor]] = self.teacher(input_sample)
        if isinstance(teacher_output, Mapping):
            if "logits" not in teacher_output:
                logger.error(
                    "The output type of teacher is dictionary and must contain logits as a key."
                    f"Got: {teacher_output.keys()}"
                )
            return teacher_output["logits"]
        return teacher_output

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
        raise NotImplementedError

    def forward(
        self,
        input_sample: Tensor,
        prediction: Union[Mapping[str, Tensor], Tensor],
        target: Tensor,
        *args,
        **kwargs,
    ) -> Union[Mapping[str, Tensor], Tensor]:
        """Computes distillation loss

        Args:
            input_sample: Input image tensor.
            prediction: Output of model. It can be a tensor or mapping of (string: Tensor). In case of a dictionary,
            `logits` is a required key.
            target: Target label tensor containing values in the range `[0, C)`, where :math:`C` is the number of classes

        Shapes:
            input_sample: The shape of input tensor is [N, C, H, W]
            prediction:
                * When prediction is a tensor, then shape is [N, C]
                * When prediction is a dictionary, then shape of prediction["logits"] is [N, C]
            target: The shape of target tensor is [N]

        Returns:
            * Scalar loss value is returned.
        """

        if isinstance(prediction, Tensor):
            return self._forward_distill(
                input_sample=input_sample, prediction=prediction, *args, **kwargs
            )
        elif isinstance(prediction, Mapping):
            if "logits" not in prediction:
                logger.error(
                    f"logits is a required key in {self.__class__.__name__} when prediction type"
                    f"is dictionary. Got keys: {prediction.keys()}"
                )

            predicted_logits = prediction["logits"]

            # compute distillation loss
            distill_loss = self._forward_distill(
                input_sample=input_sample, prediction=predicted_logits, *args, **kwargs
            )
            return distill_loss
        else:
            logger.error(
                f"Prediction should be either a Tensor or Dictionary[str, Tensor]. Got: {type(prediction)}"
            )
