#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

import argparse
from typing import Any, Dict, Union

from torch import Tensor

from corenet.loss_fn import LOSS_REGISTRY, BaseCriteria
from corenet.utils import logger


@LOSS_REGISTRY.register(name="__base__", type="classification")
class BaseClassificationCriteria(BaseCriteria):
    """Base class for defining classification loss functions. Sub-classes must implement forward function.

    Args:
        opts: command line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add criteria-specific arguments to the parser."""
        if cls != BaseClassificationCriteria:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(cls.__name__)
        group.add_argument(
            "--loss.classification.name",
            type=str,
            default=None,
            help=f"Name of the loss function in {cls.__name__}. Defaults to None.",
        )
        return parser

    def _compute_loss(
        self, prediction: Tensor, target: Tensor, *args, **kwargs
    ) -> Tensor:
        """Sub-classes must override this function to compute loss

        Args:
            prediction: Output of the model
            target: ground truth

        Returns:
            Expected to return a scalar value of loss
        """
        raise NotImplementedError

    def forward(
        self,
        input_sample: Any,
        prediction: Union[Dict[str, Tensor], Tensor],
        target: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        """Computes the cross entropy loss.

        Args:
            input_sample: Input image tensor to model.
            prediction: Output of model. It can be a tensor or mapping of (string: Tensor). In case of a dictionary,
            `logits` is a required key.
            target: Target label tensor containing values in the range `[0, C)`, where :math:`C` is the number of classes

        Shapes:
            input_sample: This loss function does not care about this argument.
            prediction:
                * When prediction is a tensor, then shape is [N, C]
                * When prediction is a dictionary, then the shape of prediction["logits"] is [N, C]
            target: The shape of target tensor is [N]

        Returns:
            Scalar loss value is returned.
        """

        if isinstance(prediction, Tensor):
            return self._compute_loss(
                prediction=prediction, target=target, *args, **kwargs
            )
        elif isinstance(prediction, Dict):
            if "logits" not in prediction:
                logger.error(
                    f"logits is a required key in {self.__class__.__name__} when prediction type"
                    f"is dictionary. Got keys: {prediction.keys()}"
                )

            predicted_logits = prediction["logits"]
            if predicted_logits is None:
                logger.error("Predicted logits can not be None.")

            ce_loss = self._compute_loss(
                prediction=predicted_logits, target=target, *args, **kwargs
            )
            return ce_loss
        else:
            logger.error(
                f"Prediction should be either a Tensor or Dictionary[str, Tensor]. Got: {type(prediction)}"
            )
