#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any, Dict, Optional, Union

from torch import Tensor

from corenet.loss_fn import LOSS_REGISTRY, BaseCriteria
from corenet.utils import logger


@LOSS_REGISTRY.register(name="__base__", type="language_modeling")
class BaseLanguageModelingCriteria(BaseCriteria):
    """Base class for defining loss functions for the task of language modeling.

    Args:
        opts: Command line arguments.

    ...note:
        Sub-classes must implement '_compute_loss' function.
    """

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add language modeling specific arguments to the parser."""
        if cls is BaseLanguageModelingCriteria:
            group = parser.add_argument_group(cls.__name__)
            group.add_argument(
                "--loss.language-modeling.name",
                type=str,
                default=None,
                help=f"Name of the loss function in {cls.__name__}. Defaults to None.",
            )
        return parser

    def _compute_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Sub-classes must override this function to compute loss.

        Args:
            prediction: Output of the model.
            target: Ground truth labels.

        Returns:
            Expected to return a scalar value of loss.
        """
        raise NotImplementedError(
            "Sub-classes must override this function to compute loss."
        )

    def forward(
        self,
        input_sample: Any,
        prediction: Union[Dict[str, Tensor], Tensor],
        target: Tensor,
        epoch: Optional[int] = None,
        iterations: Optional[int] = None,
    ) -> Union[Dict[str, Tensor], Tensor]:
        """Computes the loss.

        Args:
            input_sample: Input samples to model.
            prediction: Output of model. It can be a tensor or mapping of (string: Tensor). In case of mapping,
            `logits` is a required key.
            target: Target label tensor containing values in the range `[0, vocabulary size)`.
            epoch: Training epoch.
            iterations: Training iteration.

        Shapes:
            input_sample: This loss function does not care about this argument.
            prediction:
                * When prediction is a tensor, then shape is [batch size, sequence length, vocabulary size]
                * When prediction is a dictionary, then the shape of prediction["logits"] is [batch size, sequence length, vocabulary size]
            target: The shape of target tensor is [batch size, sequence length]

        Returns:
            Either of the following is returned as an output:
            1. 0-dimensional tensor containing the scalar loss value.
            2. Mapping of the form (string: 0-dimensional tensor) is returned with 'total_loss' as
                a mandatory key.

        ...note:
            While epoch and iteration values are currently not utilized in language modeling loss functions, they may be
            incorporated in future developments or research.
        """

        if isinstance(prediction, Tensor):
            return self._compute_loss(prediction=prediction, target=target)
        elif isinstance(prediction, Dict):
            if "logits" not in prediction:
                logger.error(
                    f"logits is a required key in {self.__class__.__name__} when prediction type"
                    f"is dictionary. Got keys: {prediction.keys()}"
                )

            predicted_logits = prediction["logits"]
            if predicted_logits is None:
                logger.error("Predicted logits can not be None.")

            loss = self._compute_loss(prediction=predicted_logits, target=target)

            return loss
        else:
            logger.error(
                f"Prediction should be either a Tensor or Dictionary[str, Tensor]. Got: {type(prediction)}"
            )
