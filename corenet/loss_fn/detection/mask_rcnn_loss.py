#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any, Dict

import torch
from torch import Tensor

from corenet.loss_fn import LOSS_REGISTRY
from corenet.loss_fn.detection.base_detection_criteria import BaseDetectionCriteria
from corenet.utils import logger


@LOSS_REGISTRY.register(name="mask_rcnn_loss", type="detection")
class MaskRCNNLoss(BaseDetectionCriteria):
    """Mask RCNN loss is computed inside the MaskRCNN model. This class is a wrapper to extract
    loss values for different heads (RPN, classification, etc.) and compute the weighted sum.

    Args:
        opts: command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

        self.classifier_weight = getattr(
            opts, "loss.detection.mask_rcnn_loss.classifier_weight"
        )
        self.box_reg_weight = getattr(
            opts, "loss.detection.mask_rcnn_loss.box_reg_weight"
        )
        self.mask_weight = getattr(opts, "loss.detection.mask_rcnn_loss.mask_weight")
        self.objectness_weight = getattr(
            opts, "loss.detection.mask_rcnn_loss.objectness_weight"
        )
        self.rpn_box_reg = getattr(opts, "loss.detection.mask_rcnn_loss.rpn_box_reg")
        # dev.device is not a part of model arguments. so test fails.
        # Setting a default value so test works
        self.device = getattr(opts, "dev.device", torch.device("cpu"))

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != MaskRCNNLoss:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.detection.mask-rcnn-loss.classifier-weight",
            type=float,
            default=1,
            help=f"Weight for classifier in {cls.__name__}. Defaults to 1.",
        )
        group.add_argument(
            "--loss.detection.mask-rcnn-loss.box-reg-weight",
            type=float,
            default=1,
            help=f"Weight for box reg in {cls.__name__}. Defaults to 1.",
        )
        group.add_argument(
            "--loss.detection.mask-rcnn-loss.mask-weight",
            type=float,
            default=1,
            help=f"Weight for mask in {cls.__name__}. Defaults to 1.",
        )
        group.add_argument(
            "--loss.detection.mask-rcnn-loss.objectness-weight",
            type=float,
            default=1,
            help=f"Weight for objectness in {cls.__name__}. Defaults to 1.",
        )
        group.add_argument(
            "--loss.detection.mask-rcnn-loss.rpn-box-reg",
            type=float,
            default=1,
            help=f"Weight for rpn box reg. in {cls.__name__}. Defaults to 1.",
        )
        return parser

    def forward(
        self,
        input_sample: Any,
        prediction: Dict[str, Tensor],
        *args,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """Compute MaskRCNN loss.

        Args:
            input_sample: Input image tensor to the model.
            prediction: Mapping of the Maskrcnn losses.

        Shapes:
            input_sample: This loss function does not care about input to the model.
            prediction: Dictionary containing scalar Mask RCNN loss values. Expected keys are
                loss_classifier, loss_box_reg, loss_mask, loss_objectness, loss_rpn_box_reg.

        Returns:
            A mapping of (string: scalar) is returned. Output contains following keys: (total_loss,
            loss_classifier, loss_box_reg, loss_mask, loss_objectness, loss_rpn_box_reg).
        """

        if not self.training:
            # MaskRCNN doesn't return the loss during validation. Therefore, we return 0.
            return {"total_loss": torch.tensor(0.0, device=self.device)}

        if not isinstance(prediction, Dict):
            logger.error(
                f"{self.__class__.__name__} requires prediction as a dictionary with "
                f"loss_classifier, loss_box_reg, loss_mask, loss_objectness, loss_rpn_box_reg as "
                f"mandatory keys. Got: {type(prediction)}."
            )

        if not {
            "loss_classifier",
            "loss_box_reg",
            "loss_mask",
            "loss_objectness",
            "loss_rpn_box_reg",
        }.issubset(prediction.keys()):
            logger.error(
                f"loss_classifier, loss_box_reg, loss_mask, loss_objectness, loss_rpn_box_reg are "
                f"required keys in {self.__class__.__name__}. Got: {prediction.keys()}"
            )

        total_loss = 0.0
        mask_rcnn_losses = {}

        for loss_key, loss_wt in zip(
            [
                "loss_classifier",
                "loss_box_reg",
                "loss_mask",
                "loss_objectness",
                "loss_rpn_box_reg",
            ],
            [
                self.classifier_weight,
                self.box_reg_weight,
                self.mask_weight,
                self.objectness_weight,
                self.rpn_box_reg,
            ],
        ):
            loss_ = prediction[loss_key] * loss_wt
            total_loss += loss_
            mask_rcnn_losses[loss_key] = loss_
        mask_rcnn_losses.update({"total_loss": total_loss})
        return mask_rcnn_losses

    def extra_repr(self) -> str:
        return (
            f"\n\t classifier_wt={self.classifier_weight}"
            f"\n\t box_reg_weight={self.box_reg_weight}"
            f"\n\t mask_weight={self.mask_weight}"
            f"\n\t objectness_weight={self.objectness_weight}"
            f"\n\t rpn_box_reg={self.rpn_box_reg}"
        )
