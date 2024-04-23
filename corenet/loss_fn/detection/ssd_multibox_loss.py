#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any, Dict

import torch
from torch import Tensor
from torch.nn import functional as F

from corenet.loss_fn import LOSS_REGISTRY
from corenet.loss_fn.detection.base_detection_criteria import BaseDetectionCriteria
from corenet.third_party.modeling.ssd_utils import hard_negative_mining
from corenet.utils import logger
from corenet.utils.ddp_utils import is_master
from corenet.utils.tensor_utils import tensor_to_python_float


@LOSS_REGISTRY.register(name="ssd_multibox_loss", type="detection")
class SSDLoss(BaseDetectionCriteria):
    """Loss for single shot multi-box object detection

    Args:
        opts: command-line arguments
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        self.unscaled_reg_loss = 1e-7
        self.unscaled_conf_loss = 1e-7
        self.neg_pos_ratio = getattr(
            opts, "loss.detection.ssd_multibox_loss.neg_pos_ratio"
        )
        self.wt_loc = 1.0
        self.curr_iter = 0
        self.max_iter = getattr(
            opts, "loss.detection.ssd_multibox_loss.max_monitor_iter"
        )
        self.update_inter = getattr(
            opts, "loss.detection.ssd_multibox_loss.update_wt_freq"
        )
        self.is_master = is_master(opts)
        self.label_smoothing = getattr(
            opts, "loss.detection.ssd_multibox_loss.label_smoothing"
        )
        if not (0.0 <= self.label_smoothing < 1.0):
            logger.error(
                "The value of --loss.detection.ssd-multibox-loss.label-smoothing should be between 0 and 1. "
                "Got: {}".format(self.label_smoothing)
            )

        # Add default value to run CI/CD smoothly
        self.is_distributed = getattr(opts, "ddp.use_distributed", False)

        self.reset_unscaled_loss_values()

    def reset_unscaled_loss_values(self) -> None:
        """Reset the unscaled coefficients for confidence and regression losses to small values"""
        # initialize with very small float values
        self.unscaled_conf_loss = 1e-7
        self.unscaled_reg_loss = 1e-7

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != SSDLoss:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser

        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--loss.detection.ssd-multibox-loss.neg-pos-ratio",
            type=int,
            default=3,
            help=f"Negative positive ratio in {cls.__name__}. Defaults to 3.",
        )
        group.add_argument(
            "--loss.detection.ssd-multibox-loss.max-monitor-iter",
            type=int,
            default=-1,
            help=f"Number of iterations for monitoring location and "
            f"classification loss in {cls.__name__}. -1 means do not monitor. "
            f"Defaults to -1.",
        )
        group.add_argument(
            "--loss.detection.ssd-multibox-loss.update-wt-freq",
            type=int,
            default=200,
            help=f"Update the weights after N number of iterations in {cls.__name__}. "
            f"Defaults to 200 iterations.",
        )
        group.add_argument(
            "--loss.detection.ssd-multibox-loss.label-smoothing",
            type=float,
            default=0.0,
            help=f"Specifies the amount of smoothing when computing the classification loss in {cls.__name__}, "
            f"where 0.0 means no smoothing. Defaults to 0.0.",
        )
        return parser

    def forward(
        self,
        input_sample: Any,
        prediction: Dict[str, Tensor],
        target: Dict[str, Tensor],
        *args,
        **kwargs,
    ) -> Dict[str, Tensor]:
        """
        Compute the SSD Loss

        Args:
            input_sample: Input image tensor to the model.
            prediction: Model output. It is a mapping of the form (string: Tensor) containing two
                mandatory keys, i.e., scores and boxes
            target: Ground truth labels. It is a mapping of the form (string: Tensor) containing two
                mandatory keys, i.e., box_labels and box_coordinates.
        Shape:
            input_sample: This loss function does not care about input to the model.
            prediction["scores"]: Shape is [Batch size, number of anchors, number of classes]
            prediction["boxes"]: Shape is [Batch size, number of anchors, 4] where 4 is the number of box coordinates

            target["box_labels"]: Shape is [Batch size, number of anchors]
            target["box_coordinates"]: Shape is [Batch size, number of anchors, 4]

        Returns:
            A mapping of (string: scalar) is returned. Output contains following keys: (total_loss, reg_loss, cls_loss).
        """

        if not {"scores", "boxes"}.issubset(prediction.keys()):
            logger.error(
                f"scores and boxes are mandatory keys for model's output in {self.__class__.__name__}."
            )

        if not {"box_labels", "box_coordinates"}.issubset(target.keys()):
            logger.error(
                f"box_labels and box_coordinates are mandatory keys for ground truth in {self.__class__.__name__}."
            )

        confidence = prediction["scores"]
        predicted_locations = prediction["boxes"]

        gt_labels = target["box_labels"]
        gt_locations = target["box_coordinates"]

        num_classes = confidence.shape[-1]
        num_coordinates = predicted_locations.shape[-1]

        pos_mask = gt_labels > 0
        predicted_locations = predicted_locations[pos_mask].reshape(-1, num_coordinates)
        gt_locations = gt_locations[pos_mask].reshape(-1, num_coordinates)
        num_pos = max(1, gt_locations.shape[0])
        smooth_l1_loss = F.smooth_l1_loss(
            predicted_locations, gt_locations, reduction="sum"
        )

        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, gt_labels, self.neg_pos_ratio)

        confidence = confidence[mask, :]
        label_smoothing = self.label_smoothing if self.training else 0.0
        classification_loss = F.cross_entropy(
            input=confidence.reshape(-1, num_classes),
            target=gt_labels[mask],
            reduction="sum",
            label_smoothing=label_smoothing,
        )

        if self.curr_iter <= self.max_iter and self.training:
            # classification loss may dominate localization loss or vice-versa
            # Therefore, to ensure that their contributions are equal towards total loss, we scale regression loss.
            # If classification loss contribution is less (or more), then scaling factor will be < 1 (or > 1)
            self.unscaled_conf_loss += tensor_to_python_float(
                classification_loss, is_distributed=self.is_distributed
            )
            self.unscaled_reg_loss += tensor_to_python_float(
                smooth_l1_loss, is_distributed=self.is_distributed
            )

            if (
                (self.curr_iter + 1) % self.update_inter == 0
            ) or self.curr_iter == self.max_iter:
                # weight value before update
                before_update = tensor_to_python_float(
                    self.wt_loc, is_distributed=self.is_distributed
                )
                before_update = round(before_update, 4)
                # update the weight value
                self.wt_loc = self.unscaled_conf_loss / self.unscaled_reg_loss
                self.reset_unscaled_loss_values()

                if self.is_master:
                    # weight value after update
                    after_update = tensor_to_python_float(
                        self.wt_loc, is_distributed=self.is_distributed
                    )
                    after_update = round(after_update, 4)
                    logger.log(
                        f"Updating localization loss multiplier from {before_update} to {after_update}"
                    )

            self.curr_iter += 1

        if self.training and self.wt_loc > 0.0:
            smooth_l1_loss = smooth_l1_loss * self.wt_loc

        ssd_loss = (smooth_l1_loss + classification_loss) / num_pos
        return {
            "total_loss": ssd_loss,
            "reg_loss": smooth_l1_loss / num_pos,
            "cls_loss": classification_loss / num_pos,
        }

    def extra_repr(self) -> str:
        return (
            f"\n\t neg_pos_ratio={self.neg_pos_ratio}"
            f"\n\t box_loss=SmoothL1"
            f"\n\t class_loss=CrossEntropy"
            f"\n\t self_weighting={True if self.max_iter > 0 else False}"
        )
