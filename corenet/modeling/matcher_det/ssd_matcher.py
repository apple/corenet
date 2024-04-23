#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from corenet.modeling.matcher_det import MATCHER_REGISTRY, BaseMatcher
from corenet.modeling.misc.box_utils import (
    center_form_to_corner_form,
    convert_boxes_to_locations,
    convert_locations_to_boxes,
    corner_form_to_center_form,
)
from corenet.third_party.modeling.ssd_utils import assign_priors
from corenet.utils import logger


@MATCHER_REGISTRY.register(name="ssd")
class SSDMatcher(BaseMatcher):
    """
    This class assigns labels to anchors via `SSD matching process <https://arxiv.org/abs/1512.02325>`_

    Args:
        opts: command line arguments
        bg_class_id: Background class index

    Shape:
        - Input:
            - gt_boxes: Ground-truth boxes in corner form (xyxy format). Shape is :math:`(N, 4)` where :math:`N` is the number of boxes
            - gt_labels: Ground-truth box labels. Shape is :math:`(N)`
            - anchors: Anchor boxes in center form (c_x, c_y, w, h). Shape is :math:`(M, 4)` where :math:`M` is the number of anchors

        - Output:
            - matched_boxes of shape :math:`(M, 4)`
            - matched_box_labels of shape :math:`(M)`
    """

    def __init__(self, opts, bg_class_id: Optional[int] = 0, *args, **kwargs) -> None:
        center_variance = getattr(opts, "matcher.ssd.center_variance", None)
        check_variable(center_variance, "--matcher.ssd.center-variance")

        size_variance = getattr(opts, "matcher.ssd.size_variance", None)
        check_variable(val=size_variance, args_str="--matcher.ssd.size-variance")

        iou_threshold = getattr(opts, "matcher.ssd.iou_threshold", None)
        check_variable(val=iou_threshold, args_str="--matcher.ssd.iou-threshold")

        super().__init__(opts=opts, *args, **kwargs)

        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold
        self.bg_class_id = bg_class_id

    def __repr__(self):
        return "{}(center_variance={}, size_variance={}, iou_threshold={})".format(
            self.__class__.__name__,
            self.center_variance,
            self.size_variance,
            self.iou_threshold,
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """
        Add SSD Matcher specific arguments
        """
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--matcher.ssd.center-variance",
            type=float,
            default=0.1,
            help="Center variance for matching",
        )
        group.add_argument(
            "--matcher.ssd.size-variance",
            type=float,
            default=0.2,
            help="Size variance.",
        )
        group.add_argument(
            "--matcher.ssd.iou-threshold",
            type=float,
            default=0.45,
            help="IOU Threshold.",
        )

        return parser

    def __call__(
        self,
        gt_boxes: Union[np.ndarray, Tensor],
        gt_labels: Union[np.ndarray, Tensor],
        anchors: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if isinstance(gt_boxes, np.ndarray):
            gt_boxes = torch.from_numpy(gt_boxes)
        if isinstance(gt_labels, np.ndarray):
            gt_labels = torch.from_numpy(gt_labels)

        # convert box priors from center [c_x, c_y] to corner_form [x, y]
        anchors_xyxy = center_form_to_corner_form(boxes=anchors)

        matched_boxes_xyxy, matched_labels = assign_priors(
            gt_boxes,  # gt_boxes are in corner form [x, y, w, h]
            gt_labels,
            anchors_xyxy,  # priors are in corner form [x, y, w, h]
            self.iou_threshold,
            background_id=self.bg_class_id,
        )

        # convert the matched boxes to center form [c_x, c_y]
        matched_boxes_cxcywh = corner_form_to_center_form(matched_boxes_xyxy)

        # Eq.(2) in paper https://arxiv.org/pdf/1512.02325.pdf
        boxes_for_regression = convert_boxes_to_locations(
            gt_boxes=matched_boxes_cxcywh,  # center form
            prior_boxes=anchors,  # center form
            center_variance=self.center_variance,
            size_variance=self.size_variance,
        )

        return boxes_for_regression, matched_labels

    def convert_to_boxes(
        self, pred_locations: torch.Tensor, anchors: torch.Tensor
    ) -> Tensor:
        """
        Decodes boxes from predicted locations and anchors.
        """

        # decode boxes in center form
        boxes = convert_locations_to_boxes(
            pred_locations=pred_locations,
            anchor_boxes=anchors,
            center_variance=self.center_variance,
            size_variance=self.size_variance,
        )
        # convert boxes from center form [c_x, c_y] to corner form [x, y]
        boxes = center_form_to_corner_form(boxes)
        return boxes


def check_variable(val, args_str: str):
    if val is None:
        logger.error("{} cannot be None".format(args_str))

    if not (0.0 < val < 1.0):
        logger.error(
            "The value of {} should be between 0 and 1. Got: {}".format(args_str, val)
        )
