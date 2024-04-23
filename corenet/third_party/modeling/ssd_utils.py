#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import math
from typing import Optional, Tuple

import torch
from torch import Tensor

"""
This source code in this file is adapted from following repos, both of which are released under MIT license.
    Repository Link:
        https://github.com/sacmehta/EdgeNets
        https://github.com/qfgaohao/pytorch-ssd
    File Link:
        https://github.com/sacmehta/EdgeNets/blob/master/model/detection/match_priors.py
"""


def assign_priors(
    gt_boxes: Tensor,
    gt_labels: Tensor,
    corner_form_priors: Tensor,
    iou_threshold: float,
    background_id: Optional[int] = 0,
    *args,
    **kwargs
) -> Tuple[Tensor, Tensor]:
    """
    Assign ground truth boxes and targets to priors (or anchors)

    Args:
        gt_boxes (Tensor): Ground-truth boxes tensor of shape (num_targets, 4)
        gt_labels (Tensor): Ground-truth labels of shape (num_targets)
        corner_form_priors (Tensor): Priors in corner form and has shape (num_priors, 4)
        iou_threshold (float): Overlap between priors and gt_boxes.
        background_id (int): Background class index. Default: 0

    Returns:
        boxes (Tensor): Boxes mapped to priors and has shape (num_priors, 4)
        labels (Tensor): Labels for mapped boxes and has shape (num_priors)
    """

    if gt_labels.nelement() == 0:
        # Images may not have any labels
        dev = corner_form_priors.device
        gt_boxes = torch.zeros((1, 4), dtype=torch.float32, device=dev)
        gt_labels = torch.zeros(1, dtype=torch.int64, device=dev)

    ious = box_iou(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))

    # size: num_priors
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: num_targets
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index
    # 2.0 is used to make sure every target has a prior assigned
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # size: num_priors
    labels = gt_labels[best_target_per_prior_index]
    labels[best_target_per_prior < iou_threshold] = background_id
    boxes = gt_boxes[best_target_per_prior_index]
    return boxes, labels


def box_iou(
    boxes0: Tensor, boxes1: Tensor, eps: Optional[float] = 1e-5, *args, **kwargs
) -> Tensor:
    """
    Computes intersection-over-union between two boxes
    Args:
        boxes0 (Tensor): Boxes 0 of shape (N, 4)
        boxes1 (Tensor): Boxes 1 of shape (N or 1, 4)
        eps (Optional[float]): A small value is added to denominator for numerical stability

    Returns:
        iou (Tensor): IoU values between boxes0 and boxes1 and has shape (N)
    """

    def area_of(left_top, right_bottom) -> torch.Tensor:
        """
        Given two corners of the rectangle, compute the area
        Args:
            left_top (N, 2): left top corner.
            right_bottom (N, 2): right bottom corner.
        Returns:
            area (N): return the area.
        """
        hw = torch.clamp(right_bottom - left_top, min=0.0)
        return hw[..., 0] * hw[..., 1]

    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_negative_mining(
    loss: Tensor, labels: Tensor, neg_pos_ratio: int, *args, **kwargs
) -> Tensor:
    """
    This function is used to suppress the presence of a large number of negative predictions. For any example/image,
    it keeps all the positive predictions and cut the number of negative predictions to make sure the ratio
    between the negative examples and positive examples is no more than the given ratio for an image.
    Args:
        loss (Tensor): the loss for each example and has shape (N, num_priors).
        labels (Tensor): the labels and has shape (N, num_priors).
        neg_pos_ratio (int):  the ratio between the negative examples and positive examples. Usually, it is set as 3.

    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask
