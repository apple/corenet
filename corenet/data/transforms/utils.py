#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Any

import numpy as np


def setup_size(size: Any, error_msg="Need a tuple of length 2"):
    if size is None:
        raise ValueError("Size can't be None")

    if isinstance(size, int):
        return size, size
    elif isinstance(size, (list, tuple)) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def intersect(box_a, box_b):
    """Computes the intersection between box_a and box_b"""
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a: np.ndarray, box_b: np.ndarray):
    """
    Computes the intersection of two boxes.
    Args:
        box_a (np.ndarray): Boxes of shape [Num_boxes_A, 4]
        box_b (np.ndarray): Box osf shape [Num_boxes_B, 4]

    Returns:
        intersection over union scores. Shape is [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])  # [A,B]
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]
