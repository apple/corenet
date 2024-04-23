#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
from torch import Tensor


def compute_class_weights(
    target: Tensor, n_classes: int, norm_val: float = 1.1
) -> Tensor:
    """Implementation of a class-weighting scheme, as defined in Section 5.2
    of `ENet <https://arxiv.org/pdf/1606.02147.pdf>`_ paper.

    Args:
        target: Tensor of shape [Batch_size, *] containing values in the range `[0, C)`.
        n_classes: Integer specifying the number of classes :math:`C`
        norm_val: Normalization value. Defaults to 1.1. This value is decided based on the
        `ESPNetv2 paper <https://arxiv.org/abs/1811.11431>`_.
        Link: https://github.com/sacmehta/ESPNetv2/blob/b78e323039908f31347d8ca17f49d5502ef1a594/segmentation/loadData.py#L16

    Returns:
        A :math:`C`-dimensional tensor containing class weights
    """

    class_hist = torch.histc(target.float(), bins=n_classes, min=0, max=n_classes - 1)
    mask_indices = class_hist == 0

    # normalize between 0 and 1 by dividing by the sum
    norm_hist = torch.div(class_hist, class_hist.sum())
    norm_hist = torch.add(norm_hist, norm_val)

    # compute class weights.
    # samples with more frequency will have less weight and vice-versa
    class_wts = torch.div(torch.ones_like(class_hist), torch.log(norm_hist))

    # mask the classes which do not have samples in the current batch
    class_wts[mask_indices] = 0.0

    return class_wts.to(device=target.device)
