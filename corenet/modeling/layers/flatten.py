#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

from torch import Tensor, nn


class Flatten(nn.Flatten):
    r"""
    This layer flattens a contiguous range of dimensions into a tensor.

    Args:
        start_dim (Optional[int]): first dim to flatten. Default: 1
        end_dim (Optional[int]): last dim to flatten. Default: -1

    Shape:
        - Input: :math:`(*, S_{\text{start}},..., S_{i}, ..., S_{\text{end}}, *)`,'
          where :math:`S_{i}` is the size at dimension :math:`i` and :math:`*` means any
          number of dimensions including none.
        - Output: :math:`(*, \prod_{i=\text{start}}^{\text{end}} S_{i}, *)`.
    """

    def __init__(self, start_dim: Optional[int] = 1, end_dim: Optional[int] = -1):
        super(Flatten, self).__init__(start_dim=start_dim, end_dim=end_dim)
