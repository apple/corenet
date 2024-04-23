#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

from torch import Tensor, nn
from torch.nn import functional as F

from corenet.modeling.layers.activation import register_act_fn


@register_act_fn(name="hard_swish")
class Hardswish(nn.Hardswish):
    """
    Applies the HardSwish function, as described in the paper
    `Searching for MobileNetv3 <https://arxiv.org/abs/1905.02244>`_
    """

    def __init__(self, inplace: Optional[bool] = False, *args, **kwargs) -> None:
        super().__init__(inplace=inplace)

    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        if hasattr(F, "hardswish"):
            return F.hardswish(input, self.inplace)
        else:
            x_hard_sig = F.relu(input + 3) / 6
            return input * x_hard_sig
