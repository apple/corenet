#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

from torch import Tensor, nn
from torch.nn import functional as F

from corenet.modeling.layers.activation import register_act_fn


@register_act_fn(name="hard_sigmoid")
class Hardsigmoid(nn.Hardsigmoid):
    """
    Applies the `Hard Sigmoid <https://arxiv.org/abs/1511.00363v3>`_ function
    """

    def __init__(self, inplace: Optional[bool] = False, *args, **kwargs) -> None:
        super().__init__(inplace=inplace)

    def forward(self, input: Tensor, *args, **kwargs) -> Tensor:
        if hasattr(F, "hardsigmoid"):
            return F.hardsigmoid(input, self.inplace)
        else:
            return F.relu(input + 3) / 6
