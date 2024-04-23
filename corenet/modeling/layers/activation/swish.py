#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

from torch import Tensor, nn

from corenet.modeling.layers.activation import register_act_fn


@register_act_fn(name="swish")
class Swish(nn.SiLU):
    """
    Applies the `Swish (also known as SiLU) <https://arxiv.org/abs/1702.03118>`_ function.
    """

    def __init__(self, inplace: Optional[bool] = False, *args, **kwargs) -> None:
        super().__init__(inplace=inplace)
