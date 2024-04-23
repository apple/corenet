#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

from torch import Tensor, nn

from corenet.modeling.layers.activation import register_act_fn


@register_act_fn(name="relu")
class ReLU(nn.ReLU):
    """
    Applies Rectified Linear Unit function
    """

    def __init__(self, inplace: Optional[bool] = False, *args, **kwargs) -> None:
        super().__init__(inplace=inplace)
