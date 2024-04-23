#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

from torch import Tensor, nn

from corenet.modeling.layers.activation import register_act_fn


@register_act_fn(name="prelu")
class PReLU(nn.PReLU):
    """
    Applies the `Parametric Rectified Linear Unit <https://arxiv.org/abs/1502.01852>`_ function
    """

    def __init__(
        self,
        num_parameters: Optional[int] = 1,
        init: Optional[float] = 0.25,
        *args,
        **kwargs
    ) -> None:
        super().__init__(num_parameters=num_parameters, init=init)
