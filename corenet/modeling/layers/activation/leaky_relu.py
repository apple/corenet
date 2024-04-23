#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

from torch import Tensor, nn

from corenet.modeling.layers.activation import register_act_fn


@register_act_fn(name="leaky_relu")
class LeakyReLU(nn.LeakyReLU):
    """
    Applies a leaky relu function. See `Rectifier Nonlinearities Improve Neural Network Acoustic Models`
    for more details.
    """

    def __init__(
        self,
        negative_slope: Optional[float] = 1e-2,
        inplace: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__(negative_slope=negative_slope, inplace=inplace)
