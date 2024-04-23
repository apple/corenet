#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from torch import Tensor, nn

from corenet.modeling.layers.activation import register_act_fn


@register_act_fn(name="gelu")
class GELU(nn.GELU):
    """
    Applies the `Gaussian Error Linear Units <https://arxiv.org/abs/1606.08415>`_ function
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
