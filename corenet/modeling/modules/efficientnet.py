#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from torch import Tensor, nn

from corenet.modeling.layers import StochasticDepth
from corenet.modeling.modules import InvertedResidualSE


class EfficientNetBlock(InvertedResidualSE):
    """
    This class implements a variant of the inverted residual block with squeeze-excitation unit,
    as described in `MobileNetv3 <https://arxiv.org/abs/1905.02244>`_ paper. This variant
    includes stochastic depth, as used in `EfficientNet <https://arxiv.org/abs/1905.11946>`_ paper.

    Args:
        stochastic_depth_prob: float,
        For other arguments, refer to the parent class.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`
    """

    def __init__(self, stochastic_depth_prob: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stochastic_depth = StochasticDepth(p=stochastic_depth_prob, mode="row")

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        y = self.block(x)
        if self.use_res_connect:
            # Pass the output through the stochastic layer module, potentially zeroing it.
            y = self.stochastic_depth(y)
            # residual connection
            y = y + x
        return y

    def __repr__(self) -> str:
        return (
            super().__repr__()[:-1]
            + f", stochastic_depth_prob={self.stochastic_depth.p})"
        )
