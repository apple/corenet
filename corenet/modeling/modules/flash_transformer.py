#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import math
from typing import Optional

from torch import Tensor, nn

from corenet.modeling.layers import (
    Dropout,
    FlashMultiHeadSelfAttention,
    Identity,
    LinearLayer,
    StochasticDepth,
    get_normalization_layer,
)
from corenet.modeling.layers.activation import build_activation_layer
from corenet.modeling.modules import BaseModule
from corenet.utils import logger


class FlashTransformerEncoder(BaseModule):
    """Pre-norm `Transformer encoder <https://arxiv.org/abs/1706.03762>`_ with `flash attention <https://arxiv.org/abs/2205.14135>`_.

    Args:
        opts: Command line arguments.
        in_features: Number of channels in the input.
        head_dim: Head dimension in multi-head attention.
        attn_dropout_prob: Dropout rate for attention in multi-head attention. Default: 0.0
        qkv_features: Number of features after linear projection in QKV branch in multi-head
            attention. If none, qkv_features=in_features. Defaults to None.
        bias: Use bias. Defaults to False.
        dropout: Dropout rate. Defaults to 0.0.
        ffn_dropout: Dropout between FFN layers. Defaults to 0.0.
        ffn_multiplier: Multiplier for controling the width in Feed-forward network (FFN). Defaults to 4.0.
        stochastic_dropout: Stochastic dropout setting. Default: 0.0.
        norm_layer_name: Normalization layer name. Defaults to "layer_norm".
        divisible_by: Ensure that FFN dimensions are divisible by this factor. Defaults to 16.

    ...note:
        * Enabling 'stochastic dropout' by setting 0 < stochastic_dropout < 1 drops the multi-head attention and feed-forward network
        layers, thus reducing the depth of the network during training. This is also known as `stochastic depth <https://arxiv.org/abs/1603.09382v3>`_.
        On the other hand, 'dropout' drops the activations and do not change the depth of the network.

        * 'dropout', 'ffn_dropout', and 'stochastic_dropout' allows to address over-fitting issue. The values of these parameters
         are dependent on a task and should be chosen empirically.
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        in_features: int,
        head_dim: int,
        attn_dropout_prob: float = 0.0,
        qkv_features: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        ffn_multiplier: float = 4.0,
        stochastic_dropout: float = 0.0,
        norm_layer_name: str = "layer_norm",
        divisible_by: int = 16,
    ) -> None:

        super().__init__()
        attn_unit = FlashMultiHeadSelfAttention(
            in_features=in_features,
            head_dim=head_dim,
            attn_dropout_prob=attn_dropout_prob,
            qkv_features=qkv_features,
            bias=bias,
        )

        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=norm_layer_name, num_features=in_features
            ),
            attn_unit,
            Dropout(p=dropout),
        )

        ffn_dim = int(
            math.ceil(in_features * ffn_multiplier / divisible_by) * divisible_by
        )
        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=norm_layer_name, num_features=in_features
            ),
            LinearLayer(in_features=in_features, out_features=ffn_dim, bias=bias),
            build_activation_layer(opts, num_parameters=1),
            Dropout(p=ffn_dropout),
            LinearLayer(in_features=ffn_dim, out_features=in_features, bias=bias),
            Dropout(p=dropout),
        )

        self.drop_path = Identity()
        if stochastic_dropout > 0.0:
            if dropout > 0.0:
                logger.error(
                    "Stochastic dropout and dropout are mutually exclusive. "
                    "Use either of them, but not both."
                    "Got: {} and {}".format(stochastic_dropout, dropout)
                )
            self.drop_path = StochasticDepth(p=stochastic_dropout, mode="row")

    def forward(self, x: Tensor) -> Tensor:
        """Forward function.

        Args:
            x: Input tensor of size :math:`(N, S, d)` where :math:`N` is batch size, :math:`S` is the sequence length,
        and :math:`d` is input embedding dim.

        Returns:
            Output tensor of the size as the input.
        """
        x = x + self.drop_path(self.pre_norm_mha(x))
        x = x + self.drop_path(self.pre_norm_ffn(x))
        return x
