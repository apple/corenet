#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any, Dict, List, Optional, Tuple

from torch import nn

from corenet.modeling.misc.common import parameter_list


class BaseLayer(nn.Module):
    """
    Base class for neural network layers. Subclass must implement `forward` function.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add layer specific arguments"""
        return parser

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Tuple[List[Dict], List[float]]:
        """
        Get parameters for training along with the learning rate.

        Args:
            weight_decay: weight decay
            no_decay_bn_filter_bias: Do not decay BN and biases. Defaults to False.

        Returns:
             Returns a tuple of length 2. The first entry is a list of dictionary with three keys
             (params, weight_decay, param_names). The second entry is a list of floats containing
             learning rate for each parameter.

        Note:
            Learning rate multiplier is set to 1.0 here as it is handled inside the Central Model.
        """
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            *args,
            **kwargs
        )
        return param_list, [1.0] * len(param_list)

    def forward(self, *args, **kwargs) -> Any:
        """Forward function."""
        raise NotImplementedError("Sub-classes should implement forward method")

    def __repr__(self):
        return "{}".format(self.__class__.__name__)
