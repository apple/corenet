#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict, Optional

from torch import nn

from corenet.modeling.layers import LinearLayer
from corenet.modeling.misc.common import parameter_list


class BaseImageProjectionHead(nn.Module):
    """Base class that projects image representations to the same space as text representations"""

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__()

        self.lr_mult = getattr(opts, "model.image_projection_head.lr_multiplier", 1.0)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model specific arguments"""
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--model.image-projection-head.name",
            type=str,
            default=None,
            help="Name of the image projection head",
        )

        group.add_argument(
            "--model.image-projection-head.lr-multiplier",
            type=float,
            default=1.0,
            help="LR multiplier for image projection head",
        )

        return parser

    def reset_parameters(self) -> None:
        """Reset weights of a given layer"""
        raise NotImplementedError

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs,
    ):
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
        )
        return param_list, [self.lr_mult] * len(param_list)

    def forward(self, input: Dict, *args, **kwargs) -> Dict:
        raise NotImplementedError


def get_in_feature_dimension(image_classifier: nn.Module) -> int:
    """Return the input feature dimension to the image classification head."""
    in_features = None
    if isinstance(image_classifier, nn.Sequential):
        # Classifier that uses nn.Sequential usually has global pooling and
        # multiple linear layers. Find the first linear layer and get its
        # in_features
        for layer in image_classifier:
            if isinstance(layer, (nn.Linear, LinearLayer)):
                in_features = layer.in_features
                break
    elif isinstance(image_classifier, (nn.Linear, LinearLayer)):
        in_features = image_classifier.in_features

    if in_features is None:
        raise NotImplementedError(
            f"Cannot get input feature dimension of {image_classifier}."
        )

    return in_features
