#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import importlib
import os
from typing import Optional

import torch.nn

from corenet.utils import logger

SUPPORTED_ACT_FNS = []
ACT_FN_REGISTRY = {}


def register_act_fn(name):
    def register_fn(cls):
        if name in SUPPORTED_ACT_FNS:
            raise ValueError(
                "Cannot register duplicate activation function ({})".format(name)
            )
        SUPPORTED_ACT_FNS.append(name)
        ACT_FN_REGISTRY[name] = cls
        return cls

    return register_fn


def arguments_activation_fn(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Non-linear functions", description="Non-linear functions"
    )

    group.add_argument(
        "--model.activation.name",
        default="relu",
        type=str,
        help="Non-linear function name",
    )
    group.add_argument(
        "--model.activation.inplace",
        action="store_true",
        help="Use non-linear functions inplace",
    )
    group.add_argument(
        "--model.activation.neg-slope",
        default=0.1,
        type=float,
        help="Negative slope in leaky relu function",
    )

    return parser


def build_activation_layer(
    opts: argparse.Namespace,
    act_type: Optional[str] = None,
    inplace: Optional[bool] = None,
    negative_slope: Optional[float] = None,
    num_parameters: int = -1,
) -> torch.nn.Module:
    """
    Helper function to build the activation function. If any of the optional
    arguments are not provided (i.e. None), the corresponding ``model.activation.*``
    config entry will be used as default value.

    Args:
        act_type: Name of the activation layer.
            Default: --model.activation.name config value.
        inplace: If true, operation will be inplace.
            Default: --model.activation.inplace config value.
        negative_slope: Negative slope parameter for leaky_relu.
            Default: --model.activation.neg_slop config value.
    """
    assert isinstance(
        opts, argparse.Namespace
    ), f"Expected first argument to be an argparse.Namespace, but received a {type(opts)}."
    if act_type is None:
        act_type = getattr(opts, "model.activation.name")
    if inplace is None:
        inplace = getattr(opts, "model.activation.inplace")
    if negative_slope is None:
        negative_slope = getattr(opts, "model.activation.neg_slope")

    act_type = act_type.lower()
    act_layer = None
    if act_type in ACT_FN_REGISTRY:
        act_layer = ACT_FN_REGISTRY[act_type](
            num_parameters=num_parameters,
            inplace=inplace,
            negative_slope=negative_slope,
        )
    else:
        logger.error(
            "Supported activation layers are: {}. Supplied argument is: {}".format(
                SUPPORTED_ACT_FNS, act_type
            )
        )
    return act_layer


# automatically import different activation functions
act_dir = os.path.dirname(__file__)
for file in os.listdir(act_dir):
    path = os.path.join(act_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(
            "corenet.modeling.layers.activation." + model_name
        )
