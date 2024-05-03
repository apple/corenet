#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Optional

from torch import nn

from corenet.modeling.layers import GroupLinear, LinearLayer, norm_layers_tuple
from corenet.utils import logger

supported_conv_inits = [
    "kaiming_normal",
    "kaiming_uniform",
    "xavier_normal",
    "xavier_uniform",
    "normal",
    "trunc_normal",
]
supported_fc_inits = [
    "kaiming_normal",
    "kaiming_uniform",
    "xavier_normal",
    "xavier_uniform",
    "normal",
    "trunc_normal",
]


def _init_nn_layers(
    module,
    init_method: Optional[str] = "kaiming_normal",
    std_val: Optional[float] = None,
) -> None:
    """
    Helper function to initialize neural network module
    """
    init_method = init_method.lower()
    if init_method == "kaiming_normal":
        if module.weight is not None:
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "kaiming_uniform":
        if module.weight is not None:
            nn.init.kaiming_uniform_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "xavier_normal":
        if module.weight is not None:
            nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "xavier_uniform":
        if module.weight is not None:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "normal":
        if module.weight is not None:
            std = 1.0 / module.weight.size(1) ** 0.5 if std_val is None else std_val
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif init_method == "trunc_normal":
        if module.weight is not None:
            std = 1.0 / module.weight.size(1) ** 0.5 if std_val is None else std_val
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    else:
        supported_conv_message = "Supported initialization methods are:"
        for i, l in enumerate(supported_conv_inits):
            supported_conv_message += "\n \t {}) {}".format(i, l)
        logger.error("{} \n Got: {}".format(supported_conv_message, init_method))


def initialize_conv_layer(
    module,
    init_method: Optional[str] = "kaiming_normal",
    std_val: Optional[float] = 0.01,
) -> None:
    """Helper function to initialize convolution layers"""
    _init_nn_layers(module=module, init_method=init_method, std_val=std_val)


def initialize_fc_layer(
    module, init_method: Optional[str] = "normal", std_val: Optional[float] = 0.01
) -> None:
    """Helper function to initialize fully-connected layers"""
    if hasattr(module, "layer"):
        _init_nn_layers(module=module.layer, init_method=init_method, std_val=std_val)
    else:
        _init_nn_layers(module=module, init_method=init_method, std_val=std_val)


def initialize_norm_layers(module) -> None:
    """Helper function to initialize normalization layers"""

    def _init_fn(module):
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.ones_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
            nn.init.zeros_(module.bias)

    _init_fn(module.layer) if hasattr(module, "layer") else _init_fn(module=module)


def initialize_weights(opts, modules) -> None:
    """Helper function to initialize different layers in a model"""
    # weight initialization
    conv_init_type = getattr(opts, "model.layer.conv_init", "kaiming_normal")
    linear_init_type = getattr(opts, "model.layer.linear_init", "normal")

    conv_std = getattr(opts, "model.layer.conv_init_std_dev", None)
    linear_std = getattr(opts, "model.layer.linear_init_std_dev", 0.01)
    group_linear_std = getattr(opts, "model.layer.group_linear_init_std_dev", 0.01)

    if isinstance(modules, nn.Sequential):
        for m in modules:
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                initialize_conv_layer(
                    module=m, init_method=conv_init_type, std_val=conv_std
                )
            elif isinstance(m, norm_layers_tuple):
                initialize_norm_layers(module=m)
            elif isinstance(m, (nn.Linear, LinearLayer)):
                initialize_fc_layer(
                    module=m, init_method=linear_init_type, std_val=linear_std
                )
            elif isinstance(m, GroupLinear):
                initialize_fc_layer(
                    module=m, init_method=linear_init_type, std_val=group_linear_std
                )
    else:
        if isinstance(modules, (nn.Conv2d, nn.Conv3d)):
            initialize_conv_layer(
                module=modules, init_method=conv_init_type, std_val=conv_std
            )
        elif isinstance(modules, norm_layers_tuple):
            initialize_norm_layers(module=modules)
        elif isinstance(modules, (nn.Linear, LinearLayer)):
            initialize_fc_layer(
                module=modules, init_method=linear_init_type, std_val=linear_std
            )
        elif isinstance(modules, GroupLinear):
            initialize_fc_layer(
                module=modules, init_method=linear_init_type, std_val=group_linear_std
            )
