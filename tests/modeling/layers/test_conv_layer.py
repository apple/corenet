#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from pprint import pformat

import pytest
import torch

from corenet.modeling.layers.activation import build_activation_layer
from corenet.modeling.layers.conv_layer import ConvLayer2d
from corenet.modeling.layers.normalization import build_normalization_layer
from tests.configs import get_config


@pytest.mark.parametrize(
    "use_norm,customize_norm", [(True, True), (True, False), (False, False)]
)
@pytest.mark.parametrize(
    "use_act,customize_act", [(True, True), (True, False), (False, False)]
)
def test_use_act_and_use_norm(
    use_act: bool, customize_act: bool, use_norm: bool, customize_norm: bool
) -> None:
    opts = get_config()
    default_act, custom_act = "relu", "gelu"  # Choose different values
    setattr(opts, "model.activation.name", default_act)
    default_norm, custom_norm = "layer_norm", "batch_norm"  # Choose different values
    setattr(opts, "model.normalization.name", default_norm)

    conv = ConvLayer2d(
        opts=opts,
        in_channels=3,
        out_channels=5,
        kernel_size=3,
        stride=3,
        use_act=use_act,
        act_layer=(
            build_activation_layer(opts, act_type=custom_act) if customize_act else None
        ),
        use_norm=use_norm,
        norm_layer=(
            build_normalization_layer(opts, norm_type=custom_norm, num_features=3)
            if customize_norm
            else None
        ),
    )

    def has_submodule(name: str) -> bool:
        name = name.lower().replace("_", "")
        for module in conv.block.children():
            if name in type(module).__name__.lower().replace("_", ""):
                return True
        return False

    has_conv2d = has_submodule("conv2d")
    has_default_norm = has_submodule(default_norm)
    has_custom_norm = has_submodule(custom_norm)
    has_default_act = has_submodule(default_act)
    has_custom_act = has_submodule(custom_act)

    assert len(list(conv.block.children())) == sum(
        map(
            int,
            [
                has_conv2d,
                has_default_act,
                has_default_norm,
                has_custom_norm,
                has_custom_act,
            ],
        )
    ), (
        "Got duplicate or unexpected submodules in"
        f" {pformat(list(conv.block.children()))}."
    )

    assert has_conv2d
    assert has_default_norm == (use_norm and not customize_norm)
    assert has_custom_norm == (use_norm and customize_norm)
    assert has_default_act == (use_act and not customize_act)
    assert has_custom_act == (use_act and customize_act)
