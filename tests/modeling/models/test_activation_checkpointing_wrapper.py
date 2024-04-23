#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#


from typing import List, Tuple, Union

import pytest
import torch
from torch import nn

from corenet.utils.activation_checkpointing_wrapper import activation_checkpointing


@pytest.mark.parametrize(
    "activation_checkpointing_module_and_count",
    [
        # _checkpoint_wrapped_module is added for each trainable parameter (e.g., weight and bias) in a layer.
        (nn.Linear, 3),
        (nn.Conv1d, 2),
        ([nn.Linear, nn.Conv1d], 5),
    ],
)
def test_activation_checkpointing(
    activation_checkpointing_module_and_count: Tuple[
        Union[torch.nn.Module, List[torch.nn.Module]], int
    ]
):

    (
        activation_checkpoint_module,
        expected_activation_checkpoinitng_layers,
    ) = activation_checkpointing_module_and_count
    # dummy model
    model = torch.nn.Sequential(
        nn.Linear(10, 10, bias=False),
        nn.Conv1d(10, 10, kernel_size=1),
        nn.Linear(10, 10),
        nn.AvgPool1d(kernel_size=1),
    )

    activation_checkpointing(model, submodule_class=activation_checkpoint_module)

    num_ckpt_modules = 0
    for p_name, _ in model.named_parameters():
        if p_name.find("_checkpoint_wrapped_module") > -1:
            num_ckpt_modules += 1

    assert num_ckpt_modules == expected_activation_checkpoinitng_layers
