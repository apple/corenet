#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#


from functools import partial
from typing import Callable, List, Union

import torch


def activation_checkpointing(
    model: torch.nn.Module,
    submodule_class: Union[List[Callable], Callable],
) -> None:
    """
    Applies activation checkpointing to `module_to_checkpoint`, a sub-module(s) inside 'model'.

    Args:
        model: The model whose submodules should be wrapped with activation checkpointing.
        submodule_class: Submodule class to be wrapped with activation checkpointing.

    Usage::
        model = nn.Sequential(
                    nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10)
                )
        module_to_checkpoint = nn.Linear
        # checkpoint activations
        activation_checkpointing(model, module_to_checkpoint)
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    if isinstance(submodule_class, list):
        for m in submodule_class:
            check_fn = lambda submodule: isinstance(submodule, m)
            apply_activation_checkpointing(
                model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
            )
    else:
        check_fn = lambda submodule: isinstance(submodule, submodule_class)
        apply_activation_checkpointing(
            model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
        )
