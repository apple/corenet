#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import os
import re
from collections import deque
from types import MethodType
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from corenet.utils import logger
from corenet.utils.common_utils import unwrap_model_fn
from corenet.utils.ddp_utils import is_master, is_start_rank_node


def clean_strip(
    obj: Union[str, List[str]], sep: Optional[str] = ",", strip: bool = True
) -> List[str]:
    # Allowing list of strings as input as well as comma-separated strings
    if isinstance(obj, list):
        strings = obj
    else:
        strings = obj.split(sep)

    if strip:
        strings = [x.strip() for x in strings]
    strings = [x for x in strings if x]
    return strings


def load_pretrained_model(
    model: torch.nn.Module, wt_loc: str, opts: argparse.Namespace, *args, **kwargs
) -> torch.nn.Module:
    """Helper function to load pre-trained weights.
    Args:
        model: Model whose weights will be loaded.
        wt_loc: Path to file to load state_dict from.
        opts: Input arguments.
    Returns:
        The model loaded with the given weights.

    """
    if not os.path.isfile(wt_loc):
        logger.error("Pretrained file is not found here: {}".format(wt_loc))

    wts = torch.load(wt_loc, map_location="cpu")

    is_master_node = is_start_rank_node(opts)

    exclude_scopes = getattr(opts, "model.resume_exclude_scopes", "")
    exclude_scopes: List[str] = clean_strip(exclude_scopes)

    missing_scopes = getattr(opts, "model.ignore_missing_scopes", "")
    missing_scopes: List[str] = clean_strip(missing_scopes)

    rename_scopes_map: List[List[str]] = getattr(opts, "model.rename_scopes_map", [])
    if rename_scopes_map:
        for entry in rename_scopes_map:
            if len(entry) != 2:
                raise ValueError(
                    "Every entry in model.rename_scopes_map must contain exactly two string elements"
                    " for before and after. Got {}.".format(str(entry))
                )

    # By default, adding scopes that we exclude to missing scopes
    # If you excluded something, you can't expect it to be there.
    missing_scopes += exclude_scopes

    # remove unwanted scopes
    if exclude_scopes:
        for key in wts.copy():
            if any([re.match(x, key) for x in exclude_scopes]):
                del wts[key]

    if rename_scopes_map:
        for before, after in rename_scopes_map:
            wts = {re.sub(before, after, key): value for key, value in wts.items()}

    strict = not bool(missing_scopes)

    try:
        module = unwrap_model_fn(model)
        missing_keys, unexpected_keys = module.load_state_dict(wts, strict=strict)

        if unexpected_keys:
            raise Exception(
                "Found unexpected keys: {}."
                "You can ignore these keys using `model.resume_exclude_scopes`.".format(
                    ", ".join(unexpected_keys)
                )
            )

        missing_keys = [
            key
            for key in missing_keys
            if not any([re.match(x, key) for x in missing_scopes])
        ]

        if missing_keys:
            raise Exception(
                "Missing keys detected. Did not find the following keys in pre-trained model: {}."
                " You can ignore the keys using `model.ignore_missing_scopes`.".format(
                    ",".join(missing_keys)
                )
            )

        if is_master_node:
            logger.log("Pretrained weights are loaded from {}".format(wt_loc))
    except Exception as e:
        if is_master_node:
            logger.error(
                "Unable to load pretrained weights from {}. Error: {}".format(wt_loc, e)
            )

    return model


def parameter_list(
    named_parameters,
    weight_decay: Optional[float] = 0.0,
    no_decay_bn_filter_bias: Optional[bool] = False,
    *args,
    **kwargs,
) -> List[Dict]:
    module_name = kwargs.get("module_name", "")
    with_decay = []
    without_decay = []
    with_decay_param_names = []
    without_decay_param_names = []
    if isinstance(named_parameters, list):
        for n_parameter in named_parameters:
            for p_name, param in n_parameter():
                if (
                    param.requires_grad
                    and len(param.shape) == 1
                    and no_decay_bn_filter_bias
                ):
                    # biases and normalization layer parameters are of len 1
                    without_decay.append(param)
                    without_decay_param_names.append(module_name + p_name)
                elif param.requires_grad:
                    with_decay.append(param)
                    with_decay_param_names.append(module_name + p_name)
    else:
        for p_name, param in named_parameters():
            if (
                param.requires_grad
                and len(param.shape) == 1
                and no_decay_bn_filter_bias
            ):
                # biases and normalization layer parameters are of len 1
                without_decay.append(param)
                without_decay_param_names.append(module_name + p_name)
            elif param.requires_grad:
                with_decay.append(param)
                with_decay_param_names.append(module_name + p_name)
    param_list = [
        {
            "params": with_decay,
            "weight_decay": weight_decay,
            "param_names": with_decay_param_names,
        }
    ]
    if len(without_decay) > 0:
        param_list.append(
            {
                "params": without_decay,
                "weight_decay": 0.0,
                "param_names": without_decay_param_names,
            }
        )
    return param_list


def freeze_module(module: torch.nn.Module, force_eval: bool = True) -> torch.nn.Module:
    """
    Sets requires_grad = False on all the given module parameters, and put the module in eval mode.
    By default, it also overrides the module's `train` method to make sure that it always stays in eval mode
    (ie calling ``module.train(mode=True)`` executes ``module.train(mode=False)``)

    >>> module = nn.Linear(10, 20).train()
    >>> module.training
    True
    >>> module.weight.requires_grad
    True
    >>> freeze_module(module).train().training
    False
    >>> module.weight.requires_grad
    False
    """

    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False

    if force_eval:

        def _force_train_in_eval(
            self: torch.nn.Module, mode: bool = True
        ) -> torch.nn.Module:
            # ignore train/eval calls: perpetually stays in eval
            return self

        module.train = MethodType(_force_train_in_eval, module)

    return module

def _module_bfs(module:torch.nn.Module, p: list[str], idx=1) -> None:
            stack = deque()
            stack.append((idx, module))
            
            while stack:
                idx, module = stack.popleft()
                if idx<len(p):
                    for submodule_name, submodule in module.named_children():
                        if re.match(p[idx], submodule_name):
                            if idx == len(p)-1:
                                freeze_module(submodule)
                                logger.info("Freezing module: {} Inside: {}".format(submodule_name,'>'.join(p[:-1])))
                            else:
                                stack.append((idx+1, submodule))


def freeze_modules_based_on_opts(
    opts: argparse.Namespace, model: torch.nn.Module, verbose: bool = True
) -> torch.nn.Module:
    """
    Allows for freezing immediate modules and parameters as well as nested modules of the model using --model.freeze-modules.

    --model.freeze-modules should be a list of strings, a comma-separated list of regex expressions or list of strings with '>' between modules to freeze particular nested layers inside immediate module of the model.

    Examples of --model.freeze-modules:
        "conv.*"  # see example below: can freeze all (top-level) conv layers
        "^((?!classifier).)*$"   # freezes everything except for "classifier": useful for linear probing
        "conv1,layer1,layer2,layer3"  # freeze all layers up to layer3
        "transformer>decoder"  # freeze decoder block inside transformer

    >>> model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1, 20, 5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20, 64, 5)),
          ('relu2', nn.ReLU())
        ]))
    >>> opts = argparse.Namespace(**{"model.freeze_modules": "conv1"})
    >>> _ = freeze_modules_based_on_opts(opts, model)
    INFO    - Freezing module: conv1
    >>> model.train()
    >>> model.conv1.training
    False
    >>> model.conv2.training
    True
    """
    freeze_patterns = getattr(opts, "model.freeze_modules", "")
    freeze_patterns = clean_strip(freeze_patterns)

    verbose = verbose and is_master(opts)

    if freeze_patterns:
        immediate_children_patterns = []
        nested_modules_patterns = []
        # separate nested expressions from the rest
        for p in freeze_patterns:
            if ">" in p:
                nested_modules_patterns.append([part for part in re.split(r'\s*>\s*', p) if part.strip()])
            else:
                immediate_children_patterns.append(p)
        
        for name, module in model.named_children():
            for p in nested_modules_patterns:
                if re.match(p[0], name):
                    _module_bfs(module, p, 1)        
                    
            if any([re.match(p, name) for p in immediate_children_patterns]):
                freeze_module(module)
                if verbose:
                    logger.info("Freezing module: {}".format(name))

        for name, param in model.named_parameters():
            if any([re.match(p, name) for p in immediate_children_patterns]):
                param.requires_grad = False
                if verbose:
                    logger.info("Freezing parameter: {}".format(name))

    if verbose and hasattr(model, "get_trainable_parameters"):
        param_list, _ = model.get_trainable_parameters()
        for params in param_list:
            if (
                not isinstance(params["param_names"], List)
                or not isinstance(params["params"], List)
                or not isinstance(params["weight_decay"], (float, int))
            ):
                param_types = {k: type(v) for k, v in params.items()}
                logger.error(
                    "Expected parameter format: {{ params: List, weight_decay: float, param_names: List }}. "
                    "Got: {}".format(param_types)
                )
        # Flatten all parameter names
        trainable_param_names = [p for x in param_list for p in x["param_names"]]
        logger.info("Trainable parameters: {}".format(trainable_param_names))

    return model


def get_tensor_sizes(data: Union[Dict, Tensor]) -> Union[List[str], List[Tuple[int]]]:
    """Utility function for extracting tensor shapes (for printing purposes only)."""
    if isinstance(data, Dict):
        tensor_sizes = []
        for k, v in data.items():
            size_ = get_tensor_sizes(v)
            if size_:
                tensor_sizes.append(f"{k}: {size_}")
        return tensor_sizes
    elif isinstance(data, Tensor):
        return [*data.shape]
    else:
        return []
