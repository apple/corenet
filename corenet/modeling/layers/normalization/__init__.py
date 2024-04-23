#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import importlib
import os
from typing import Optional

import torch

from corenet.modeling.layers.identity import Identity
from corenet.utils import logger

SUPPORTED_NORM_FNS = []
NORM_LAYER_REGISTRY = {}
NORM_LAYER_CLS = []


def register_norm_fn(name):
    def register_fn(cls):
        if name in SUPPORTED_NORM_FNS:
            raise ValueError(
                "Cannot register duplicate normalization function ({})".format(name)
            )
        SUPPORTED_NORM_FNS.append(name)
        NORM_LAYER_REGISTRY[name] = cls
        NORM_LAYER_CLS.append(cls)
        return cls

    return register_fn


def build_normalization_layer(
    opts: argparse.Namespace,
    num_features: int,
    norm_type: Optional[str] = None,
    num_groups: Optional[int] = None,
    momentum: Optional[float] = None,
) -> torch.nn.Module:
    """
    Helper function to build the normalization layer.
    The function can be used in either of below mentioned ways:
    Scenario 1: Set the default normalization layers using command line arguments. This is useful when the same normalization
    layer is used for the entire network (e.g., ResNet).
    Scenario 2: Network uses different normalization layers. In that case, we can override the default normalization
    layer by specifying the name using `norm_type` argument.
    """
    if norm_type is None:
        norm_type = getattr(opts, "model.normalization.name")
    if num_groups is None:
        num_groups = getattr(opts, "model.normalization.groups")
    if momentum is None:
        momentum = getattr(opts, "model.normalization.momentum")

    norm_layer = None
    norm_type = norm_type.lower()

    if norm_type in NORM_LAYER_REGISTRY:
        # For detecting non-cuda envs, we do not use torch.cuda.device_count() < 1
        # condition because tests always use CPU, even if cuda device is available.
        # Otherwise, we will get "ValueError: SyncBatchNorm expected input tensor to be
        # on GPU" Error when running tests on a cuda-enabled node (usually linux).
        #
        # Note: We provide default value for getattr(opts, ...) because the configs may
        # be missing "dev.device" attribute in the test env.
        if (
            "cuda" not in str(getattr(opts, "dev.device", "cpu"))
            and "sync_batch" in norm_type
        ):
            # for a CPU-device, Sync-batch norm does not work. So, change to batch norm
            norm_type = norm_type.replace("sync_", "")
        norm_layer = NORM_LAYER_REGISTRY[norm_type](
            normalized_shape=num_features,
            num_features=num_features,
            momentum=momentum,
            num_groups=num_groups,
        )
    elif norm_type == "identity":
        norm_layer = Identity()
    else:
        logger.error(
            "Supported normalization layer arguments are: {}. Got: {}".format(
                SUPPORTED_NORM_FNS, norm_type
            )
        )
    return norm_layer


def arguments_norm_layers(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Normalization layers", description="Normalization layers"
    )

    group.add_argument(
        "--model.normalization.name",
        default="batch_norm",
        type=str,
        help="Normalization layer. Defaults to 'batch_norm'.",
    )
    group.add_argument(
        "--model.normalization.groups",
        default=1,
        type=str,
        help="Number of groups in group normalization layer. Defaults to 1.",
    )
    group.add_argument(
        "--model.normalization.momentum",
        default=0.1,
        type=float,
        help="Momentum in normalization layers. Defaults to 0.1",
    )

    return parser


# automatically import different normalization layers
norm_dir = os.path.dirname(__file__)
for file in os.listdir(norm_dir):
    path = os.path.join(norm_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        model_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module(
            "corenet.modeling.layers.normalization." + model_name
        )
