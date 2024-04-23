#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Optional

from corenet.modeling.misc.common import (
    freeze_modules_based_on_opts,
    load_pretrained_model,
)
from corenet.modeling.models.base_model import BaseAnyNNModel
from corenet.modeling.models.fsdp_wrapper import FullyShardedDataParallelWrapper
from corenet.third_party.modeling import lora
from corenet.utils import logger
from corenet.utils.download_utils import get_local_path
from corenet.utils.registry import Registry

MODEL_REGISTRY = Registry(
    registry_name="model_registry",
    base_class=BaseAnyNNModel,
    lazy_load_dirs=["corenet/modeling/models"],
    internal_dirs=["corenet/internal", "corenet/internal/projects/*"],
)


def get_model(
    opts: argparse.Namespace,
    category: Optional[str] = None,
    model_name: Optional[str] = None,
    use_lora: Optional[bool] = None,
    *args,
    **kwargs,
) -> BaseAnyNNModel:
    """Create a task-specific model from command-line arguments. If model category (or task) and name are
    passed as an argument, then they are used. Otherwise, `dataset.category` and `model.{category}.name` are
    read from command-line arguments to read model category and name, respectively.

    Args:
        opts: Command-line arguments
        category: Category or task (e.g., segmentation)
        model_name: Model name for a specific task (e.g., vit for classification)
        use_lora: If False, LoRA will not be used. If True, LoRA will be used. If
            None, the value of @use_lora will be read from @opts.

    Returns:
        An instance of `corenet.modeling.models.BaseAnyNNModel`.
    """

    if category is None:
        category = getattr(opts, "dataset.category")

    if model_name is None:
        model_name = getattr(opts, f"model.{category}.name")

    if model_name == "__base__":
        # __base__ is used to register the task-specific base classes. These classes often
        # provide functionalities that can be re-used by sub-classes, but does not provide
        # task-specific models.
        logger.error(
            f"For {category} task, model name can't be __base__. Please check."
        )

    model = MODEL_REGISTRY[model_name, category].build_model(opts, *args, **kwargs)

    if use_lora is None:
        use_lora = getattr(opts, "model.lora.use_lora")
    if use_lora:
        lora_config = getattr(opts, "model.lora.config")
        lora.add_lora_layers(model, lora_config)

    # for some categories, we do not have pre-trained path (e.g., segmentation_head).
    # Therefore, we need to set the default value.
    pretrained_wts_path = getattr(opts, f"model.{category}.pretrained", None)
    if pretrained_wts_path is not None:
        pretrained_model_path = get_local_path(opts, path=pretrained_wts_path)
        model = load_pretrained_model(
            model=model, wt_loc=pretrained_model_path, opts=opts
        )

    model = freeze_modules_based_on_opts(opts, model)
    return model


def arguments_model(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser = BaseAnyNNModel.add_arguments(parser=parser)
    parser = MODEL_REGISTRY.all_arguments(parser=parser)
    parser = FullyShardedDataParallelWrapper.add_arguments(parser=parser)
    return parser
