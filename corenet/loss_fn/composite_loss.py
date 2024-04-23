#  For licensing see accompanying LICENSE file.
#  Copyright (C) 2024 Apple Inc. All Rights Reserved.

from __future__ import annotations

import argparse
import copy
import json
from typing import Any, List, Mapping, Tuple

from torch import Tensor

from corenet.loss_fn import LOSS_REGISTRY, BaseCriteria, build_loss_fn
from corenet.options.utils import flatten_yaml_as_dict
from corenet.utils import logger


# CompositeLoss can be used with any task. Therefore, we register both name and type
# as the same.
@LOSS_REGISTRY.register(name="composite_loss", type="composite_loss")
class CompositeLoss(BaseCriteria):
    """Combines different loss functions and returns the weighted sum of these losses.
    `loss_category` and `loss_weight` are two mandatory keys that allows us to combine
    different losses and compute their weighted sum. The `loss_category` specifies the category
    of a loss function and is a string (e.g., classification). The `loss_weight` specifies
    the contribution of a loss function and is a float value (e.g., 1.0). The sum of `loss_weight`s
    corresponding to different loss functions is not required to be 1.

    Args:
        opts: command-line arguments

    Example::
    # Example yaml config for combining classification and neural_augmentation loss function is given below.
    # Please note that configuration for each loss function should start with `-` in `composite_loss`.

    loss:
      category: "composite_loss"
      composite_loss:
        - loss_category: "classification"
          loss_weight: 1.0
          classification:
            name: "cross_entropy"
            cross_entropy:
              label_smoothing: 0.1
        - loss_category: "neural_augmentation"
          loss_weight: 1.0
          neural_augmentation:
            perceptual_metric: "psnr"
            target_value: [ 40, 10 ]
            curriculum_method: "cosine"
    """

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:

        (
            task_loss_fn_mapping,
            task_loss_wts_mapping,
        ) = CompositeLoss.build_composite_loss_fn(opts, *args, **kwargs)

        super().__init__(opts, *args, **kwargs)
        self.loss_fns = task_loss_fn_mapping
        self.loss_weights = task_loss_wts_mapping

    @classmethod
    def build_composite_loss_fn(
        cls, opts: argparse.Namespace, *args, **kwargs
    ) -> Tuple[Mapping[str, BaseCriteria], Mapping[str, float]]:
        """Build loss functions from command line arguments and loss registry

        Args:
            opts: command-line arguments

        Returns:
            A tuple of two dictionaries. The first dictionary, task_loss_fn_mapping, contains
            information about loss function category and module. The second dictionary, `task_loss_wts_mapping`
            contains the information about loss function category and weight.
        """
        composite_loss_opts = getattr(opts, "loss.composite_loss")
        if composite_loss_opts is None:
            logger.error(
                f"{cls.__name__} can't be None. Please specify --loss.composite-loss using yaml file"
            )

        if not isinstance(composite_loss_opts, List):
            logger.error(
                f"{cls.__name__} options are expected as a List. "
                f"Got type: {type(composite_loss_opts)} and values: {composite_loss_opts}"
            )

        num_loss_fns = len(composite_loss_opts)
        if num_loss_fns < 1:
            logger.error(f"We need at least one loss function if using {cls.__name__}")

        task_loss_fn_mapping = {}
        task_loss_wts_mapping = {}
        for i, composite_loss_opts_as_dict in enumerate(composite_loss_opts):
            if "loss_category" not in composite_loss_opts_as_dict:
                logger.error("loss_category is a mandatory key")
            if "loss_weight" not in composite_loss_opts_as_dict:
                logger.error("Loss weight is a mandatory")
            loss_category = composite_loss_opts_as_dict.pop("loss_category")

            loss_weight = composite_loss_opts_as_dict.pop("loss_weight")
            if not isinstance(loss_weight, (float, int)):
                logger.error(
                    f"loss weight should be either int or float. "
                    f"Got: value={loss_weight}, type={type(loss_weight)}"
                )

            # flatten the dictionary
            composite_loss_opts_as_dict = flatten_yaml_as_dict(
                composite_loss_opts_as_dict
            )

            # `composite_loss_opts_as_dict` only contains the values of command-line arguments that are
            # defined in the yaml file. Therefore, if a user misses few arguments, we won't have access
            # to default values, leading to an error. To avoid this, we create a local copy of global
            # command-line arguments and update it with `composite_loss_opts_as_dict` arguments
            loss_opts = copy.deepcopy(opts)

            # override the global opts with loss_fn specific opts in local copy
            for k, v in composite_loss_opts_as_dict.items():
                # we need to prefix each argument with loss because we define individual losses as
                # `loss.classification.*` and not `classification.*`
                setattr(loss_opts, "loss." + k, v)

            # given the category of a loss function, build the criteria
            task_loss_fn_mapping[loss_category] = build_loss_fn(
                opts=loss_opts, category=loss_category, *args, **kwargs
            )

            task_loss_wts_mapping[loss_category] = loss_weight

        # see if the keys in task_loss_fn_mapping and task_loss_wts_mapping are the same or not
        # i.e., intersection is null.
        is_intersection = task_loss_fn_mapping.keys().isdisjoint(task_loss_wts_mapping)
        assert is_intersection is False, (
            f"The keys in task_loss_fn_mapping and task_loss_wts_mapping are not the same. "
            f"Got: {task_loss_fn_mapping.keys()} and {task_loss_wts_mapping.keys()}"
        )

        return task_loss_fn_mapping, task_loss_wts_mapping

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add criteria-specific arguments to the parser."""
        if cls != CompositeLoss:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(cls.__name__)

        group.add_argument("--loss.composite-loss", type=json.loads, action="append")
        return parser

    def forward(
        self, input_sample: Any, prediction: Any, target: Any, *args, **kwargs
    ) -> Any:
        """Compute the weighted sum of different loss functions.

        Args:
            input_sample: Input to the model.
            prediction: Model's output
            target: Ground truth labels

        Returns:
            A mapping of the form (str: scalar loss value) with `total_loss` as a mandatory key.
            The other keys corresponds to loss category names and their values contain category-specific
            scalar loss values. total_loss is weighted sum of these category-specific losses.
        """
        outputs = {}
        total_loss = 0.0
        for loss_name, loss_layer in self.loss_fns.items():
            loss_wt = self.loss_weights[loss_name]
            loss_val = loss_layer(
                input_sample=input_sample,
                prediction=prediction,
                target=target,
                *args,
                **kwargs,
            )

            if not isinstance(loss_val, (Tensor, Mapping)):
                logger.error(
                    "Loss value is expected as a scalar or dictionary of scalars with total_loss as a "
                    "mandatory key."
                )

            if isinstance(loss_val, Mapping) and "total_loss" in loss_val:
                loss_val = loss_val["total_loss"]

                if not isinstance(loss_val, Tensor):
                    logger.error(
                        f"Value corresponding to total_loss key in {loss_val} is expected to be scalar."
                        f"Got: {type(loss_val)}"
                    )
            # scale the loss
            loss_val = loss_val * loss_wt

            total_loss += loss_val
            outputs[loss_name] = loss_val
        outputs.update({"total_loss": total_loss})
        return outputs

    def train(self, mode: bool = True) -> None:
        """Sets the loss functions in training mode."""
        for loss_name, loss_layer in self.loss_fns.items():
            loss_layer.train(mode=mode)

    def eval(self) -> None:
        """Sets the loss functions in evaluation mode."""
        for loss_name, loss_layer in self.loss_fns.items():
            loss_layer.eval()

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}(\n\t"
        for k, v in self.loss_fns.items():
            repr_str += (
                v.__repr__()
                .replace("\n\t", " ")
                .replace("\n)", f" loss_wt={self.loss_weights[k]})")
            )
            repr_str += "\n\t"
        repr_str += "\n)"
        return repr_str
