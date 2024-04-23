#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from __future__ import annotations

import argparse
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import torch
from torch import nn

from corenet.modeling.layers import norm_layers_tuple
from corenet.modeling.misc.common import get_tensor_sizes, parameter_list
from corenet.modeling.misc.init_utils import initialize_weights
from corenet.options.parse_args import JsonValidator
from corenet.third_party.modeling import lora
from corenet.utils import logger
from corenet.utils.common_utils import check_frozen_norm_layer
from corenet.utils.ddp_utils import is_master


class BaseAnyNNModel(nn.Module):
    """Base class for any neural network"""

    def __init__(self, opts, *args, **kwargs) -> None:
        super().__init__()
        self.opts = opts

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model-specific arguments"""

        if cls != BaseAnyNNModel:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.resume-exclude-scopes",
            type=str,
            default="",
            help="Comma-separated list of parameter scopes (regex strings) to exclude when "
            "loading a pre-trained model",
        )
        group.add_argument(
            "--model.ignore-missing-scopes",
            type=str,
            default="",
            help="Comma-separated list of parameter scopes (regex strings) to ignore "
            "if they are missing from the pre-training model",
        )
        group.add_argument(
            "--model.rename-scopes-map",
            type=JsonValidator(List[List[str]]),
            default=None,
            help="A mapping from checkpoint variable names to match the existing model names."
            " The mapping is represented as a List[List[str]], e.g. [['before', 'after'], ['this', 'that']]."
            " Note: only loading from Yaml file is supported for this argument.",
        )
        group.add_argument(
            "--model.freeze-modules",
            type=str,
            default="",
            help="Comma-separated list of parameter scopes (regex strings) to freeze.",
        )

        group.add_argument(
            "--model.activation-checkpointing",
            action="store_true",
            help="If enabled, layer specified in model using 'get_activation_checkpoint_submodule_class' would be used \
                for activation checkpointing (a.k.a. gradient checkpointing).",
        )

        group.add_argument(
            "--model.lora.config",
            type=JsonValidator(List[Dict[str, Any]]),
            help="A json-formatted configuration for the LoRA model. See corenet/modeling/models/language_modeling/lora.py for details.",
        )
        group.add_argument(
            "--model.lora.use-lora",
            action="store_true",
            help="If set, use LoRA for the model. Note that parameters are "
            "not automatically frozen. They must be frozen with "
            "--model.freeze-modules.",
        )
        return parser

    def reset_parameters(self, opts: argparse.Namespace) -> None:
        """Initialize model weights"""
        initialize_weights(opts=opts, modules=self.modules())

    def forward(self, x: Any, *args, **kwargs) -> Any:
        """Implement the model-specific forward function in sub-classes."""
        raise NotImplementedError

    def _apply_layer_wise_lr(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs,
    ):
        """This function can be used to adjust the learning rate of each layer in a model. The
        functionality of this function may vary from model to model, so we do not implement
        it in the base class and expects child model classes will implement this function, if desired.
        """
        raise NotImplementedError(
            f"Please implement _apply_layer_wise_lr function for {type(self).__name__}"
        )

    def get_trainable_parameters(
        self,
        weight_decay: float = 0.0,
        no_decay_bn_filter_bias: bool = False,
        module_name: str = "",
        *args,
        **kwargs,
    ) -> Tuple[List[Mapping], List[float]]:
        """Get parameters for training along with the learning rate.

        Args:
            weight_decay: weight decay
            no_decay_bn_filter_bias: Do not decay BN and biases. Defaults to False.

        Returns:
             Returns a tuple of length 2. The first entry is a list of dictionary with three keys
             (params, weight_decay, param_names). The second entry is a list of floats containing
             learning rate for each parameter.

        Note:
            Kwargs may contain module_name. To avoid multiple arguments with the same name,
            we pop it and concatenate with encoder or head name
        """
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            module_name=module_name,
            *args,
            **kwargs,
        )
        return param_list, [1.0] * len(param_list)

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        """Create dummy input and labels for CI/CD purposes. Child classes should implement it."""
        raise NotImplementedError(
            f"Please implement dummy_input_and_label function for {type(self).__name__}"
        )

    def get_exportable_model(self) -> nn.Module:
        """This function can be used to prepare the architecture for inference. For example,
        re-parameterizing branches when possible. The functionality of this method may vary
        from model to model, so child model classes have to implement this method, if such a
        transformation exists.
        """
        return self

    @classmethod
    def freeze_norm_layers(
        cls, opts: argparse.Namespace, model: BaseAnyNNModel
    ) -> None:
        """Freeze normalization layers in the model

        Args:
            opts: Command-line arguments
            model: An instance of `BaseAnyNNModel`
        """
        is_maseter_node = is_master(opts)
        for m in model.modules():
            if isinstance(m, norm_layers_tuple):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False

        # check if layers are frozen or not
        frozen_state, count_norm = check_frozen_norm_layer(model)
        if count_norm > 0 and frozen_state and is_maseter_node:
            logger.error(
                "Something is wrong while freezing normalization layers. Please check"
            )

        if is_maseter_node:
            logger.log("Normalization layers are frozen")

    @classmethod
    def build_model(cls, opts: argparse.Namespace, *args, **kwargs) -> BaseAnyNNModel:
        """Build a model from command-line arguments. Sub-classes must implement this
        method.


        Args:
            opts: Command-line arguments

        ...note::
            This class method is useful when we have a common logic for constructing the
            abstract class for a given task. The concrete sub-classes typically don't
            override this method.
        """
        return cls(opts, *args, **kwargs)

    @torch.no_grad()
    def info(self) -> None:
        """Prints model, parameters, and FLOPs on start rank."""

        train_mode = False
        if self.training:
            # do profiling in eval mode
            train_mode = True
            self.eval()

        logger.log(logger.color_text("Model"))
        # print the model skeleton
        print(self)

        logger.double_dash_line(dashes=65)
        print("{:>35} Summary".format(self.__class__.__name__))
        logger.double_dash_line(dashes=65)

        # compute the network parameters
        overall_params = sum([p.numel() for p in self.parameters()])
        overall_trainable_params = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        print("{:<20} = {:>8.3f} M".format("Total parameters", overall_params / 1e6))
        print(
            "{:<20} = {:>8.3f} M\n".format(
                "Total trainable parameters", overall_trainable_params / 1e6
            )
        )

        try:
            # Compute FLOPs using FVCore
            try:
                input_fvcore = self.dummy_input_and_label(batch_size=1)["samples"]
            except NotImplementedError:
                logger.warning(
                    "Profiling not available, dummy_input_and_label not implemented for this model."
                )
                return

            # compute flops using FVCore
            from fvcore.nn import FlopCountAnalysis, flop_count_table

            flop_analyzer = FlopCountAnalysis(self, input_fvcore)
            logger.log(f"FVCore Analysis:")
            # print input sizes
            input_sizes = get_tensor_sizes(input_fvcore)
            logger.log("Input sizes: {}".format(input_sizes))
            print(flop_count_table(flop_analyzer))

            logger.warning(
                "\n** Please be cautious when using the results in papers. "
                "Certain operations may or may not be accounted in FLOP computation in FVCore. "
                "Therefore, you want to manually ensure that FLOP computation is correct."
            )

            uncalled_modules = flop_analyzer.uncalled_modules()
            if len(uncalled_modules) > 0:
                logger.warning(f"Uncalled Modules:\n{uncalled_modules}")
            else:
                logger.log(f"No uncalled modules detected by fvcore.")

            unsupported_ops = flop_analyzer.unsupported_ops()
            if len(unsupported_ops) > 0:
                logger.warning(f"Unsupported Ops:\n{unsupported_ops}")
            else:
                logger.log(f"No unsupported ops detected by fvcore.")
        except:
            logger.ignore_exception_with_warning(
                "Unable to compute FLOPs using FVCore. Please check"
            )
        logger.double_dash_line(dashes=65)

        if train_mode:
            # switching back to train mode.
            self.train()

    def get_fsdp_wrap_policy(self) -> Optional[Callable[[nn.Module, bool, int], bool]]:
        """Returns the auto wrapping policy for FSDP.

        This is either ``None`` or a callable of a fixed signature.

        If it is ``None``, then ``module`` is wrapped with only a top-level FSDP instance without any nested wrapping.
        If it is a callable, then it should take in three arguments ``module: nn.Module``, ``recurse: bool``, and
        ``nonwrapped_numel: int`` and should return a ``bool`` specifying whether the passed-in ``module`` should be
        wrapped if ``recurse=False`` or if the traversal should continue down the subtree if ``recurse=True``.

        Additional custom arguments may be added to the callable. The ``size_based_auto_wrap_policy``
        in ``torch.distributed.fsdp.wrap.py`` gives an example callable that wraps a module if the parameters
        in its subtree exceed 100M numel. A good practice is to print the model after wrapping and adjust as needed.
        """
        raise NotImplementedError(
            "Wrapping policy to be used with FSDP is not defined."
        )

    def get_activation_checkpoint_submodule_class(self) -> Callable:
        """Returns the sub-module class that needs to be checkpointed.

        Activations of checkpointed sub-module are stored, and recomputed during the backward pass,
        thus providing a trade-off between memory and compute.
        """
        raise NotImplementedError(
            "Activation checkpoint submodule class is not implemented."
        )
