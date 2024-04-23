#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Any, Callable, Dict, Optional

import torch
from torch import Tensor, nn

from corenet.modeling import parameter_list
from corenet.modeling.layers import norm_layers_tuple
from corenet.modeling.misc.init_utils import initialize_weights
from corenet.utils.ddp_utils import is_master


class BaseTextEncoder(nn.Module):
    """Base class for text encoder"""

    def __init__(self, opts, projection_dim: int, *args, **kwargs) -> None:
        is_master_node = is_master(opts)
        super(BaseTextEncoder, self).__init__()
        self.opts = opts
        self.projection_dim = projection_dim
        self.is_master_node = is_master_node

    @property
    def vocab_size(self):
        vocab_size = getattr(self.opts, "model.text.vocab_size")
        assert (
            vocab_size is not None
        ), "Vocab size can't be None. Please specify 'model.text.vocab_size' argument."
        return vocab_size

    @property
    def context_length(self):
        context_length = getattr(self.opts, "model.text.context_length")
        assert (
            context_length is not None
        ), "Context length can't be None. Please specify 'model.text.context_length' argument."
        return context_length

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add model specific arguments"""
        if cls == BaseTextEncoder:
            group = parser.add_argument_group(title=cls.__name__)

            group.add_argument(
                "--model.text.name",
                type=str,
                default=None,
                help="Name of the text encoder",
            )
            group.add_argument(
                "--model.text.padding-index",
                default=None,
                type=int,
                help="Padding index. Defaults to None.",
            )

            group.add_argument(
                "--model.text.context-length",
                default=None,
                type=int,
                help="Context length. Defaults to None.",
            )
            group.add_argument(
                "--model.text.vocab-size",
                default=None,
                type=int,
                help="Vocabulary size. Defaults to None.",
            )

        return parser

    @property
    def padding_index(self) -> int:
        """Padding index."""
        pad_index = getattr(self.opts, "model.text.padding_index")
        assert (
            pad_index is None or pad_index > -1
        ), "Padding index should be None or a non-negative number."
        return pad_index

    def reset_parameters(self):
        """Initialize model weights"""
        initialize_weights(opts=self.opts, modules=self.modules())

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs
    ):

        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            *args,
            **kwargs
        )
        return param_list, [1.0] * len(param_list)

    def freeze_norm_layers(self) -> None:
        for m in self.modules():
            if isinstance(m, norm_layers_tuple):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False

    def forward(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        *args,
        **kwargs
    ) -> Any:
        raise NotImplementedError

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        """Create dummy input and labels for CI/CD purposes. Child classes must override it
        if functionality is different.
        """
        seq_length = 77
        vocab_size = 10
        text_tensor = torch.randint(
            low=0, high=vocab_size, size=(batch_size, seq_length)
        ).long()
        return {"text": text_tensor}

    def get_activation_checkpoint_submodule_class(self) -> Callable:
        """Returns model's submodule that needs to be checkpointed.

        Activations of checkpointed module are stored, and recomputed during the backward pass,
        thus providing a trade-off between memory and compute.
        """
        raise NotImplementedError("Activation checkpoint module is not implemented.")
