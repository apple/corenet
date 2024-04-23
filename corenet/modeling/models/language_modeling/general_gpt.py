#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import functools
from dataclasses import dataclass, field
from numbers import Number
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn import functional as F

from corenet.modeling.layers import (
    Embedding,
    LinearLayer,
    RotaryEmbedding,
    get_normalization_layer,
    norm_layers_tuple,
)
from corenet.modeling.layers.activation import build_activation_layer
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.language_modeling.base_lm import BaseLanguageModel
from corenet.utils import logger
from corenet.utils.math_utils import make_divisible


def compute_heads(model_dim: int, head_dim: int) -> int:
    """Compute the number of heads.

    Args:
        model_dim: Model dimension.
        head_dim: Head dimension.

    ...note:
        If model dimension is not divisible by head dimension, ValueError is raised. Otherwise, integer denoting
        number of heads in multi-head attention is returned.
    """
    if model_dim % head_dim == 0:
        return model_dim // head_dim
    else:
        raise ValueError(
            f"Model dimension should be divisible by head dimension. Got: {model_dim} and {head_dim}."
        )


@dataclass
class GPTConfig:
    vocab_size: int = 32000
    max_context_length: int = 2048

    num_transformer_layers: int = 12
    model_dim: int = 2048
    head_dim: int = 128
    qkv_multipliers: Union[Number, List[Number]] = 1.0
    num_query_heads: int = compute_heads(model_dim=model_dim, head_dim=head_dim)
    # This variable allows to switch between multi-head attention, group query attention, and multi-query attention.
    # When num_gqa_groups == 1, then it is multi-head attention.
    # When 1 < num_gqa_groups < num_heads and num_heads is divisible by num_gqa_groups, then it is group query attention
    # When num_gqa_groups == num_heads, then it is multi-query attention
    num_gqa_groups: int = 1

    # Multipliers for the feed-forward network.
    ffn_multipliers: Union[Number, List[Number]] = 4.0
    # use FFN with Gated Linear Unit (GLU)
    ffn_with_glu: bool = True
    ffn_dim_divisor: int = 256

    activation_fn_name: str = "swish"
    normalization_layer_name: str = "rms_norm"
    normalize_qk_projections: bool = False
    share_input_output_layers: bool = False

    rope_freq_constant: int = 10000
    # Note that rope_max_length is set to twice of max_context_length.
    # This allows flexibility in token lengths during training or fine-tuning.
    rope_max_length: int = 4096

    def __post_init__(self) -> None:
        if self.num_gqa_groups is not None:
            head_multiple_of = self.num_gqa_groups
        else:
            head_multiple_of = 2

        if isinstance(self.qkv_multipliers, Number):
            # All attention layers have the same latent dimensions, resulting in uniform allocation of parameters.
            qkv_dim = make_divisible(
                self.model_dim * self.qkv_multipliers,
                divisor=self.head_dim * head_multiple_of,
            )
            query_dims = [int(qkv_dim)] * self.num_transformer_layers

        elif (
            isinstance(self.qkv_multipliers, (tuple, list))
            and len(self.qkv_multipliers) == 2
        ):
            # Each attention layer have different latent dimensions assuming qkv_multipliers[0] != qkv_multipliers[1].
            # This results in variable allocation of parameters in attention layer.
            # This scaling is known as layer-wise or block-wise scaling: https://arxiv.org/abs/2008.00623
            qkv_multipliers = [
                round(v, 2)
                for v in np.linspace(
                    self.qkv_multipliers[0],
                    self.qkv_multipliers[1],
                    num=self.num_transformer_layers,
                    dtype=float,
                )
            ]
            # Make sure that scaled model dimension is divisible by scaled head dimension.
            query_dims = [
                int(
                    make_divisible(
                        self.model_dim * m, divisor=self.head_dim * head_multiple_of
                    )
                )
                for m in qkv_multipliers
            ]
        else:
            raise NotImplementedError(
                f"QKV multipliers should be a single number or a list containing exactly two numbers. Got: {qkv_multipliers}."
            )

        # compute the number of query, key, and value heads
        # For multi-head and multi-query attention, the number of heads for query, key, and value are the same.
        # For group query attention, the number of key and value heads are the same.
        self.num_query_heads = [
            int(compute_heads(q_dim, self.head_dim)) for q_dim in query_dims
        ]
        self.num_kv_heads = [
            q_heads // self.num_gqa_groups for q_heads in self.num_query_heads
        ]

        # Feed-forward network (FFN) multipliers
        if isinstance(self.ffn_multipliers, Number):
            # All FFN layers have the same latent dimensions, resulting in uniform allocation of parameters.
            self.ffn_multipliers = [self.ffn_multipliers] * self.num_transformer_layers
        elif (
            isinstance(self.ffn_multipliers, (tuple, list))
            and len(self.ffn_multipliers) == 2
        ):
            # Each FFN layer have different latent dimensions assuming ffn_multipliers[0] != ffn_multipliers[1].
            # This results in variable allocation of parameters in FFN layer.
            # This scaling is known as layer-wise or block-wise scaling: https://arxiv.org/abs/2008.00623
            self.ffn_multipliers = [
                round(v, 2)
                for v in np.linspace(
                    self.ffn_multipliers[0],
                    self.ffn_multipliers[1],
                    num=self.num_transformer_layers,
                    dtype=float,
                )
            ]
        else:
            raise NotImplementedError(
                f"FFN multipliers should be a single number or a list containing exactly two numbers. Got: {qkv_multipliers}."
            )

    @classmethod
    def from_name(
        cls, model_name: str, vocab_size: int, max_context_length: int
    ) -> "GPTConfig":
        if model_name in gpt_configs:
            config = gpt_configs[model_name]
        else:
            raise NotImplementedError(f"{model_name} is not yet implemented")

        config["vocab_size"] = vocab_size
        config["max_context_length"] = max_context_length
        return cls(**config)


gpt_configs = {
    "gpt-test": dict(
        num_transformer_layers=1,
        model_dim=128,
        head_dim=64,
        num_gqa_groups=1,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multiplier to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.25, 0.75),
        qkv_multipliers=(0.25, 0.5),
    ),
    # A sample GPT configuration.
    "gpt-1_3B": dict(
        num_transformer_layers=24,
        model_dim=2048,
        head_dim=64,
        max_context_length=2048,
        # For gated FFN, the value is around 3. while for standard FFN, the value is 4.0.
        ffn_multipliers=3.0,
        # Number of GQA groups.
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
    ),
    "OpenELM-270M": dict(
        num_transformer_layers=16,
        model_dim=1280,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multiplier to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
    "OpenELM-450M": dict(
        num_transformer_layers=20,
        model_dim=1536,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multiplier to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
    "OpenELM-1_1B": dict(
        num_transformer_layers=28,
        model_dim=2048,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multiplier to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
    "OpenELM-3B": dict(
        num_transformer_layers=36,
        model_dim=3072,
        head_dim=128,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multiplier to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
    ),
}


class MultiHeadCausalAttention(nn.Module):
    """Multi-head causal attention.

    Args:
        opts: Command-line arguments.
        model_config: Model configuration.
        layer_idx: Layer index.
    """

    def __init__(
        self, opts: argparse.Namespace, model_config: GPTConfig, layer_idx: int
    ) -> None:
        super().__init__()
        assert (
            model_config.num_query_heads[layer_idx]
            % model_config.num_kv_heads[layer_idx]
            == 0
        ), f"Number of query heads are not divisible by number of key/value heads. Got: {model_config.num_query_heads[layer_idx]} and {model_config.num_kv_heads[layer_idx]}."

        head_dim = model_config.head_dim
        q_heads = model_config.num_query_heads[layer_idx]
        k_heads = model_config.num_kv_heads[layer_idx]
        v_heads = model_config.num_kv_heads[layer_idx]

        self.qkv_proj = LinearLayer(
            in_features=model_config.model_dim,
            out_features=(q_heads + k_heads + v_heads) * head_dim,
            bias=False,
        )

        self.pos_embedding = RotaryEmbedding(
            model_dim=model_config.head_dim,
            max_seq_length=model_config.rope_max_length,
            freq_constant=model_config.rope_freq_constant,
        )

        if model_config.normalize_qk_projections:
            self.q_norm = get_normalization_layer(
                opts,
                num_features=model_config.head_dim,
                norm_type=model_config.normalization_layer_name,
            )
            self.k_norm = get_normalization_layer(
                opts,
                num_features=model_config.head_dim,
                norm_type=model_config.normalization_layer_name,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.out_proj = LinearLayer(
            in_features=q_heads * head_dim,
            out_features=model_config.model_dim,
            bias=False,
        )

        self.head_dim = model_config.head_dim
        self.num_q_heads = q_heads
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        self.model_dim = model_config.model_dim
        self.num_groups = self.num_q_heads // self.num_k_heads

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f"model_dim={self.model_dim}, num_query_heads={self.num_q_heads}, num_key_heads={self.num_k_heads}, num_value_heads={self.num_v_heads}"
        )

    def forward(
        self,
        x: Tensor,
        past_keys: Optional[Tensor] = None,
        past_values: Optional[Tensor] = None,
        use_kv_cache: bool = False,
        is_causal: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Forward pass of multi-head self-attention.

        Args:
            x: Input tensor of the shape [batch size, sequence length, model dimension].
            past_keys: Tensor storing the cached keys.
                The shape of tensor is [batch size, number of key heads, sequence length, head dimension].
            past_values: Tensor storing the cached values. The shape of the tensor is the same as 'past_keys'.
            use_kv_cache: Cache the output of key and value projection layers for faster inference.
            is_causal: Specifies whether to apply causal masking in scaled dot-product attention.

        Returns:
            The output of the same shape as the input, optionally with a tensor containing cached keys and values.
        """
        batch_size, seq_length, d_model = x.shape

        # [batch_size, seq_length, d_model] --> [batch_size, seq_length, (num_q_heads + num_k_heads + num_v_heads) * head_dim]
        qkv = self.qkv_proj(x)
        # [batch_size, seq_length, (num_q_heads + num_k_heads + num_v_heads) * head_dim] --> [batch_size, seq_length, (num_q_heads + num_k_heads + num_v_heads), head_dim]
        qkv = qkv.reshape(
            batch_size,
            seq_length,
            self.num_q_heads + self.num_k_heads + self.num_v_heads,
            self.head_dim,
        )
        # [batch_size, seq_length, (num_q_heads + num_k_heads + num_v_heads), head_dim] --> [batch_size, (num_q_heads + num_k_heads + num_v_heads), seq_length, head_dim]
        qkv = qkv.transpose(1, 2)
        # [batch_size, (num_q_heads + num_k_heads + num_v_heads), seq_length, head_dim] --> [batch_size, num_q_heads, seq_length, head_dim], [batch_size, num_k_heads, seq_length, head_dim], [batch_size, num_v_heads, seq_length, head_dim]
        queries, keys, values = qkv.split(
            [self.num_q_heads, self.num_k_heads, self.num_v_heads], dim=1
        )

        if self.q_norm is not None:
            queries = self.q_norm(queries)

        if self.k_norm is not None:
            keys = self.k_norm(keys)

        if use_kv_cache:
            if past_keys is not None:
                assert past_values is not None
                # concatenate past and current keys along the sequence dimension.
                keys = torch.cat([past_keys, keys], dim=-2)
                values = torch.cat([past_values, values], dim=-2)

            past_keys = keys
            past_values = values

        # Add positional embedding
        queries, keys = self.pos_embedding(queries, keys)

        if self.num_groups != 1:
            # Group-query attention.
            # [batch_size, num_k_heads, seq_length, head_dim] --> [batch_size, num_q_heads, seq_length, head_dim]
            keys = keys.repeat_interleave(self.num_groups, dim=1)
            # [batch_size, num_v_heads, seq_length, head_dim] --> [batch_size, num_q_heads, seq_length, head_dim]
            values = values.repeat_interleave(self.num_groups, dim=1)

        # scaled dot-product attention.
        # The output of this operation has size of [batch_size, num_q_heads, seq_length, head_dim]
        attn_output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            attn_mask=None,
            dropout_p=0,
            is_causal=is_causal,
        )
        # [batch_size, num_q_heads, seq_length, head_dim] --> [batch_size, seq_length, num_q_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [batch_size, seq_length, num_q_heads, head_dim] --> [batch_size, seq_length, num_q_heads * head_dim]
        attn_output = attn_output.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )
        # [batch_size, seq_length, num_q_heads * head_dim] --> [batch_size, seq_length, d_model]
        out = self.out_proj(attn_output)
        return out, past_keys, past_values


class FeedForwardNetwork(nn.Module):
    """Feed-forward network.

    Args:
        opts: Command-line arguments.
        model_config: Model configuration.
        layer_idx: Layer index.
    """

    def __init__(
        self, opts: argparse.Namespace, model_config: GPTConfig, layer_idx: int
    ) -> None:
        super().__init__()
        ffn_multiplier = model_config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * model_config.model_dim,
                divisor=model_config.ffn_dim_divisor,
            )
        )
        if model_config.ffn_with_glu:
            # FFN with Gated linear unit, as described in https://arxiv.org/abs/2002.05202v1.
            self.proj_1 = LinearLayer(
                in_features=model_config.model_dim,
                out_features=2 * intermediate_dim,
                bias=False,
            )
            self.proj_2 = LinearLayer(
                in_features=intermediate_dim,
                out_features=model_config.model_dim,
                bias=False,
            )
            self.ffn_with_glu = True
        else:
            # Standard FFN, as described in https://arxiv.org/abs/1706.03762
            self.proj_1 = LinearLayer(
                in_features=model_config.model_dim,
                out_features=intermediate_dim,
                bias=False,
            )
            self.proj_2 = LinearLayer(
                in_features=intermediate_dim,
                out_features=model_config.model_dim,
                bias=False,
            )
            self.ffn_with_glu = False

        self.act = build_activation_layer(
            opts=opts, act_type=model_config.activation_fn_name
        )

    def extra_repr(self) -> str:
        return super().extra_repr() + f"(ffn_with_glu) : {self.ffn_with_glu}"

    def forward(self, x: Tensor) -> Tensor:
        """Forward function of FFN layer.

        Args:
            x: Input tensor of the shape [batch size, sequence length, model dimension].

        Returns:
            A tensor of the same shape as the input.
        """
        if self.ffn_with_glu:
            y_12 = self.proj_1(x)
            y_1, y_2 = y_12.chunk(2, dim=-1)
            y = self.act(y_1) * y_2
            return self.proj_2(y)
        else:
            return self.proj_2(self.act(self.proj_1(x)))


class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer.

    Args:
        opts: Command-line arguments.
        model_config: Model configuration.
        layer_idx: Layer index.
    """

    def __init__(
        self, opts: argparse.Namespace, model_config: GPTConfig, layer_idx: int
    ) -> None:
        super().__init__()
        self.attn = MultiHeadCausalAttention(
            opts, model_config=model_config, layer_idx=layer_idx
        )
        self.ffn = FeedForwardNetwork(
            opts, model_config=model_config, layer_idx=layer_idx
        )
        self.ffn_norm = get_normalization_layer(
            opts,
            num_features=model_config.model_dim,
            norm_type=model_config.normalization_layer_name,
        )
        self.attn_norm = get_normalization_layer(
            opts,
            num_features=model_config.model_dim,
            norm_type=model_config.normalization_layer_name,
        )

    def forward(
        self,
        x: Tensor,
        past_keys: Optional[Tensor] = None,
        past_values: Optional[Tensor] = None,
        use_kv_cache: bool = False,
        is_causal: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Forward pass of decoder layer.

        Args:
            x: Input tensor of the shape [batch size, sequence length, model dimension].
            past_keys: Tensor storing the cached keys.
                The shape of tensor is [batch size, number of key heads, sequence length, head dimension].
            past_values: Tensor storing the cached values. The shape of the tensor is the same as 'past_keys'.
            use_kv_cache: Cache the output of key and value projection layers for faster inference.
            is_causal: Specifies whether to apply causal masking in scaled dot-product attention.

        Returns:
            The output of the same shape as the input, optionally with a tensor containing cached keys and values.
        """
        # Pre-norm attention.
        y_attn = self.attn_norm(x)
        y_attn, past_keys, past_values = self.attn(
            y_attn, past_keys, past_values, use_kv_cache, is_causal
        )
        y_attn = x + y_attn

        # Pre-norm FFN.
        y_ffn = y_attn + self.ffn(self.ffn_norm(y_attn))
        return y_ffn, past_keys, past_values


@MODEL_REGISTRY.register(name="general_gpt", type="language_modeling")
class GeneralGPTModel(BaseLanguageModel):
    """General GPT model.

    Args:
        opts: Command-line arguments.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)
        model_name = getattr(opts, "model.language_modeling.general_gpt.model_name")
        if model_name is None:
            logger.error(
                "Please specify model name using 'model.language_modeling.general_gpt.model_name' parameter in your configuration file."
            )

        vocab_size = getattr(opts, "model.language_modeling.general_gpt.vocab_size")
        if vocab_size is None:
            logger.error(
                "Please specify vocabulary size using 'model.language_modeling.general_gpt.vocab_size' parameter in your configuration file."
            )

        max_context_length = getattr(
            opts, "model.language_modeling.general_gpt.max_context_length"
        )
        if max_context_length is None:
            logger.error(
                "Please specify maximum context length using 'model.language_modeling.general_gpt.max_context_length' parameter in your configuration file."
            )

        padding_index = getattr(
            opts, "model.language_modeling.general_gpt.padding_index"
        )

        model_config = GPTConfig.from_name(
            model_name=model_name,
            vocab_size=vocab_size,
            max_context_length=max_context_length,
        )
        self.token_embeddings = Embedding(
            opts,
            embedding_dim=model_config.model_dim,
            num_embeddings=model_config.vocab_size,
            padding_idx=padding_index,
        )

        self.layers = nn.ModuleList(
            TransformerDecoderLayer(
                opts, model_config=model_config, layer_idx=layer_idx
            )
            for layer_idx in range(model_config.num_transformer_layers)
        )
        self.norm = get_normalization_layer(
            opts,
            num_features=model_config.model_dim,
            norm_type=model_config.normalization_layer_name,
        )

        if model_config.share_input_output_layers:
            self.classifier = None
        else:
            self.classifier = LinearLayer(
                in_features=model_config.model_dim,
                out_features=model_config.vocab_size,
                bias=False,
            )
        self.reset_parameters(model_config=model_config)
        self.num_transformer_layers = model_config.num_transformer_layers

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add General GPT model arguments."""
        if cls == GeneralGPTModel:
            group = parser.add_argument_group(cls.__name__)
            group.add_argument(
                "--model.language-modeling.general-gpt.model-name",
                type=str,
                default=None,
                choices=list(gpt_configs.keys()),
                help="Name of the generative transformer-based LM model. Defaults to None (i.e., user need to specify the model name.).",
            )
            group.add_argument(
                "--model.language-modeling.general-gpt.max-context-length",
                type=int,
                default=None,
                help="Maximum context length. Defaults to None (i.e., user needs to specify the maximum contenxt length value.).",
            )
            group.add_argument(
                "--model.language-modeling.general-gpt.vocab-size",
                type=int,
                default=None,
                help="Vocabulary size. Defaults to None (i.e., user needs to specify the vocabulary size.).",
            )
            group.add_argument(
                "--model.language-modeling.general-gpt.padding-index",
                type=int,
                default=None,
                help="Padding index. Defaults to None (i.e., no padding).",
            )
        return parser

    def forward(
        self, model_input: Union[Tensor, Dict[str, Tensor]]
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Forward function of GPT model.

        Args:
            model_input: Input to the model. It can be a tensor or a dictionary.
                In case of a tensor, the expected shape is [batch size, sequence length].
                In case of a dictionary, the expected keys are 'input_ids', 'past_keys', 'past_values',
                'use_kv_cache', and 'is_causal'. The shape of the values for each key is:
                    {
                        "input_ids": [batch size, sequence length],
                        "past_keys": [ [batch size, number of key heads, sequence length, head dimension] ]* number of transformer layers,
                        "past_values": [ [batch size, number of value heads, sequence length, head dimension] ] * number of transformer layers,
                        "use_kv_cache": boolean,
                        "is_causal": boolean,
                    }
                where
                    'input_ids' represents input token indices.
                    'past_keys' and 'past_values' represents the cached tensor outputs of key and value branch in multi-head attention respectively.
                        These values can be None.
                    'use_kv_cache' indicates to use KV caching or not.
                    'is_causal' indicates to use causal masking in scaled dot-product attention or not.

        Returns:
            Output of the model.
            1. When 'use_kv_cache' is enabled, a dictionary with 'logits', 'past_keys', and 'past_values' is returned.
            The expected shape of the values is
                {
                    "logits": [batch size, sequence length, vocabular size],
                    "past_keys": [ [batch size, number of key heads, sequence length, head dimension] ] * number of transformer layers,
                    "past_values": [ [batch size, number of value heads, sequence length, head dimension] ] * number of transformer layers,
                }
            2. Logits tensor is returned. The shape of logits tensor is [batch size, sequence length, vocabulary size].

        ...note:
            1. For pre-training, 'model_input' is typically a tensor.
            2. For inference, we have two scenarios.
                2.a. Processing prefix or prompt: When dealing with a prefix or prompt, it is expected that the 'sequence length' is more than one and past keys
                    or values are None. If the intention of the user is to perform generation following a prefix, it's recommended to provide the prefix inputs
                    as a dictionary, specifying 'use_kv_cache=True', 'is_causal=True', 'past_keys=None', and 'past_values=None'. Otherwise, users should pass token
                    indices as a tensor.

                2.b. Generation: In this case, 'sequence length' should be one. In other words, one token is generated at a time with KV caching.
                    Ideally, when using KV caching, 'is_causal' should be set to False.

            The generation logic may vary from task to task and we rely on user for correctly passing the inputs.
        """
        if isinstance(model_input, dict):
            expected_input_keys = {
                "input_ids",
                "past_keys",
                "past_values",
                "use_kv_cache",
                "is_causal",
            }

            assert expected_input_keys == set(
                model_input.keys()
            ), f"Model input does not contain all keys. Expected keys are {expected_input_keys}, but got {set(model_input.keys())}."

            input_ids = model_input["input_ids"]
            past_keys = model_input["past_keys"]
            past_values = model_input["past_values"]
            use_kv_cache = model_input["use_kv_cache"]
            is_causal = model_input["is_causal"]
            if past_keys is None:
                assert past_values is None
                past_keys = [None] * self.num_transformer_layers
                past_values = [None] * self.num_transformer_layers
        elif isinstance(model_input, Tensor):
            input_ids = model_input
            past_keys = [None] * self.num_transformer_layers
            past_values = [None] * self.num_transformer_layers
            use_kv_cache = False
            is_causal = True
        else:
            raise NotImplementedError(
                f"Supported input types are either Tensor or Dictionary. Got: {type(model_input)}."
            )

        x = self.token_embeddings(input_ids)

        for layer_idx in range(self.num_transformer_layers):
            past_keys_layer_i = past_keys[layer_idx]
            past_values_layer_i = past_values[layer_idx]

            x, past_keys_layer_i, past_values_layer_i = self.layers[layer_idx](
                x, past_keys_layer_i, past_values_layer_i, use_kv_cache, is_causal
            )
            # update the kv cache
            past_keys[layer_idx] = past_keys_layer_i
            past_values[layer_idx] = past_values_layer_i

        x = self.norm(x)
        if self.classifier is None:
            logits = F.linear(x, weight=self.token_embeddings.weight)
        else:
            logits = self.classifier(x)

        if use_kv_cache:
            return {
                "logits": logits,
                "past_keys": past_keys,
                "past_values": past_values,
            }
        else:
            return logits

    def get_fsdp_wrap_policy(
        self,
    ) -> Callable[[torch.nn.Module, bool, int], bool]:
        """Returns the FSDP policy."""
        general_gpt_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerDecoderLayer},
        )
        return general_gpt_auto_wrap_policy

    def get_activation_checkpoint_submodule_class(self) -> Callable:
        """Returns the layer that should be used for activation checkpointing."""
        return TransformerDecoderLayer

    def reset_parameters(self, model_config: GPTConfig) -> None:
        """Initialize the parameters of language model.

        Args:
            model_config: Model configuration.
        """
        for module in self.modules():
            if isinstance(module, (LinearLayer, nn.Linear)):
                std = module.in_features**-0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Embedding, Embedding)):
                std = module.embedding_dim**-0.5
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, norm_layers_tuple):
                if module.weight is not None:
                    torch.nn.init.ones_(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        model_dim = model_config.model_dim
        n_layers = model_config.num_transformer_layers
        # standard deviation of output layers in transformer block is scaled,
        # following https://arxiv.org/pdf/2205.01068.pdf
        std = (model_dim**-0.5) * ((2 * n_layers) ** -0.5)
        for param_name, param in self.named_parameters():
            if param_name.endswith("out_proj.weight") or param_name.endswith(
                "ffn.proj_2.weight"
            ):
                torch.nn.init.normal_(param, mean=0.0, std=std)
