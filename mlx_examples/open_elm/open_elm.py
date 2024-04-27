#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import dataclasses
import json
import pathlib
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mlx
import sentencepiece
from mlx import core as mx
from mlx import nn


def get_normalization_layer(num_features: int, norm_type: str) -> nn.Module:
    if norm_type == "rms_norm":
        return nn.RMSNorm(dims=num_features, eps=1e-6)
    raise ValueError(f"Unsupported normalization layer: {norm_type}.")


def compute_heads(model_dim: int, head_dim: int) -> int:
    if model_dim % head_dim == 0:
        return model_dim // head_dim
    raise ValueError(
        "Model dimension should be divisible by head dimension. Got: "
        f"{model_dim} and {head_dim}."
    )


def make_divisible(val: Union[int, float], divisor: int) -> int:
    """Make val divisible by divisor, rounding down no more than 10%, otherwise
    rounding up. If val is less than divisor, returns divisor."""
    assert val >= 0.0, val
    assert divisor > 0, divisor
    assert isinstance(divisor, int)

    if val < divisor:
        return divisor

    # First round down to a whole multiple of divisor and see if it's within 10%
    # of val. If it is, return it, if not, add another `divisor` to get the
    # upper rounding.
    round_down = int(val + divisor / 2) // divisor * divisor

    if round_down <= 0.9 * val:
        return round_down + divisor
    else:
        return round_down


@dataclasses.dataclass
class GPTConfig:
    vocab_size: int = 32000
    max_context_length: int = 2048

    num_transformer_layers: int = 12
    model_dim: int = 2048

    head_dim: int = 128
    qkv_multipliers: Union[Number, List[Number]] = 1.0
    num_query_heads: int = compute_heads(model_dim=model_dim, head_dim=head_dim)
    # This variable allows to switch between multi-head attention, group query
    # attention, and multi-query attention.
    # When num_gqa_groups == 1, then it is multi-head attention.
    # When 1 < num_gqa_groups < num_heads and num_heads is divisible by
    # num_gqa_groups, then it is group query attention
    # When num_gqa_groups == num_heads, then it is multi-query attention
    num_gqa_groups: int = 1

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
            qkv_dim = make_divisible(
                int(self.model_dim * self.qkv_multipliers),
                divisor=self.head_dim * head_multiple_of,
            )
            query_dims = [int(qkv_dim)] * self.num_transformer_layers

        elif (
            isinstance(self.qkv_multipliers, (tuple, list))
            and len(self.qkv_multipliers) == 2
        ):
            # Each attention layer have different latent dimensions assuming
            # qkv_multipliers[0] != qkv_multipliers[1].
            # This results in variable allocation of parameters in attention layer.
            # This scaling is known as layer-wise or block-wise scaling:
            # https://arxiv.org/abs/2008.00623
            qkv_multipliers = [
                round(v.item(), 2)
                for v in mx.linspace(
                    self.qkv_multipliers[0],
                    self.qkv_multipliers[1],
                    num=self.num_transformer_layers,
                    dtype=mx.float32,
                )
            ]
            query_dims = [
                make_divisible(
                    self.model_dim * mult, divisor=self.head_dim * head_multiple_of
                )
                for mult in qkv_multipliers
            ]
        else:
            raise NotImplementedError(
                "QKV multipliers should be a single number or a list containing exactly "
                f"two numbers. Got: {qkv_multipliers}."
            )

        # compute the number of query, key, and value heads
        # For multi-head and multi-query attention, the number of heads for
        # query, key, and value are the same.
        # For group query attention, the number of key and value heads are the same.
        self.num_query_heads = [
            compute_heads(q_dim, self.head_dim) for q_dim in query_dims
        ]
        self.num_kv_heads = [
            q_heads // self.num_gqa_groups for q_heads in self.num_query_heads
        ]

        # Feed-forward network (FFN) multipliers
        if isinstance(self.ffn_multipliers, Number):
            # All FFN layers have the same latent dimensions, resulting in
            # uniform allocation of parameters.
            self.ffn_multipliers = [self.ffn_multipliers] * self.num_transformer_layers
        elif (
            isinstance(self.ffn_multipliers, (tuple, list))
            and len(self.ffn_multipliers) == 2
        ):
            # Each FFN layer have different latent dimensions assuming
            # ffn_multipliers[0] != ffn_multipliers[1].
            # This results in variable allocation of parameters in FFN layer.
            # This scaling is known as layer-wise or block-wise scaling:
            # https://arxiv.org/abs/2008.00623
            self.ffn_multipliers = [
                round(v.item(), 2)
                for v in mx.linspace(
                    self.ffn_multipliers[0],
                    self.ffn_multipliers[1],
                    num=self.num_transformer_layers,
                    dtype=mx.float32,
                )
            ]
        else:
            raise NotImplementedError(
                "FFN multipliers should be a single number or a list containing exactly "
                f"two numbers. Got: {qkv_multipliers}."
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
    "OpenELM-270M": {
        "num_transformer_layers": 16,
        "model_dim": 1280,
        "head_dim": 64,
        "num_gqa_groups": 4,
        "normalize_qk_projections": True,
        "share_input_output_layers": True,
        # Vary the FFN and QKV multiplier to create variable FFN and attention
        # layers respectively.
        "ffn_multipliers": (0.5, 4.0),
        "qkv_multipliers": (0.5, 1.0),
    },
    "OpenELM-450M": {
        "num_transformer_layers": 20,
        "model_dim": 1536,
        "head_dim": 64,
        "num_gqa_groups": 4,
        "normalize_qk_projections": True,
        "share_input_output_layers": True,
        "ffn_multipliers": (0.5, 4.0),
        "qkv_multipliers": (0.5, 1.0),
    },
    "OpenELM-1_1B": {
        "num_transformer_layers": 28,
        "model_dim": 2048,
        "head_dim": 64,
        "num_gqa_groups": 4,
        "normalize_qk_projections": True,
        "share_input_output_layers": True,
        "ffn_multipliers": (0.5, 4.0),
        "qkv_multipliers": (0.5, 1.0),
    },
    "OpenELM-3B": {
        "num_transformer_layers": 36,
        "model_dim": 3072,
        "head_dim": 128,
        "num_gqa_groups": 4,
        "normalize_qk_projections": True,
        "share_input_output_layers": True,
        "ffn_multipliers": (0.5, 4.0),
        "qkv_multipliers": (0.5, 1.0),
    },
}


class MultiHeadCausalAttention(nn.Module):
    """Multi-head causal attention.

    Args:
        model_config: Model configuration.
        layer_idx: Layer index.
    """

    def __init__(self, model_config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        assert (
            model_config.num_query_heads[layer_idx]
            % model_config.num_kv_heads[layer_idx]
            == 0
        )

        head_dim = model_config.head_dim
        q_heads = model_config.num_query_heads[layer_idx]
        k_heads = model_config.num_kv_heads[layer_idx]
        v_heads = model_config.num_kv_heads[layer_idx]

        self.qkv_proj = nn.Linear(
            input_dims=model_config.model_dim,
            output_dims=(q_heads + k_heads + v_heads) * head_dim,
            bias=False,
        )

        self.pos_embedding = nn.RoPE(
            model_config.head_dim, base=model_config.rope_freq_constant
        )

        if model_config.normalize_qk_projections:
            self.q_norm = get_normalization_layer(
                num_features=model_config.head_dim,
                norm_type=model_config.normalization_layer_name,
            )
            self.k_norm = get_normalization_layer(
                num_features=model_config.head_dim,
                norm_type=model_config.normalization_layer_name,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.out_proj = nn.Linear(
            input_dims=q_heads * head_dim,
            output_dims=model_config.model_dim,
            bias=False,
        )

        self.head_dim = model_config.head_dim
        self.num_q_heads = q_heads
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        self.model_dim = model_config.model_dim
        self.num_groups = self.num_q_heads // self.num_k_heads
        self.scale = head_dim**-0.5

    def extra_repr(self) -> str:
        return (
            super().__repr__()
            + f", query_heads={self.num_q_heads}, "
            + f"key_heads={self.num_k_heads}, "
            + f"value_heads={self.num_v_heads}"
        )

    def __call__(
        self,
        x: mx.array,
        past_key_value: Optional[Tuple[mx.array, mx.array]] = None,
        use_kv_cache: bool = False,
        causal_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
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

        # [batch_size, seq_length, d_model] --> [batch_size, seq_length,
        # (num_q_heads + num_k_heads + num_v_heads) * head_dim]
        qkv = self.qkv_proj(x)
        # [batch_size, seq_length, (num_q_heads + num_k_heads + num_v_heads) * head_dim] --> [batch_size, seq_length, (num_q_heads + num_k_heads + num_v_heads), head_dim]
        qkv = qkv.reshape(
            batch_size,
            seq_length,
            self.num_q_heads + self.num_k_heads + self.num_v_heads,
            self.head_dim,
        )
        # [batch_size, seq_length, (num_q_heads + num_k_heads + num_v_heads), head_dim] --> [batch_size, (num_q_heads + num_k_heads + num_v_heads), seq_length, head_dim]
        qkv = qkv.transpose(0, 2, 1, 3)
        # [batch_size, (num_q_heads + num_k_heads + num_v_heads), seq_length, head_dim] --> [batch_size, num_q_heads, seq_length, head_dim], [batch_size, num_k_heads, seq_length, head_dim], [batch_size, num_v_heads, seq_length, head_dim]
        queries, keys, values = qkv.split(
            [self.num_q_heads, self.num_q_heads + self.num_k_heads], axis=1
        )

        if self.q_norm is not None:
            queries = self.q_norm(queries)

        if self.k_norm is not None:
            keys = self.k_norm(keys)

        if use_kv_cache:
            if past_key_value is not None:
                past_keys = past_key_value[0]
                past_values = past_key_value[1]
                queries = self.pos_embedding(queries, offset=past_keys.shape[2])
                keys = self.pos_embedding(keys, offset=past_keys.shape[2])
                # concatenate past and current keys along the sequence dimension.
                keys = mx.concatenate([past_keys, keys], axis=2)
                values = mx.concatenate([past_values, values], axis=2)
            else:
                queries = self.pos_embedding(queries)
                keys = self.pos_embedding(keys)

            past_key_value = (keys, values)

        if self.num_groups != 1:
            # Group-query attention.
            # [batch_size, num_k_heads, seq_length, head_dim] --> [batch_size, num_q_heads, seq_length, head_dim]
            keys = mx.repeat(keys, self.num_groups, axis=1)
            # [batch_size, num_v_heads, seq_length, head_dim] --> [batch_size, num_q_heads, seq_length, head_dim]
            values = mx.repeat(values, self.num_groups, axis=1)

        # scaled dot-product attention.
        # The output of this operation has size of [batch_size, num_q_heads, seq_length, head_dim]
        attn_output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=self.scale,
            mask=causal_mask,
        )
        # [batch_size, num_q_heads, seq_length, head_dim] --> [batch_size, seq_length, num_q_heads, head_dim]
        attn_output = attn_output.transpose(0, 2, 1, 3)
        # [batch_size, seq_length, num_q_heads, head_dim] --> [batch_size, seq_length, num_q_heads * head_dim]
        attn_output = attn_output.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )
        # [batch_size, seq_length, num_q_heads * head_dim] --> [batch_size, seq_length, d_model]
        out = self.out_proj(attn_output)

        return out, past_key_value


class FeedForwardNetwork(nn.Module):
    """Feed-forward network.

    Args:
        opts: Command-line arguments.
        model_config: Model configuration.
        layer_idx: Layer index.
    """

    def __init__(self, model_config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        ffn_multiplier = model_config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * model_config.model_dim,
                divisor=model_config.ffn_dim_divisor,
            )
        )
        if model_config.ffn_with_glu:
            # FFN with Gated linear unit
            self.proj_1 = nn.Linear(
                input_dims=model_config.model_dim,
                output_dims=2 * intermediate_dim,
                bias=False,
            )
            self.proj_2 = nn.Linear(
                input_dims=intermediate_dim,
                output_dims=model_config.model_dim,
                bias=False,
            )
            self.ffn_with_glu = True
        else:
            # Standard FFN, as described in https://arxiv.org/abs/1706.03762
            self.proj_1 = nn.Linear(
                input_dims=model_config.model_dim,
                output_dims=intermediate_dim,
                bias=False,
            )
            self.proj_2 = nn.Linear(
                input_dims=intermediate_dim,
                output_dims=model_config.model_dim,
                bias=False,
            )
            self.ffn_with_glu = False

        assert (
            model_config.activation_fn_name == "swish"
        ), model_config.activation_fn_name
        self.act = nn.SiLU()  # AKA Swish.

    def extra_repr(self) -> str:
        return super().__repr__() + f"(ffn_with_glu) : {self.ffn_with_glu}"

    def __call__(self, x):
        """Forward function of FFN layer.

        Args:
            x: Input tensor of the shape [batch size, sequence length, model dimension].

        Returns:
            A tensor of the same shape as the input.
        """
        if self.ffn_with_glu:
            y_12 = self.proj_1(x)
            y_1, y_2 = y_12.split(2, axis=-1)
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

    def __init__(self, model_config: GPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = get_normalization_layer(
            num_features=model_config.model_dim,
            norm_type=model_config.normalization_layer_name,
        )
        self.attn = MultiHeadCausalAttention(
            model_config=model_config, layer_idx=layer_idx
        )
        self.ffn_norm = get_normalization_layer(
            num_features=model_config.model_dim,
            norm_type=model_config.normalization_layer_name,
        )
        self.ffn = FeedForwardNetwork(model_config=model_config, layer_idx=layer_idx)

    def __call__(
        self,
        x: mx.array,
        past_key_value: Optional[mx.array] = None,
        use_kv_cache: bool = False,
        causal_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
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
        y_attn, past_key_value = self.attn(
            y_attn, past_key_value, use_kv_cache, causal_mask
        )
        y_attn = x + y_attn

        # Pre-norm FFN.
        y_ffn = y_attn + self.ffn(self.ffn_norm(y_attn))
        return y_ffn, past_key_value


class OpenELM(nn.Module):
    """General GPT model.

    Args:
        model_name: name key for the predefined configuration
        vocab_size: the size of the input token embedding table
        max_context_length: max context length this model will be able to process

    Returns:
        None
    """

    def __init__(
        self,
        model_name: str,
        vocab_size: int,
        max_context_length: int,
    ) -> None:
        super().__init__()
        model_config = GPTConfig.from_name(
            model_name=model_name,
            vocab_size=vocab_size,
            max_context_length=max_context_length,
        )

        self.token_embeddings = nn.Embedding(
            num_embeddings=model_config.vocab_size,
            dims=model_config.model_dim,
        )

        self.layers = [
            TransformerDecoderLayer(model_config=model_config, layer_idx=layer_idx)
            for layer_idx in range(model_config.num_transformer_layers)
        ]
        self.norm = get_normalization_layer(
            num_features=model_config.model_dim,
            norm_type=model_config.normalization_layer_name,
        )
        if model_config.share_input_output_layers:
            self.classifier = None
        else:
            self.classifier = nn.Linear(
                input_dims=model_config.model_dim,
                output_dims=model_config.vocab_size,
                bias=False,
            )
        self.reset_parameters(model_config=model_config)
        self.num_transformer_layers = model_config.num_transformer_layers

    def __call__(
        self, model_input: Union[mx.array, Dict[str, Any]]
    ) -> Union[mx.array, Dict[str, mx.array]]:
        if isinstance(model_input, dict):
            assert {
                "input_ids",
                "past_key_values",
                "use_kv_cache",
                "is_causal",
            }.issubset(model_input.keys())
            input_ids = model_input["input_ids"]
            past_key_values = model_input["past_key_values"]
            use_kv_cache = model_input["use_kv_cache"]
            is_causal = model_input["is_causal"]
            if past_key_values is None:
                past_key_values = [None] * self.num_transformer_layers
        elif isinstance(model_input, mx.array):
            input_ids = model_input
            past_key_values = [None] * self.num_transformer_layers
            use_kv_cache = False
            is_causal = True
        else:
            raise NotImplementedError(
                "Supported input types are either mx.array or Dictionary. "
                f"Got: {type(model_input)}."
            )

        x = self.token_embeddings(input_ids)

        causal_mask = None
        seq_length = x.shape[1]
        if is_causal and seq_length > 1:
            causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_length)
            causal_mask = causal_mask.astype(x.dtype)

        for layer_idx in range(self.num_transformer_layers):
            x, past_key_values[layer_idx] = self.layers[layer_idx](
                x, past_key_values[layer_idx], use_kv_cache, causal_mask
            )

        x = self.norm(x)
        if self.classifier is None:
            logits = mx.matmul(x, self.token_embeddings.weight.transpose())
        else:
            logits = self.classifier(x)

        if use_kv_cache:
            return {"logits": logits, "past_key_values": past_key_values}
        else:
            return logits

    def reset_parameters(self, model_config: GPTConfig) -> None:
        """Initialize the layers in Language Model

        The initialization scheme is followed, following `OPT
        <https://arxiv.org/pdf/2205.01068.pdf>`_.

        Returns:
            None
        """

        def name_filter(value_name: str) -> Callable[[Any, str, Any], bool]:
            def filter(layer: nn.Module, name: str, val: mx.array) -> bool:
                return name == value_name

            return filter

        model_dim = model_config.model_dim
        for module in self.modules():
            if isinstance(module, (nn.Linear)):
                std = module.weight.shape[0] ** -0.5
                module.apply(nn.init.normal(std=std), filter_fn=name_filter("weight"))
                module.apply(nn.init.constant(0.0), filter_fn=name_filter("bias"))
            elif isinstance(module, (nn.Embedding)):
                std = model_dim**-0.5
                module.apply(nn.init.normal(std), filter_fn=name_filter("weight"))
            elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
                module.apply(nn.init.constant(1.0), filter_fn=name_filter("weight"))
                module.apply(nn.init.constant(0.0), filter_fn=name_filter("bias"))

        n_layers = model_config.num_transformer_layers
        std = (model_dim**-0.5) * ((2 * n_layers) ** -0.5)
        proj_init_fn = nn.init.normal(std=std)
        updates = []
        for param_name, param in mlx.utils.tree_flatten(self.parameters()):
            if param_name.endswith("out_proj.weight") or param_name.endswith(
                "ffn.proj_2.weight"
            ):
                updates.append((param_name, proj_init_fn(param)))
        self.update(mlx.utils.tree_unflatten(updates))


def load_model(
    model_dir: pathlib.Path,
) -> Tuple[OpenELM, sentencepiece.SentencePieceProcessor]:
    assert model_dir.is_dir(), f"{model_dir} must be a directory."
    weights_path = model_dir / "weights.safetensors"
    config_path = model_dir / "config.json"

    # Load tokenizer.
    tokenizer_path = model_dir / "tokenizer.model"
    if not tokenizer_path.is_file():
        raise ValueError(
            f"Tokenizer model not found at {tokenizer_path}. Please copy "
            "the LLaMA tokenizer.model file into the model directory."
        )
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer_path))

    # Load the model.
    with config_path.open("r") as f:
        config = json.load(f)
    quantization = config.pop("quantization", None)
    model = OpenELM(**config)
    if quantization is not None:
        nn.QuantizedLinear.quantize_module(model, **quantization)
    with weights_path.open("rb") as f:
        model_state = mx.load(f)
    model_state = mlx.utils.tree_unflatten(list(model_state.items()))
    model.update(model_state)
    mx.eval(model.parameters())

    return model, tokenizer
