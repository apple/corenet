#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

"""
A KV Prediction Model (https://arxiv.org/abs/2410.08391).

The model contains 2 submodels. The smaller "auxiliary"
model is used to quickly process context. The larger
"base" model is used to perform generation without
compromising accuracy.
"""

import argparse
import copy
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from corenet.modeling.models import MODEL_REGISTRY, get_model
from corenet.modeling.models.language_modeling import general_gpt
from corenet.modeling.models.language_modeling.base_lm import BaseLanguageModel
from corenet.options.parse_args import JsonValidator
from corenet.options.utils import flatten_yaml_as_dict


@dataclass
class LayerPrunedGPTConfig(general_gpt.GPTConfig):
    select_layers: Optional[List[int]] = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.select_layers is not None:
            # Remove unselected layers.
            self.num_transformer_layers = len(self.select_layers)
            self.num_query_heads = [self.num_query_heads[i] for i in self.select_layers]
            self.num_kv_heads = [self.num_kv_heads[i] for i in self.select_layers]
            self.ffn_multipliers = [self.ffn_multipliers[i] for i in self.select_layers]

    @classmethod
    def from_name(
        cls, model_name: str, vocab_size: int, max_context_length: int
    ) -> "LayerPrunedGPTConfig":
        if model_name in layer_pruned_gpt_configs:
            config = layer_pruned_gpt_configs[model_name]
        else:
            raise NotImplementedError(f"{model_name} is not yet implemented")

        config["vocab_size"] = vocab_size
        config["max_context_length"] = max_context_length
        return cls(**config)


layer_pruned_gpt_configs = {
    "gpt-test-base": dict(
        num_transformer_layers=3,
        model_dim=128,
        head_dim=64,
        num_gqa_groups=1,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multiplier to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.25, 0.75),
        qkv_multipliers=(0.25, 0.5),
    ),
    "gpt-test-aux": dict(
        num_transformer_layers=2,
        model_dim=128,
        head_dim=48,
        num_gqa_groups=1,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multiplier to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.25, 0.75),
        qkv_multipliers=(0.25, 0.5),
    ),
    "OpenELM-1_1B-0.25l": dict(
        num_transformer_layers=28,
        model_dim=2048,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multiplier to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
        select_layers=[0, 4, 8, 12, 16, 20, 24],
    ),
    "OpenELM-1_1B-0.50l": dict(
        num_transformer_layers=28 // 2,
        model_dim=2048,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multiplier to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, (4.0) - (1 / (28 - 1)) * (4.0 - 0.5)),
        qkv_multipliers=(0.5, (1.0) - (1 / (28 - 1)) * (1.0 - 0.5)),
    ),
    "OpenELM-1_1B-0.75l": dict(
        num_transformer_layers=28,
        model_dim=2048,
        head_dim=64,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multiplier to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
        select_layers=[
            0,
            1,
            2,
            4,
            5,
            6,
            8,
            9,
            10,
            12,
            13,
            14,
            16,
            17,
            18,
            20,
            21,
            22,
            24,
            25,
            26,
        ],
    ),
    "OpenELM-3B-0.25l": dict(
        num_transformer_layers=36,
        model_dim=3072,
        head_dim=128,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multiplier to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
        select_layers=[0, 4, 8, 12, 16, 20, 24, 28, 32],
    ),
    "OpenELM-3B-0.50l": dict(
        num_transformer_layers=36 // 2,
        model_dim=3072,
        head_dim=128,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multiplier to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, (4.0) - (1 / (36 - 1)) * (4.0 - 0.5)),
        qkv_multipliers=(0.5, (1.0) - (1 / (36 - 1)) * (1.0 - 0.5)),
    ),
    "OpenELM-3B-0.75l": dict(
        num_transformer_layers=36,
        model_dim=3072,
        head_dim=128,
        num_gqa_groups=4,
        normalize_qk_projections=True,
        share_input_output_layers=True,
        # Vary the FFN and QKV multiplier to create variable FFN and attention layers respectively.
        ffn_multipliers=(0.5, 4.0),
        qkv_multipliers=(0.5, 1.0),
        select_layers=[
            0,
            1,
            2,
            4,
            5,
            6,
            8,
            9,
            10,
            12,
            13,
            14,
            16,
            17,
            18,
            20,
            21,
            22,
            24,
            25,
            26,
            28,
            29,
            30,
            32,
            33,
            34,
        ],
    ),
}


@MODEL_REGISTRY.register(name="layer_pruned_general_gpt", type="language_modeling")
class LayerPrunedGeneralGPTModel(general_gpt.GeneralGPTModel):
    """Layer Pruned General GPT model.

    Functions the same as GeneralGPTModel, but provides different available
    model configurations.
    """

    config = LayerPrunedGPTConfig


class AuxLayerToBaseLayer(nn.Module):
    """
    The mapping from auxiliary layers to base layers.

    Used in KV Prediction to map auxiliary cache layers to base cache
    layers.

    Args:
        auxkv_num_layers_to_basekv_num_layers: A list of length equal to
            the number of layers in the base model. The ith element tells
            which layer in the auxiliary cache gets mapped to the ith
            layer in the base cache.
    """

    def __init__(self, auxkv_num_layers_to_basekv_num_layers: List[int]) -> None:
        super().__init__()
        self.auxkv_num_layers_to_basekv_num_layers = (
            auxkv_num_layers_to_basekv_num_layers
        )

    def forward(
        self, past_keys: List[Tensor], past_values: List[Tensor]
    ) -> List[Tensor]:
        """
        Args:
            past_keys: [ [batch size, number of key heads, sequence length, head dimension] ] * number of transformer layers,
            past_values: Same shape as past_keys.
        Returns:
            The remapped cache.
        """
        ret_keys = [past_keys[i] for i in self.auxkv_num_layers_to_basekv_num_layers]
        ret_values = [
            past_values[i] for i in self.auxkv_num_layers_to_basekv_num_layers
        ]
        return [ret_keys, ret_values]


class LinearAuxKVToBaseKV(nn.Module):
    """
    A linear mapping from KV cache elements of one dimensionality
    to KV cache elements of another dimensionality.

    Args:
        auxiliary_k_dims: A list containing the size of the auxiliary key dimension
            for each layer of the auxiliary KV cache.
        auxiliary_v_dims: A list containing the size of the auxiliary value dimension
            for each layer of the auxiliary KV cache.
        base_k_dims: A list containing the size of the base key dimension
            for each layer of the base KV cache.
        base_v_dims: A list containing the size of the base value dimension
            for each layer of the base KV cache.
        base_k_num_heads: A list containing the number of heads
            for each layer of the base KV cache.
        base_v_num_heads: A list containing the number of heads
            for each layer of the base KV cache.
    """

    def __init__(
        self,
        auxiliary_k_dims: List[int],
        auxiliary_v_dims: List[int],
        base_k_dims: List[int],
        base_v_dims: List[int],
        base_k_num_heads: List[int],
        base_v_num_heads: List[int],
    ) -> None:
        super().__init__()

        # Inputs will be [batch_size, seq_len, dim].
        self.key_transforms = nn.ModuleList(
            [
                nn.Linear(auxiliary_k_dim, base_k_dim)
                for auxiliary_k_dim, base_k_dim in zip(auxiliary_k_dims, base_k_dims)
            ]
        )
        self.value_transforms = nn.ModuleList(
            [
                nn.Linear(auxiliary_v_dim, base_v_dim)
                for auxiliary_v_dim, base_v_dim in zip(auxiliary_v_dims, base_v_dims)
            ]
        )

        self.base_k_num_heads = base_k_num_heads
        self.base_v_num_heads = base_v_num_heads

    def forward(
        self, past_keys: List[Tensor], past_values: List[Tensor]
    ) -> List[List[Tensor]]:
        """
        Args:
            past_keys: [ [batch size, number of key heads, sequence length, head dimension] ] * number of transformer layers,
            past_values: Same shape as past_keys.
        Returns:
            A list of two elements. The first contains the predicted keys, the second contains the predicted values. Their
                shapes are:
                  - [ [batch size, output number of key heads, sequence length, output head dimension] ] * number of
                    transformer layers
                  - [ [batch size, output number of value heads, sequence length, output head dimension] ] * number of
                    transformer layers
        """

        def get_transformed_cache(
            transforms,
            caches,
            output_num_heads,
        ):
            rets = []
            for layer_transform, layer_cache, output_num_head in zip(
                transforms, caches, output_num_heads
            ):
                batch_size, num_heads, seq_len, head_dim = layer_cache.shape
                layer_cache = layer_cache.permute(0, 2, 1, 3).reshape(
                    batch_size, seq_len, num_heads * head_dim
                )
                layer_cache = layer_transform(layer_cache)
                layer_cache = layer_cache.reshape(
                    batch_size, seq_len, output_num_head, -1
                )  # Inferred dim is new head dim.
                layer_cache = layer_cache.permute(0, 2, 1, 3)
                rets.append(layer_cache)
            return rets

        return [
            get_transformed_cache(
                self.key_transforms, past_keys, self.base_k_num_heads
            ),
            get_transformed_cache(
                self.value_transforms, past_values, self.base_v_num_heads
            ),
        ]

    def set_as_identity(self) -> None:
        """
        Reset this layer to compute the identity. Mainly for testing.
        """
        for t in itertools.chain(self.key_transforms, self.value_transforms):
            dim1, dim2 = t.weight.shape
            assert dim1 == dim2, f"Expected a square matrix."
            matr = torch.eye(dim1, dtype=t.weight.dtype, device=t.weight.device)
            t.weight[:, :] = matr
            t.bias.fill_(0)


class KVPredicter(nn.Module):
    """
    Class containing tools needed to perform KV prediction.

    Args:
        auxkv_num_layers_to_basekv_num_layers: A list of length equal to
            the number of layers in the base model. The ith element tells
            which layer in the auxiliary cache gets mapped to the ith
            layer in the base cache.
        auxiliary_k_dims: A list containing the size of the auxiliary key dimension
            for each layer of the auxiliary KV cache.
        auxiliary_v_dims: A list containing the size of the auxiliary value dimension
            for each layer of the auxiliary KV cache.
        base_k_dims: A list containing the size of the base key dimension
            for each layer of the base KV cache.
        base_v_dims: A list containing the size of the base value dimension
            for each layer of the base KV cache.
        base_k_num_heads: A list containing the number of heads
            for each layer of the base KV cache.
        base_v_num_heads: A list containing the number of heads
            for each layer of the base KV cache.
    """

    def __init__(
        self,
        auxkv_num_layers_to_basekv_num_layers: List[int],
        auxiliary_k_dims: List[int],
        auxiliary_v_dims: List[int],
        base_k_dims: List[int],
        base_v_dims: List[int],
        base_k_num_heads: List[int],
        base_v_num_heads: List[int],
    ) -> None:
        super().__init__()
        self.auxkv_num_layers_to_basekv_num_layers = (
            auxkv_num_layers_to_basekv_num_layers
        )
        self.auxiliary_k_dims = auxiliary_k_dims
        self.auxiliary_v_dims = auxiliary_v_dims
        self.base_k_dims = base_k_dims
        self.base_v_dims = base_v_dims
        self.base_k_num_heads = base_k_num_heads
        self.base_v_num_heads = base_v_num_heads

        self.build_kv_cache_predicters()

    @property
    def base_layers(self) -> int:
        return len(self.base_k_dims)

    @property
    def auxiliary_layers(self) -> int:
        return len(self.auxiliary_k_dims)

    def build_kv_cache_predicters(self) -> None:
        """
        Initialize the KV cache prediction modules.
        """
        # Each index should be [0, num_aux_layers).
        if not all(
            0 <= k < self.auxiliary_layers
            for k in self.auxkv_num_layers_to_basekv_num_layers
        ):
            raise ValueError(
                "Expected layer mappings in the range "
                f"[0, num_aux_layers={self.auxiliary_layers}). "
                f"Invalid layer mappings {self.auxkv_num_layers_to_basekv_num_layers=}."
            )
        self.auxiliary_to_base_layer_mapping = AuxLayerToBaseLayer(
            self.auxkv_num_layers_to_basekv_num_layers
        )

        auxkv_num_layers_to_basekv_num_layers = (
            self.auxkv_num_layers_to_basekv_num_layers
        )

        def remap(x: List[int]) -> List[int]:
            return [x[i] for i in auxkv_num_layers_to_basekv_num_layers]

        self.auxkv_to_basekv = LinearAuxKVToBaseKV(
            remap(self.auxiliary_k_dims),
            remap(self.auxiliary_v_dims),
            (self.base_k_dims),
            (self.base_v_dims),
            (self.base_k_num_heads),
            (self.base_v_num_heads),
        )

    def forward(
        self, auxiliary_outputs: Dict[str, Any]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get the predicted KV cache for the base model using the outputs
        from the auxiliary model.

        Our prediction operates on merged heads. Thus, we need to reorganize the key-value
        cache to merge heads.

        Args:
            auxiliary_outputs: A dictionary with:
                {
                    "logits": [batch size, sequence length, vocabular size],
                    "past_keys": [ [batch size, number of key heads, sequence length, head dimension] ] * num_aux_layers
                    "past_values": [ [batch size, number of value heads, sequence length, head dimension] ] * num_aux_layers
                }
        Returns:
            A tuple of 2 lists, containing the predictions of the KV cache for the base model:
                (
                    [ [batch size, number of key heads, sequence length, head dimension] ] * num_base_layers
                    [ [batch size, number of value heads, sequence length, head dimension] ] * num_base_layers
                )
        """
        intermediate_cache = self.auxiliary_to_base_layer_mapping(
            auxiliary_outputs["past_keys"], auxiliary_outputs["past_values"]
        )
        return self.auxkv_to_basekv(*intermediate_cache)


def get_overrided_opts(
    opts: argparse.Namespace, config: Dict[str, Any]
) -> argparse.Namespace:
    """
    Get a copy of @opts with overrides from @config applied.

    Args:
        opts: The opts to copy.
        config: A dictionary with overrides.

    Returns:
        A copy of @opts with the overrides applied.
    """
    ret = copy.deepcopy(opts)
    flattened_config = flatten_yaml_as_dict(config)
    for k, v in flattened_config.items():
        setattr(ret, k, v)
    return ret


@MODEL_REGISTRY.register(name="kv_prediction", type="language_modeling")
class KVPredictionLLM(BaseLanguageModel):
    """General KV Prediction LLM model.

    Args:
        opts: Command-line arguments.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

        # As a workaround, we make this a list, to prevent the parser from throwing
        # an error about unexpected keys.
        base_model_config = getattr(
            opts, "model.language_modeling.kv_prediction.base_model"
        )[0]
        base_opts = get_overrided_opts(opts, base_model_config)
        self.base = get_model(base_opts)

        auxiliary_model_config = getattr(
            opts, "model.language_modeling.kv_prediction.auxiliary_model"
        )[0]
        auxiliary_opts = get_overrided_opts(opts, auxiliary_model_config)
        self.auxiliary = get_model(auxiliary_opts)

        # Process arguments and build the KVPredicter.
        auxkv_num_layers_to_basekv_num_layers = getattr(
            opts,
            "model.language_modeling.kv_prediction.auxkv_num_layers_to_basekv_num_layers",
        )
        assert isinstance(auxkv_num_layers_to_basekv_num_layers, list)
        assert all(isinstance(x, int) for x in auxkv_num_layers_to_basekv_num_layers)
        auxiliary_k_dims = [
            self.auxiliary.k_dim_at_layer(i) for i in range(len(self.auxiliary.layers))
        ]
        auxiliary_v_dims = [
            self.auxiliary.v_dim_at_layer(i) for i in range(len(self.auxiliary.layers))
        ]
        base_k_dims = [
            self.base.k_dim_at_layer(i) for i in range(len(self.base.layers))
        ]
        base_v_dims = [
            self.base.v_dim_at_layer(i) for i in range(len(self.base.layers))
        ]
        base_k_num_heads = [
            self.base.k_num_heads_at_layer(i) for i in range(len(self.base.layers))
        ]
        base_v_num_heads = [
            self.base.v_num_heads_at_layer(i) for i in range(len(self.base.layers))
        ]

        self.predicter = KVPredicter(
            auxkv_num_layers_to_basekv_num_layers=auxkv_num_layers_to_basekv_num_layers,
            auxiliary_k_dims=auxiliary_k_dims,
            auxiliary_v_dims=auxiliary_v_dims,
            base_k_dims=base_k_dims,
            base_v_dims=base_v_dims,
            base_k_num_heads=base_k_num_heads,
            base_v_num_heads=base_v_num_heads,
        )

        self.dtype = next(self.base.parameters()).dtype
        self.predicter.to(self.dtype)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls == KVPredictionLLM:
            group = parser.add_argument_group(cls.__name__)
            group.add_argument(
                "--model.language-modeling.kv-prediction.auxkv-num-layers-to-basekv-num-layers",
                type=str,
                help="The mapping from auxiliary layers to base model layers. The element "
                "at index i is used to tell which Auxiliary layer is used to predict the KV "
                "cache at Base layer i.",
            )
            group.add_argument(
                "--model.language-modeling.kv-prediction.base-model",
                type=JsonValidator,
                help="A config for the base model.",
            )
            group.add_argument(
                "--model.language-modeling.kv-prediction.auxiliary-model",
                type=JsonValidator,
                help="A config for the auxiliary model.",
            )
        return parser

    def predict_kv_cache(self, auxiliary_outputs: Dict[str, Any]) -> List[List[Tensor]]:
        """
        Predict the base KV cache from the auxiliary KV cache.

        Args:
            auxiliary_outputs: The full outputs of the auxiliary model, as a dictionary:
                {
                    "logits": [batch size, sequence length, vocabular size],
                    "past_keys": [ [batch size, number of key heads, sequence length, head dimension] ] * number of transformer layers,
                    "past_values": [ [batch size, number of value heads, sequence length, head dimension] ] * number of transformer layers,
                }
        Returns:
            The predicted KV cache for the base model. It is a list of two elements. The first contains
            the predicted keys, the second contains the predicted values. Their shapes are:
                  - [ [batch size, output number of key heads, sequence length, output head dimension] ] * number of
                    transformer layers
                  - [ [batch size, output number of value heads, sequence length, output head dimension] ] * number of
                    transformer layers

        """
        predicted_cache = self.predicter(auxiliary_outputs)
        return [
            [x for x in predicted_cache[0]],
            [x for x in predicted_cache[1]],
        ]

    def forward(
        self, model_input: Union[Tensor, Dict[str, Tensor]], base_only: bool = False
    ) -> Tensor:
        """
        Run the forward pass of the KVPredictionLLM.

        If the KV cache is configured to be used, we assume evaluation is happening.
        In this case:
            1) If the cache is currently empty (which is a sign that we are still processing
               the prompt), we predict the KV cache with the auxiliary model and the predicter
               network (which contains linear layers). Then, we perform a step of generation.
            2) If the cache is not currently empty, we only perform a step of generation.

        If the KV cache is not configured to be used, we perform a training step. This
        involves generating a cache prediction (using the auxiliary network and predicter
        network), then generating outputs with the base model using the cache prediction.
        Gradients flow back to the auxiliary model through the predicted KV cache.

        NOTE: base_only is reserved exclusively for testing purposes.

        Args:
            model_input: A model input suitable for general_gpt.GeneralGPT.forward.
            base_only: If set, only perform the forward pass with the base model.
                Used for testing.
        Returns:
            The output logits and the updated KV cache.
        """
        # Process the inputs.
        if isinstance(model_input, dict):
            expected_input_keys = {
                "input_ids",
                "past_keys",
                "past_values",
                "use_kv_cache",
                "is_causal",
            }
            for expected_key in expected_input_keys:
                assert (
                    expected_key in model_input
                ), f"Model input does not contain {expected_key}"

            input_ids = model_input["input_ids"]
            past_keys = model_input["past_keys"]
            past_values = model_input["past_values"]
            use_kv_cache = model_input["use_kv_cache"]
            if past_keys is None:
                assert past_values is None
                past_keys = [None] * len(self.base.layers)
                past_values = [None] * len(self.base.layers)
        elif isinstance(model_input, Tensor):
            input_ids = model_input
            past_keys = [None] * len(self.base.layers)
            past_values = [None] * len(self.base.layers)
            use_kv_cache = False
        else:
            raise NotImplementedError(
                f"Supported input types are either Tensor or Dictionary. Got: {type(model_input)}."
            )

        if base_only:
            if isinstance(model_input, dict):
                is_causal = model_input.get("is_causal", True)
            else:
                is_causal = True
            return self.base(
                {
                    "input_ids": input_ids,
                    "past_keys": past_keys,
                    "past_values": past_values,
                    "use_kv_cache": use_kv_cache,
                    "is_causal": is_causal,
                }
            )

        if use_kv_cache:
            # A user-provided KV cache is given. We are in evaluation mode.
            # There are two cases. If the past_keys and past_values
            # are empty, we populate the cache with the auxiliary model, then
            # perform a generation step. If they are not empty, we simply
            # proceed with the base model in generation mode.
            cache_is_empty = past_keys[0] is None
            if cache_is_empty:
                # Populate the KV cache, then perform a single-token
                # generation step with the base model.
                past_keys = [None] * len(self.auxiliary.layers)
                past_values = [None] * len(self.auxiliary.layers)
                auxiliary_outputs = self.auxiliary(
                    {
                        "input_ids": input_ids[:, :-1],
                        "past_keys": past_keys,
                        "past_values": past_values,
                        "use_kv_cache": use_kv_cache,
                        "is_causal": True,
                    },
                    concat_kvs=True,
                    apply_k_norm_to_past_keys_before_cache_write=False,  # Do not apply K norm. It will be applied in the base model.
                    apply_k_norm_before_cache_write=False,
                )
                past_keys, past_values = self.predict_kv_cache(auxiliary_outputs)

                # Process the last token using the base model rather than the
                # auxiliary model, to ensure that only prompt processing is done
                # by the auxiliary model, and generation is done by the base
                # model.
                # NOTE: We also want to apply key norm to the past_keys, since
                # it hasn't been applied. We cannot simply apply it in the
                # @predict_kv_cache method because FSDP will not work properly
                # if we call that wrapped module individually. The key norm will
                # be applied because apply_k_norm_to_past_keys_before_cache_write is True.
                base_outputs = self.base(
                    {
                        "input_ids": input_ids[:, -1:],
                        "past_keys": past_keys,
                        "past_values": past_values,
                        "use_kv_cache": use_kv_cache,
                        "is_causal": False,  # Attend to all the KVs.
                    },
                    concat_kvs=True,
                    apply_k_norm_to_past_keys_before_cache_write=True,  # The auxiliary model didn't apply key norm. Do it with the base.
                    apply_k_norm_before_cache_write=True,
                )

                # Return the auxiliary and base logits.
                output_logits = self.merge_outputs(
                    auxiliary_outputs["logits"], base_outputs["logits"]
                )
                assert (output_logits.shape[1]) == (input_ids.shape[1])

                past_keys = base_outputs["past_keys"]
                past_values = base_outputs["past_values"]
                return {
                    "logits": output_logits,
                    "past_keys": past_keys,
                    "past_values": past_values,
                }

            else:
                # The cache is not empty. We assume the cache has already been populated
                # by the auxiliary model, and we are in "generation mode".
                assert input_ids.shape[1] == 1
                return self.base(
                    {
                        "input_ids": input_ids,
                        "past_keys": past_keys,
                        "past_values": past_values,
                        "use_kv_cache": use_kv_cache,
                        "is_causal": False,
                    },
                    concat_kvs=True,
                    apply_k_norm_to_past_keys_before_cache_write=False,
                    apply_k_norm_before_cache_write=True,
                )

        else:
            # We are not using the cache. This means we are in
            # training mode, or we are performing likelihood
            # evaluations.
            past_keys = [None] * len(self.auxiliary.layers)
            past_values = [None] * len(self.auxiliary.layers)
            auxiliary_outputs = self.auxiliary(
                {
                    "input_ids": input_ids,
                    "past_keys": past_keys,
                    "past_values": past_values,
                    "use_kv_cache": True,
                    "is_causal": True,
                },
                concat_kvs=True,
                apply_k_norm_to_past_keys_before_cache_write=False,  # Do not apply K norm. It will be applied in the base model.
                apply_k_norm_before_cache_write=False,
            )

            past_keys, past_values = self.predict_kv_cache(auxiliary_outputs)

            ret = self.base(
                {
                    "input_ids": input_ids,
                    "past_keys": past_keys,
                    "past_values": past_values,
                    "use_kv_cache": True,
                    "is_causal": True,
                },
                concat_kvs=False,
                apply_k_norm_to_past_keys_before_cache_write=True,  # The auxiliary model didn't apply key norm. Do it with the base.
                apply_k_norm_before_cache_write=True,
            )
            ret["auxiliary_logits"] = auxiliary_outputs["logits"]

            if self.training:
                # Get the ground-truth KV cache so we can compute a loss over it.
                with torch.no_grad():
                    ret2 = self.base(
                        {
                            "input_ids": input_ids,
                            "past_keys": None,
                            "past_values": None,
                            "use_kv_cache": True,
                            "is_causal": True,
                        },
                        # Make sure the key targets are normalized.
                        apply_k_norm_before_cache_write=True,
                    )
                    ret.update(
                        {
                            "base_past_keys": ret2["past_keys"],
                            "base_past_values": ret2["past_values"],
                        }
                    )

            return ret

    def merge_outputs(self, aux_logits: Tensor, base_logits: Tensor) -> Tensor:
        """
        Merge the auxiliary and base logits.

        Args:
            aux_logits: The auxiliary logits.
            base_logits: The base logits.
        Returns:
            The merged logits.
        """
        return torch.cat([aux_logits, base_logits], dim=1)

    def get_fsdp_wrap_policy(
        self,
    ) -> Callable[[torch.nn.Module, bool, int], bool]:
        """Returns the FSDP policy."""
        return self.auxiliary.get_fsdp_wrap_policy()

    def get_activation_checkpoint_submodule_class(self) -> Callable:
        """Returns the layer that should be used for activation checkpointing."""
        return self.auxiliary.get_activation_checkpoint_submodule_class()
