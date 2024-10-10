#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import copy

import pytest
import torch

from corenet.modeling.models import get_model
from tests.configs import get_config


def set_cache_transforms_to_identity(model) -> None:
    model.predicter.auxkv_to_basekv.set_as_identity()


def get_openelm_kv_prediction_model(
    arch,
    config,
):
    opts = get_config(
        config_file="tests/modeling/models/language_modeling/config/kv_prediction_config.yaml"
    )

    for k, v in config.items():
        setattr(opts, k, v)

    for elem in ["general_gpt"]:
        getattr(opts, "model.language_modeling.kv_prediction.base_model")[0]["model"][
            "language_modeling"
        ][elem]["model_name"] = arch[0]
        getattr(opts, "model.language_modeling.kv_prediction.auxiliary_model")[0][
            "model"
        ]["language_modeling"][elem]["model_name"] = arch[1]

    model = get_model(opts)
    model.eval()
    return model


@torch.no_grad()
@pytest.mark.parametrize(
    "arch,config",
    (
        [
            ["gpt-test-base", "gpt-test-aux"],
            {
                "model.language_modeling.kv_prediction.auxkv_num_layers_to_basekv_num_layers": [
                    0,
                    0,
                    1,
                ],
            },
        ],
    ),
)
def test_kv_prediction(arch, config) -> None:
    batch_size = 3
    seq_length = 5
    vocab_size = 16
    max_context_length = seq_length

    config["model.language_modeling.general_gpt.vocab_size"] = vocab_size
    config["model.language_modeling.general_gpt.max_context_length"] = (
        max_context_length
    )
    model = get_openelm_kv_prediction_model(
        arch,
        config,
    )

    model_inputs_outputs = [
        # pre-training
        {
            "name": "pre-training",
            "input": torch.randint(2, vocab_size, size=(batch_size, seq_length)),
            "expected_output_size": {
                "logits": [batch_size, seq_length, vocab_size],
                "auxiliary_logits": [batch_size, seq_length, vocab_size],
                "past_keys": [
                    [
                        batch_size,
                        model.base.k_num_heads_at_layer(i),
                        seq_length,
                        model.base.head_dim_at_layer(i),
                    ]
                    for i in range(len(model.base.layers))
                ],
                "past_values": [
                    [
                        batch_size,
                        model.base.v_num_heads_at_layer(i),
                        seq_length,
                        model.base.head_dim_at_layer(i),
                    ]
                    for i in range(len(model.base.layers))
                ],
            },
            "expected_output_type": "dictionary",
        },
        # prefix (same as pre-training, but use dictionary format for inputs)
        {
            "name": "prefix (same as pre-training, but use dictionary format for inputs)",
            "input": {
                "input_ids": torch.randint(
                    2, vocab_size, size=(batch_size, seq_length)
                ),
                "past_keys": None,
                "past_values": None,
                "use_kv_cache": False,
                "is_causal": True,
            },
            "expected_output_size": {
                "logits": [batch_size, seq_length, vocab_size],
                "auxiliary_logits": [batch_size, seq_length, vocab_size],
                "past_keys": [
                    [
                        batch_size,
                        model.base.k_num_heads_at_layer(i),
                        seq_length,
                        model.base.head_dim_at_layer(i),
                    ]
                    for i in range(len(model.base.layers))
                ],
                "past_values": [
                    [
                        batch_size,
                        model.base.v_num_heads_at_layer(i),
                        seq_length,
                        model.base.head_dim_at_layer(i),
                    ]
                    for i in range(len(model.base.layers))
                ],
            },
            "expected_output_type": "dictionary",
        },
        # prefix with KV caching
        {
            "name": "prefix with KV caching",
            "input": {
                "input_ids": torch.randint(
                    2, vocab_size, size=(batch_size, seq_length)
                ),
                "past_keys": None,
                "past_values": None,
                "use_kv_cache": True,
                "is_causal": True,
            },
            "expected_output_size": {
                "logits": [batch_size, seq_length, vocab_size],
                "past_keys": [
                    [
                        batch_size,
                        model.base.k_num_heads_at_layer(i),
                        seq_length,
                        model.base.head_dim_at_layer(i),
                    ]
                    for i in range(len(model.base.layers))
                ],
                "past_values": [
                    [
                        batch_size,
                        model.base.v_num_heads_at_layer(i),
                        seq_length,
                        model.base.head_dim_at_layer(i),
                    ]
                    for i in range(len(model.base.layers))
                ],
            },
            "expected_output_type": "dictionary",
        },
        # Generation with KV caching (input sequence length is 1)
        {
            "name": "Generation with KV caching (input sequence length is 1)",
            "input": {
                "input_ids": torch.randint(2, vocab_size, size=(batch_size, 1)),
                "past_keys": [
                    torch.randint(
                        2,
                        vocab_size,
                        size=(
                            batch_size,
                            model.base.k_num_heads_at_layer(i),
                            seq_length,
                            model.base.head_dim_at_layer(i),
                        ),
                    )
                    for i in range(len(model.base.layers))
                ],
                "past_values": [
                    torch.randint(
                        2,
                        vocab_size,
                        size=(
                            batch_size,
                            model.base.k_num_heads_at_layer(i),
                            seq_length,
                            model.base.head_dim_at_layer(i),
                        ),
                    )
                    for i in range(len(model.base.layers))
                ],
                "use_kv_cache": True,
                "is_causal": True,
            },
            "expected_output_size": {
                "logits": [batch_size, 1, vocab_size],
                # expected 1 more token in kv cache
                "past_keys": [
                    [
                        batch_size,
                        model.base.k_num_heads_at_layer(i),
                        seq_length + 1,
                        model.base.head_dim_at_layer(i),
                    ]
                    for i in range(len(model.base.layers))
                ],
                "past_values": [
                    [
                        batch_size,
                        model.base.v_num_heads_at_layer(i),
                        seq_length + 1,
                        model.base.head_dim_at_layer(i),
                    ]
                    for i in range(len(model.base.layers))
                ],
            },
            "expected_output_type": "dictionary",
        },
    ]

    for model_io in model_inputs_outputs:
        out = model(model_io["input"])
        if model_io["expected_output_type"] == "tensor":
            assert isinstance(out, torch.Tensor)
            assert list(out.size()) == model_io["expected_output_size"]
            assert torch.all(torch.isfinite(out))
        elif model_io["expected_output_type"] == "dictionary":
            assert isinstance(out, dict)
            assert set(out.keys()) in (
                set(model_io["expected_output_size"].keys()),
                set(model_io["expected_output_size"].keys())
                | {"base_past_keys", "base_past_values"},
            )
            for k_name, expected_out_size in model_io["expected_output_size"].items():
                if k_name in ["past_keys", "past_values"]:
                    actual_out_size = [
                        list(out[k_name][i].size()) for i in range(len(out[k_name]))
                    ]
                    assert all(
                        [
                            torch.all(torch.isfinite(out[k_name][i]))
                            for i in range(len(out[k_name]))
                        ]
                    )
                else:
                    actual_out_size = list(out[k_name].size())
                    assert torch.all(torch.isfinite(out[k_name]))

                assert actual_out_size == expected_out_size, f"{k_name=},{model_io=}"


@torch.no_grad()
@pytest.mark.parametrize(
    "arch,config,test_outputs_identical",
    (
        [
            ["gpt-test-base", "gpt-test-aux"],
            {
                "model.language_modeling.kv_prediction.auxkv_num_layers_to_basekv_num_layers": [
                    0,
                    0,
                    1,
                ],
            },
            False,
        ],
        [
            ["gpt-test-base", "gpt-test-base"],
            {
                "model.language_modeling.kv_prediction.auxkv_num_layers_to_basekv_num_layers": [
                    0,
                    1,
                    2,
                ],
            },
            True,
        ],
    ),
)
def test_cache_behavior(
    arch,
    config,
    test_outputs_identical,
) -> None:
    batch_size = 3
    seq_length = 5
    vocab_size = 16
    max_context_length = seq_length

    batch_size = 3
    seq_length = 5
    vocab_size = 16
    max_context_length = seq_length

    config["model.language_modeling.general_gpt.vocab_size"] = vocab_size
    config["model.language_modeling.general_gpt.max_context_length"] = (
        max_context_length
    )
    model = get_openelm_kv_prediction_model(arch, config)

    if test_outputs_identical:
        model.base.load_state_dict(model.auxiliary.state_dict())
        set_cache_transforms_to_identity(model)

    storage = {}

    def make_record_forward(model):
        old_forward = model.forward

        def forward(*args2, **kwargs2):
            storage.clear()
            storage["args"] = copy.deepcopy(args2)
            storage["kwargs"] = copy.deepcopy(kwargs2)
            return old_forward(*args2, **kwargs2)

        return forward

    model.base.forward = make_record_forward(model.base)

    position_ids = torch.arange(
        0,
        seq_length,
    ).unsqueeze(0)
    # Case 1: use_kv_cache is False. In this case, we
    # pass the KV cache to the Base model after computing
    # it.
    inputs = {
        "input_ids": torch.randint(2, vocab_size, size=(batch_size, seq_length)),
        "position_ids": position_ids,
        "past_keys": None,
        "past_values": None,
        "use_kv_cache": False,
        "is_causal": True,
    }
    base_only_outputs = model(inputs, base_only=True)

    outputs1 = model(inputs)
    assert storage["args"][0]["input_ids"].shape == (batch_size, seq_length)
    assert len(storage["args"][0]["past_keys"]) == len(model.base.layers)
    assert (storage["args"][0]["past_keys"][0]).shape[0] == (batch_size)
    assert (storage["args"][0]["past_keys"][0]).shape[2] == (seq_length)
    assert (storage["args"][0]["past_keys"][0]).dim() == (4)
    assert len(storage["args"][0]["past_values"]) == len(model.base.layers)
    assert (storage["args"][0]["past_values"][0]).shape[0] == (batch_size)
    assert (storage["args"][0]["past_values"][0]).shape[2] == (seq_length)
    assert (storage["args"][0]["past_values"][0]).dim() == (4)
    assert (storage["args"][0]["use_kv_cache"]) is True
    assert (storage["args"][0]["is_causal"]) is True
    assert storage["kwargs"]["concat_kvs"] is False
    assert storage["kwargs"]["apply_k_norm_to_past_keys_before_cache_write"] is True
    assert storage["kwargs"]["apply_k_norm_before_cache_write"] is True

    # Case 2: We are using the cache, and we are in prompt-processing mode.
    # We will first generate the KV cache (which is not yet present in @inputs),
    # then we will call the base model to generate 1 token.
    # Since storage["args"] stores only the inputs to the base model,
    # the sequence length checked below is 1.
    inputs.update({"use_kv_cache": True})
    outputs2 = model(inputs)
    assert storage["args"][0]["input_ids"].shape == (batch_size, 1)  # Seq length is 1.
    assert len(storage["args"][0]["past_keys"]) == len(model.base.layers)
    assert (storage["args"][0]["past_keys"][0]).shape[0] == (batch_size)
    assert (storage["args"][0]["past_keys"][0]).shape[2] == (seq_length - 1)
    assert (storage["args"][0]["past_keys"][0]).dim() == (4)
    assert len(storage["args"][0]["past_values"]) == len(model.base.layers)
    assert (storage["args"][0]["past_values"][0]).shape[0] == (batch_size)
    assert (storage["args"][0]["past_values"][0]).shape[2] == (seq_length - 1)
    assert (storage["args"][0]["past_values"][0]).dim() == (4)
    assert (storage["args"][0]["use_kv_cache"]) is True
    assert (storage["args"][0]["is_causal"]) is False
    assert storage["kwargs"]["concat_kvs"] is True
    assert storage["kwargs"]["apply_k_norm_to_past_keys_before_cache_write"] is True
    assert storage["kwargs"]["apply_k_norm_before_cache_write"] is True

    # Case 3: We are using the cache, and we are in generation mode.
    # To get an appropriate KV cache, we must first call the model.
    inputs3a = copy.deepcopy(inputs)
    inputs3a.update(
        {
            "input_ids": inputs["input_ids"][:, :-2],
            "position_ids": inputs["position_ids"][:, :-2],
        }
    )
    outputs3a = model(inputs3a)

    inputs3b = copy.deepcopy(inputs)
    inputs3b.update(
        {
            "input_ids": inputs["input_ids"][:, -2:-1],
            "position_ids": inputs["position_ids"][:, -2:-1],
            "past_keys": outputs3a["past_keys"],
            "past_values": outputs3a["past_values"],
        }
    )
    outputs3b = model(inputs3b)

    inputs3c = copy.deepcopy(inputs)
    inputs3c.update(
        {
            "input_ids": inputs["input_ids"][:, -1:],
            "position_ids": inputs["position_ids"][:, -1:],
            "past_keys": outputs3b["past_keys"],
            "past_values": outputs3b["past_values"],
        }
    )
    outputs3c = model(inputs3c)

    assert storage["args"][0]["input_ids"].shape == (batch_size, 1)  # Seq length is 1.
    assert len(storage["args"][0]["past_keys"]) == len(model.base.layers)
    assert (storage["args"][0]["past_keys"][0]).shape[0] == (batch_size)
    assert (storage["args"][0]["past_keys"][0]).shape[2] == (seq_length - 1)
    assert (storage["args"][0]["past_keys"][0]).dim() == (4)
    assert len(storage["args"][0]["past_values"]) == len(model.base.layers)
    assert (storage["args"][0]["past_values"][0]).shape[0] == (batch_size)
    assert (storage["args"][0]["past_values"][0]).shape[2] == (seq_length - 1)
    assert (storage["args"][0]["past_values"][0]).dim() == (4)
    assert (storage["args"][0]["use_kv_cache"]) is True
    assert (storage["args"][0]["is_causal"]) is False
    assert storage["kwargs"]["concat_kvs"] is True
    assert storage["kwargs"]["apply_k_norm_to_past_keys_before_cache_write"] is False
    assert storage["kwargs"]["apply_k_norm_before_cache_write"] is True

    if test_outputs_identical:
        assert torch.allclose(base_only_outputs, outputs1["logits"], atol=1e-5)
        assert torch.allclose(outputs1["logits"], outputs2["logits"], atol=1e-5)
        assert torch.allclose(
            outputs2["logits"][:, -2:-1], outputs3b["logits"], atol=1e-5
        )
        assert torch.allclose(
            outputs2["logits"][:, -1:], outputs3c["logits"], atol=1e-5
        )
