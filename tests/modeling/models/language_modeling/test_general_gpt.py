#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch

from corenet.modeling.models import get_model
from tests.configs import get_config


@torch.no_grad()
def test_general_gpt():
    batch_size = 2
    seq_length = 5
    vocab_size = 16
    max_context_length = seq_length

    opts = get_config(
        config_file="tests/modeling/models/language_modeling/config/gpt_config.yaml"
    )
    setattr(opts, "model.language_modeling.general_gpt.vocab_size", vocab_size)
    setattr(
        opts,
        "model.language_modeling.general_gpt.max_context_length",
        max_context_length,
    )

    model = get_model(opts)
    model.eval()

    model_inputs_outputs = [
        # pre-training
        {
            "input": torch.randint(0, vocab_size, size=(batch_size, seq_length)),
            "expected_output_size": [batch_size, seq_length, vocab_size],
            "expected_output_type": "tensor",
        },
        # prefix (same as pre-training, but use dictionary format for inputs)
        {
            "input": {
                "input_ids": torch.randint(
                    0, vocab_size, size=(batch_size, seq_length)
                ),
                "past_keys": None,
                "past_values": None,
                "use_kv_cache": False,
                "is_causal": True,
            },
            "expected_output_size": [batch_size, seq_length, vocab_size],
            "expected_output_type": "tensor",
        },
        # prefix with KV caching
        {
            "input": {
                "input_ids": torch.randint(
                    0, vocab_size, size=(batch_size, seq_length)
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
                        model.layers[0].attn.num_k_heads,
                        seq_length,
                        model.layers[0].attn.head_dim,
                    ]
                ],
                "past_values": [
                    [
                        batch_size,
                        model.layers[0].attn.num_k_heads,
                        seq_length,
                        model.layers[0].attn.head_dim,
                    ]
                ],
            },
            "expected_output_type": "dictionary",
        },
        # Generation with KV caching (input sequence length is 1)
        {
            "input": {
                "input_ids": torch.randint(0, vocab_size, size=(batch_size, 1)),
                "past_keys": [
                    torch.randint(
                        0,
                        vocab_size,
                        size=(
                            batch_size,
                            model.layers[0].attn.num_k_heads,
                            seq_length,
                            model.layers[0].attn.head_dim,
                        ),
                    )
                ],
                "past_values": [
                    torch.randint(
                        0,
                        vocab_size,
                        size=(
                            batch_size,
                            model.layers[0].attn.num_k_heads,
                            seq_length,
                            model.layers[0].attn.head_dim,
                        ),
                    )
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
                        model.layers[0].attn.num_k_heads,
                        seq_length + 1,
                        model.layers[0].attn.head_dim,
                    ]
                ],
                "past_values": [
                    [
                        batch_size,
                        model.layers[0].attn.num_k_heads,
                        seq_length + 1,
                        model.layers[0].attn.head_dim,
                    ]
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
            assert set(out.keys()) == set(model_io["expected_output_size"].keys())
            for k_name, expected_out_size in model_io["expected_output_size"].items():
                if k_name in ["past_keys", "past_values"]:
                    assert (
                        len(out[k_name]) == 1
                    ), "Only single transformer layer in test model"
                    actual_out_size = [list(out[k_name][0].size())]
                    assert torch.all(torch.isfinite(out[k_name][0]))
                else:
                    actual_out_size = list(out[k_name].size())
                    assert torch.all(torch.isfinite(out[k_name]))
                assert actual_out_size == expected_out_size
