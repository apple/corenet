#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Dict

import torch

from corenet.modeling.models import get_model
from corenet.third_party.modeling import lora
from tests.configs import get_config


def get_opts(config_file_name: str) -> Dict:
    opts = get_config(config_file_name)

    # Override with LoRA settings.
    setattr(opts, "model.freeze_modules", [".*base_layer.*", ".*norm.*"])
    setattr(opts, "model.lora.use_lora", True)

    config = [
        {
            "regex": r".*token_embedding.*",
            "module_type": "embedding",
            "params": {
                "adapter_name": "lora",
                "r": 5,
                "lora_alpha": 2,
                "lora_dropout": 0.1,
                "init_lora_weights": True,
                "use_rslora": False,
                "use_dora": False,
            },
        },
        {
            "regex": r".*out_proj.*",
            "module_type": "linear",
            "params": {
                "adapter_name": "lora",
                "r": 5,
                "lora_alpha": 2,
                "lora_dropout": 0.1,
                "init_lora_weights": True,
                "use_rslora": False,
                "use_dora": True,
            },
        },
        {
            "regex": r".*qkv_proj.*",
            "module_type": "linear",
            "params": {
                "adapter_name": "lora",
                "r": 5,
                "lora_alpha": 2,
                "lora_dropout": 0.1,
                "init_lora_weights": True,
                "use_rslora": False,
                "use_dora": True,
            },
        },
        {
            "regex": r".*proj_\d.*",
            "module_type": "linear",
            "params": {
                "adapter_name": "lora",
                "r": 5,
                "lora_alpha": 2,
                "lora_dropout": 0.1,
                "init_lora_weights": True,
                "use_rslora": False,
                "use_dora": True,
            },
        },
    ]
    setattr(opts, "model.lora.config", config)
    return opts


def test_build_model() -> None:
    config = "tests/modeling/models/language_modeling/config/gpt_config.yaml"
    opts = get_opts(config)
    model = get_model(opts)

    # Number of parameters.
    assert sum([param.numel() for name, param in model.named_parameters()]) == 154164
    # Number of trainable parameters.
    assert (
        sum(
            [
                param.numel()
                for name, param in model.named_parameters()
                if param.requires_grad
            ]
        )
        == 9780
    )

    vocab_size = getattr(opts, "model.language_modeling.general_gpt.vocab_size")

    batch_size, seq_len = 1, 4

    x = torch.randint(low=0, high=2, size=[batch_size, seq_len])
    y = model(x)
    assert y.shape == (batch_size, seq_len, vocab_size)
