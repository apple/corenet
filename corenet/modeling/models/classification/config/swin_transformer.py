#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Dict

from corenet.utils import logger


def get_configuration(opts) -> Dict:
    mode = getattr(opts, "model.classification.swin.mode", "tiny")
    if mode is None:
        logger.error("Please specify mode")

    stochastic_depth_prob = getattr(
        opts, "model.classification.swin.stochastic_depth_prob", None
    )

    if stochastic_depth_prob is None:
        default_stochastic_depth_prob = {"tiny": 0.2, "small": 0.3, "base": 0.5}
        stochastic_depth_prob = default_stochastic_depth_prob[mode]

    mode = mode.lower()
    if mode == "tiny":
        config = {
            "patch_size": (4, 4),
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": [7, 7],
            "stochastic_depth_prob": stochastic_depth_prob,  # 0.2
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "norm_layer": "layer_norm",
        }
    elif mode == "small":
        config = {
            "patch_size": (4, 4),
            "embed_dim": 96,
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24],
            "window_size": [7, 7],
            "stochastic_depth_prob": stochastic_depth_prob,  # 0.3
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "norm_layer": "layer_norm",
        }
    elif mode == "base":
        config = {
            "patch_size": (4, 4),
            "embed_dim": 128,
            "depths": [2, 2, 18, 2],
            "num_heads": [4, 8, 16, 32],
            "window_size": [7, 7],
            "stochastic_depth_prob": stochastic_depth_prob,  # 0.5
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "norm_layer": "layer_norm",
        }
    else:
        raise NotImplementedError

    return config
