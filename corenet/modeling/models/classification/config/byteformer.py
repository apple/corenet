#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Dict

from corenet.utils import logger


def get_configuration(opts: argparse.Namespace) -> Dict:
    """
    Get configuration parameters associated with ByteFormer.

    These parameters are similar to those of DeIT
    (https://arxiv.org/pdf/2012.12877.pdf).

    Args:
        opts: The options configuration.

    Returns:
        A dict with keys specifying the parameters needed for ByteFormer.
    """
    mode = getattr(opts, "model.classification.byteformer.mode")
    mode = mode.lower()
    dropout = getattr(opts, "model.classification.byteformer.dropout")
    norm_layer = getattr(opts, "model.classification.byteformer.norm_layer")

    byteformer_config = dict()
    if mode == "tiny":
        byteformer_config = {
            "embed_dim": 192,
            "n_transformer_layers": 12,
            "n_attn_heads": 3,
            "ffn_dim": 192 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.1,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    elif mode == "small":
        byteformer_config = {
            "embed_dim": 384,
            "n_transformer_layers": 12,
            "n_attn_heads": 6,
            "ffn_dim": 384 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    elif mode == "base":
        byteformer_config = {
            "embed_dim": 768,
            "n_transformer_layers": 12,
            "n_attn_heads": 12,
            "ffn_dim": 768 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    elif mode == "huge":
        byteformer_config = {
            "embed_dim": 1280,
            "n_transformer_layers": 32,
            "n_attn_heads": 20,  # each head dimension is 64
            "ffn_dim": 1280 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    else:
        logger.error("Got unsupported ByteFormer configuration: {}".format(mode))
    return byteformer_config
