#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
from typing import Dict

from corenet.utils import logger


def get_configuration(opts: argparse.Namespace) -> Dict:
    """
    Gets the ViT model configuration.

    The 'tiny' and 'small' model configurations were used in the
    `"Training data-efficient image transformers & distillation through attention" <https://arxiv.org/abs/2012.12877>`_ paper.

    The 'base', 'large', and 'huge' variants were used in the
    `"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" <https://arxiv.org/abs/2010.11929>`_ paper.

    Args:
        opts: Command line options.

    Returns:
        vit_config: The ViT model configuration dict.
    """
    mode = getattr(opts, "model.classification.vit.mode", "tiny")
    if not mode:
        logger.error("Please specify mode")

    mode = mode.lower()
    dropout = getattr(opts, "model.classification.vit.dropout", 0.0)
    norm_layer = getattr(opts, "model.classification.vit.norm_layer", "layer_norm")

    vit_config = dict()
    if mode == "test":
        vit_config = {
            "embed_dim": 128,
            "n_transformer_layers": 1,
            "n_attn_heads": 2,
            "ffn_dim": 128,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.1,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    elif mode == "tiny":
        vit_config = {
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
        vit_config = {
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
        vit_config = {
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
    elif mode == "large":
        vit_config = {
            "embed_dim": 1024,
            "n_transformer_layers": 24,
            "n_attn_heads": 16,
            "ffn_dim": 1024 * 4,
            "norm_layer": norm_layer,
            "pos_emb_drop_p": 0.0,
            "attn_dropout": 0.0,
            "ffn_dropout": 0.0,
            "dropout": dropout,
        }
    elif mode == "huge":
        vit_config = {
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
        logger.error("Got unsupported ViT configuration: {}".format(mode))
    return vit_config
