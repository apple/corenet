#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
from functools import partial
from typing import Dict

from corenet.modeling.modules.fastvit import RepCPE
from corenet.utils import logger


def get_configuration(opts: argparse.Namespace) -> Dict:
    """Get configuration of FastViT models."""
    variant = getattr(opts, "model.classification.fastvit.variant")
    config = dict()

    if variant == "T8":
        config = {
            "layers": [2, 2, 4, 2],
            "embed_dims": [48, 96, 192, 384],
            "mlp_ratios": [3, 3, 3, 3],
            "downsamples": [True, True, True, True],
            "pos_embs": None,
            "token_mixers": ["repmixer", "repmixer", "repmixer", "repmixer"],
            "down_patch_size": 7,
            "down_stride": 2,
            "cls_ratio": 2.0,
            "repmixer_kernel_size": 3,
        }
    elif variant == "T12":
        config = {
            "layers": [2, 2, 6, 2],
            "embed_dims": [64, 128, 256, 512],
            "mlp_ratios": [3, 3, 3, 3],
            "downsamples": [True, True, True, True],
            "pos_embs": None,
            "token_mixers": ["repmixer", "repmixer", "repmixer", "repmixer"],
            "down_patch_size": 7,
            "down_stride": 2,
            "cls_ratio": 2.0,
            "repmixer_kernel_size": 3,
        }
    elif variant == "S12":
        config = {
            "layers": [2, 2, 6, 2],
            "embed_dims": [64, 128, 256, 512],
            "mlp_ratios": [4, 4, 4, 4],
            "downsamples": [True, True, True, True],
            "pos_embs": None,
            "token_mixers": ["repmixer", "repmixer", "repmixer", "repmixer"],
            "down_patch_size": 7,
            "down_stride": 2,
            "cls_ratio": 2.0,
            "repmixer_kernel_size": 3,
        }
    elif variant == "SA12":
        config = {
            "layers": [2, 2, 6, 2],
            "embed_dims": [64, 128, 256, 512],
            "mlp_ratios": [4, 4, 4, 4],
            "downsamples": [True, True, True, True],
            "pos_embs": [None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
            "token_mixers": ["repmixer", "repmixer", "repmixer", "attention"],
            "down_patch_size": 7,
            "down_stride": 2,
            "cls_ratio": 2.0,
            "repmixer_kernel_size": 3,
        }
    elif variant == "SA24":
        config = {
            "layers": [4, 4, 12, 4],
            "embed_dims": [64, 128, 256, 512],
            "mlp_ratios": [4, 4, 4, 4],
            "downsamples": [True, True, True, True],
            "pos_embs": [None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
            "token_mixers": ["repmixer", "repmixer", "repmixer", "attention"],
            "down_patch_size": 7,
            "down_stride": 2,
            "cls_ratio": 2.0,
            "repmixer_kernel_size": 3,
        }
    elif variant == "SA36":
        config = {
            "layers": [6, 6, 18, 6],
            "embed_dims": [64, 128, 256, 512],
            "mlp_ratios": [4, 4, 4, 4],
            "downsamples": [True, True, True, True],
            "pos_embs": [None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
            "token_mixers": ["repmixer", "repmixer", "repmixer", "attention"],
            "down_patch_size": 7,
            "down_stride": 2,
            "cls_ratio": 2.0,
            "repmixer_kernel_size": 3,
        }
    elif variant == "MA36":
        config = {
            "layers": [6, 6, 18, 6],
            "embed_dims": [76, 152, 304, 608],
            "mlp_ratios": [4, 4, 4, 4],
            "downsamples": [True, True, True, True],
            "pos_embs": [None, None, None, partial(RepCPE, spatial_shape=(7, 7))],
            "token_mixers": ["repmixer", "repmixer", "repmixer", "attention"],
            "down_patch_size": 7,
            "down_stride": 2,
            "cls_ratio": 2.0,
            "repmixer_kernel_size": 3,
        }
    else:
        logger.error(
            "FastViT supported variants: `T8`, `T12`, `S12`, `SA12`, `SA24`,"
            "`SA36` and `MA36`. Please specify variant using "
            "--model.classification.fastvit.variant flag. Got: {}".format(variant)
        )

    return config
