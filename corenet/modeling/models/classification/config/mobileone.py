#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Dict

from corenet.utils import logger


def get_configuration(opts: argparse.Namespace) -> Dict:
    """Get configuration of MobileOne models."""
    variant = getattr(opts, "model.classification.mobileone.variant")
    config = dict()

    if variant == "s0":
        config = {
            "num_blocks_per_stage": [2, 8, 10, 1],
            "width_multipliers": (0.75, 1.0, 1.0, 2.0),
            "num_conv_branches": 4,
            "use_se": False,
        }
    elif variant == "s1":
        config = {
            "num_blocks_per_stage": [2, 8, 10, 1],
            "width_multipliers": (1.5, 1.5, 2.0, 2.5),
            "num_conv_branches": 1,
            "use_se": False,
        }
    elif variant == "s2":
        config = {
            "num_blocks_per_stage": [2, 8, 10, 1],
            "width_multipliers": (1.5, 2.0, 2.5, 4.0),
            "num_conv_branches": 1,
            "use_se": False,
        }
    elif variant == "s3":
        config = {
            "num_blocks_per_stage": [2, 8, 10, 1],
            "width_multipliers": (2.0, 2.5, 3.0, 4.0),
            "num_conv_branches": 1,
            "use_se": False,
        }
    elif variant == "s4":
        config = {
            "num_blocks_per_stage": [2, 8, 10, 1],
            "width_multipliers": (3.0, 3.5, 3.5, 4.0),
            "num_conv_branches": 1,
            "use_se": True,
        }
    else:
        logger.error(
            "MobileOne supported variants: `s0`, `s1`, `s2`, `s3` and `s4`. Please specify variant using "
            "--model.classification.mobileone.variant flag. Got: {}".format(variant)
        )

    return config
