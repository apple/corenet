#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from typing import Dict, List

from corenet.utils import logger


def add_squeeze_channels(
    config_dict: Dict, per_layer_squeeze_channels: List[int]
) -> None:
    """Given the config_dict for the specified ResNet model, for each layer, adds a new key
    ('squeeze_channels') with the corresponding channels for the squeeze-excitation module.

    Args:
        config_dict: The dict constructed by the get_configuration function.
        per_layer_squeeze_channels: A list of length 4 where the ith element specifies
            the number of channels for squeeze-excitation module of layer i.
    """
    for layer, squeeze_channels in zip(range(2, 6), per_layer_squeeze_channels):
        config_dict[f"layer{layer}"]["squeeze_channels"] = squeeze_channels


def get_configuration(opts) -> Dict:
    depth = getattr(opts, "model.classification.resnet.depth")

    # Whether to build an SE-ResNet model (https://arxiv.org/abs/1709.01507)
    se_resnet = getattr(opts, "model.classification.resnet.se_resnet")

    resnet_config = dict()

    if depth == 18:
        resnet_config["layer2"] = {
            "num_blocks": 2,
            "mid_channels": 64,
            "block_type": "basic",
            "stride": 1,
        }
        resnet_config["layer3"] = {
            "num_blocks": 2,
            "mid_channels": 128,
            "block_type": "basic",
            "stride": 2,
        }
        resnet_config["layer4"] = {
            "num_blocks": 2,
            "mid_channels": 256,
            "block_type": "basic",
            "stride": 2,
        }
        resnet_config["layer5"] = {
            "num_blocks": 2,
            "mid_channels": 512,
            "block_type": "basic",
            "stride": 2,
        }
        se_resnet_channels = [8, 8, 16, 32]
    elif depth == 34:
        resnet_config["layer2"] = {
            "num_blocks": 3,
            "mid_channels": 64,
            "block_type": "basic",
            "stride": 1,
        }
        resnet_config["layer3"] = {
            "num_blocks": 4,
            "mid_channels": 128,
            "block_type": "basic",
            "stride": 2,
        }
        resnet_config["layer4"] = {
            "num_blocks": 6,
            "mid_channels": 256,
            "block_type": "basic",
            "stride": 2,
        }
        resnet_config["layer5"] = {
            "num_blocks": 3,
            "mid_channels": 512,
            "block_type": "basic",
            "stride": 2,
        }
        se_resnet_channels = [8, 8, 16, 32]
    elif depth == 50:
        resnet_config["layer2"] = {
            "num_blocks": 3,
            "mid_channels": 64,
            "block_type": "bottleneck",
            "stride": 1,
        }
        resnet_config["layer3"] = {
            "num_blocks": 4,
            "mid_channels": 128,
            "block_type": "bottleneck",
            "stride": 2,
        }
        resnet_config["layer4"] = {
            "num_blocks": 6,
            "mid_channels": 256,
            "block_type": "bottleneck",
            "stride": 2,
        }
        resnet_config["layer5"] = {
            "num_blocks": 3,
            "mid_channels": 512,
            "block_type": "bottleneck",
            "stride": 2,
        }
        se_resnet_channels = [16, 32, 64, 128]
    elif depth == 101:
        resnet_config["layer2"] = {
            "num_blocks": 3,
            "mid_channels": 64,
            "block_type": "bottleneck",
            "stride": 1,
        }
        resnet_config["layer3"] = {
            "num_blocks": 4,
            "mid_channels": 128,
            "block_type": "bottleneck",
            "stride": 2,
        }
        resnet_config["layer4"] = {
            "num_blocks": 23,
            "mid_channels": 256,
            "block_type": "bottleneck",
            "stride": 2,
        }
        resnet_config["layer5"] = {
            "num_blocks": 3,
            "mid_channels": 512,
            "block_type": "bottleneck",
            "stride": 2,
        }
        se_resnet_channels = [16, 32, 64, 128]
    else:
        logger.error(
            "ResNet (or SE-ResNet) models are supported with depths of 18, 34, 50 and 101. Please specify depth using "
            "--model.classification.resnet.depth flag. Got: {}".format(depth)
        )

    if se_resnet:
        add_squeeze_channels(resnet_config, se_resnet_channels)

    return resnet_config
