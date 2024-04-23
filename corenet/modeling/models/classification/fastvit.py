#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import copy
from typing import List

import torch.nn as nn

from corenet.modeling.layers import GlobalPool, Identity, LinearLayer
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.modeling.models.classification.config.fastvit import get_configuration
from corenet.modeling.modules.fastvit import (
    AttentionBlock,
    PatchEmbed,
    RepMixerBlock,
    convolutional_stem,
)
from corenet.modeling.modules.mobileone_block import MobileOneBlock
from corenet.utils import logger


def basic_blocks(
    opts: argparse.Namespace,
    dim: int,
    block_index: int,
    num_blocks: List[int],
    token_mixer_type: str,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    inference_mode: bool = False,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
) -> nn.Sequential:
    """Build FastViT blocks within a stage.

    Args:
        opts: Command line arguments.
        dim: Number of embedding dimensions.
        block_index: block index.
        num_blocks: List containing number of blocks per stage.
        token_mixer_type: Token mixer type.
        kernel_size: Kernel size for repmixer.
        mlp_ratio: MLP expansion ratio.
        drop_rate: Dropout rate.
        drop_path_rate: Drop path rate.
        inference_mode: Flag to instantiate block in inference mode.
        use_layer_scale: Flag to turn on layer scale regularization.
        layer_scale_init_value: Layer scale value at initialization.

    Returns:
        nn.Sequential object of all the blocks within the stage.
    """
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = (
            drop_path_rate
            * (block_idx + sum(num_blocks[:block_index]))
            / (sum(num_blocks) - 1)
        )
        if token_mixer_type == "repmixer":
            blocks.append(
                RepMixerBlock(
                    opts,
                    dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    inference_mode=inference_mode,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        elif token_mixer_type == "attention":
            blocks.append(
                AttentionBlock(
                    opts,
                    dim,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        else:
            raise ValueError(
                "Token mixer type: {} not supported".format(token_mixer_type)
            )

    blocks = nn.Sequential(*blocks)
    return blocks


@MODEL_REGISTRY.register(name="fastvit", type="classification")
class FastViT(BaseImageEncoder):
    """
    This class implements `FastViT architecture <todo: add arxiv link here>`_
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        cfg = get_configuration(opts=opts)
        super().__init__(opts, *args, **kwargs)

        # Warn users if model is instantiated in inference mode.
        inference_mode = getattr(opts, "model.classification.fastvit.inference_mode")
        if inference_mode:
            logger.warning(
                'Model instantiated in "Inference mode". '
                "This is not a desired mode for training."
            )

        # Get metaformer parameters
        self.opts = opts
        image_channels = 3
        layers = cfg["layers"]
        pos_embs = cfg["pos_embs"]
        embed_dims = cfg["embed_dims"]
        token_mixers = cfg["token_mixers"]
        mlp_ratios = cfg["mlp_ratios"]

        # Patch embedding configurations
        downsamples = cfg["downsamples"]
        down_patch_size = cfg["down_patch_size"]
        down_stride = cfg["down_stride"]

        # Get regularization parameters
        drop_rate = getattr(opts, "model.classification.fastvit.dropout")
        drop_path_rate = getattr(opts, "model.classification.fastvit.drop_path")
        use_layer_scale = getattr(opts, "model.classification.fastvit.use_layer_scale")
        layer_scale_init_value = getattr(
            opts, "model.classification.fastvit.layer_scale_init_value"
        )

        if pos_embs is None:
            pos_embs = [None] * len(cfg["layers"])

        # convolutional stem
        self.model_conf_dict = dict()
        self.conv_1 = convolutional_stem(opts, image_channels, embed_dims[0])
        self.model_conf_dict["conv1"] = {"in": image_channels, "out": embed_dims[0]}

        self.layer_1 = Identity()
        self.model_conf_dict["layer1"] = {"in": embed_dims[0], "out": embed_dims[0]}

        # Build all stages of the network.
        network = []
        for i in range(len(layers)):
            blocks_per_stage = []
            # Add position embeddings if requested
            if pos_embs[i] is not None:
                blocks_per_stage.append(pos_embs[i](opts, embed_dims[i], embed_dims[i]))
            stage = basic_blocks(
                opts,
                embed_dims[i],
                i,
                layers,
                token_mixer_type=token_mixers[i],
                kernel_size=cfg["repmixer_kernel_size"],
                mlp_ratio=mlp_ratios[i],
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                inference_mode=inference_mode,
            )
            blocks_per_stage.append(stage)
            if i >= len(layers) - 1:
                network.append(nn.Sequential(*blocks_per_stage))
                self.model_conf_dict["layer{}".format(i + 2)] = {
                    "in": embed_dims[i],
                    "out": embed_dims[i],
                }
                break

            # Downsampling+PatchEmb. between two stages
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                blocks_per_stage.append(
                    PatchEmbed(
                        opts=opts,
                        patch_size=down_patch_size,
                        stride=down_stride,
                        in_channels=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                    )
                )

            network.append(nn.Sequential(*blocks_per_stage))
            self.model_conf_dict["layer{}".format(i + 2)] = {
                "in": embed_dims[i],
                "out": embed_dims[i + 1],
            }

        self.layer_2, self.layer_3, self.layer_4, self.layer_5 = network

        # Build 1x1 exp
        self.conv_1x1_exp = nn.Sequential(
            *[
                MobileOneBlock(
                    opts=opts,
                    in_channels=embed_dims[-1],
                    out_channels=int(embed_dims[-1] * cfg["cls_ratio"]),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=embed_dims[-1],
                    inference_mode=inference_mode,
                    use_se=True,
                    num_conv_branches=1,
                )
            ]
        )
        self.model_conf_dict["exp_before_cls"] = {
            "in": embed_dims[-1],
            "out": int(embed_dims[-1] * cfg["cls_ratio"]),
        }

        # Build classifier
        pool_type = getattr(opts, "model.layer.global_pool")
        self.classifier = nn.Sequential()
        self.classifier.add_module(
            name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False)
        )
        self.classifier.add_module(
            name="classifier_fc",
            module=LinearLayer(
                in_features=int(embed_dims[-1] * cfg["cls_ratio"]),
                out_features=self.n_classes,
                bias=True,
            ),
        )
        self.model_conf_dict["cls"] = {
            "in": int(embed_dims[-1] * cfg["cls_ratio"]),
            "out": self.n_classes,
        }

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add model specific arguments"""
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.fastvit.variant",
            type=str,
            default="T8",
            help="Variant string for FastViT. Default: T8",
        )
        group.add_argument(
            "--model.classification.fastvit.inference-mode",
            type=bool,
            default=False,
            help="Flag to instantiate inference mode architecture. Default: False",
        )
        group.add_argument(
            "--model.classification.fastvit.dropout",
            type=float,
            default=0.0,
            help="Dropout rate for regularization. Default: 0.0",
        )
        group.add_argument(
            "--model.classification.fastvit.drop-path",
            type=float,
            default=0.0,
            help="Drop path rate. Default: 0.0",
        )
        group.add_argument(
            "--model.classification.fastvit.use-layer-scale",
            type=bool,
            default=True,
            help="Flag to turn on layer scale regularization. Default: True",
        )
        group.add_argument(
            "--model.classification.fastvit.layer-scale-init-value",
            type=float,
            default=1e-5,
            help="Drop path rate. Default: 1e-5",
        )
        return parser

    def get_exportable_model(self) -> nn.Module:
        """
        Method returns a reparameterized model for faster inference.

        Returns:
            Reparametrized FastViT model for faster inference.
        """
        # Avoid editing original graph
        model = copy.deepcopy(self)
        for module in model.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()
        return model
