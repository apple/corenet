#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from corenet.modeling.layers import (
    ConvLayer2d,
    Dropout,
    GlobalPool,
    Identity,
    LinearLayer,
    get_normalization_layer,
)
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.modeling.models.classification.config.swin_transformer import (
    get_configuration,
)
from corenet.modeling.modules import PatchMerging, Permute, SwinTransformerBlock
from corenet.utils import logger


@MODEL_REGISTRY.register(name="swin", type="classification")
class SwinTransformer(BaseImageEncoder):
    """
    Implements Swin Transformer from the `"Swin Transformer: Hierarchical Vision Transformer using
    Shifted Windows" <https://arxiv.org/pdf/2103.14030>`_ paper.

    The code is adapted from `"Torchvision repository" <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
    """

    def __init__(self, opts, *args, **kwargs) -> None:
        image_channels = 3
        classifier_dropout = getattr(
            opts, "model.classification.classifier_dropout", 0.0
        )
        pool_type = getattr(opts, "model.layer.global_pool", "mean")
        super().__init__(opts, *args, **kwargs)

        cfg = get_configuration(opts=opts)

        patch_size = cfg["patch_size"]
        embed_dim = cfg["embed_dim"]
        depths = cfg["depths"]
        window_size = cfg["window_size"]
        mlp_ratio = cfg["mlp_ratio"]
        num_heads = cfg["num_heads"]
        dropout = cfg["dropout"]
        attn_dropout = cfg["attn_dropout"]
        ffn_dropout = cfg["ffn_dropout"]
        stochastic_depth_prob = cfg["stochastic_depth_prob"]
        norm_layer = cfg["norm_layer"]

        # store model configuration in a dictionary
        self.model_conf_dict = dict()
        self.conv_1 = nn.Sequential(
            *[
                ConvLayer2d(
                    opts=opts,
                    in_channels=image_channels,
                    out_channels=embed_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                    use_norm=False,
                    use_act=False,
                ),
                Permute([0, 2, 3, 1]),
                get_normalization_layer(
                    opts=opts, norm_type=norm_layer, num_features=embed_dim
                ),
            ]
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": embed_dim}
        in_channels = embed_dim

        self.model_conf_dict["layer1"] = {"in": embed_dim, "out": embed_dim}

        # build SwinTransformer blocks
        layers: List[nn.Module] = []
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob
                    * float(stage_block_id)
                    / (total_stage_blocks - 1)
                )
                stage.append(
                    SwinTransformerBlock(
                        opts,
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[
                            0 if i_layer % 2 == 0 else w // 2 for w in window_size
                        ],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attn_dropout=attn_dropout,
                        ffn_dropout=ffn_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                stage += [PatchMerging(opts, dim, norm_layer)]

            layers.append(nn.Sequential(*stage))
            self.model_conf_dict["layer{}".format(i_stage + 2)] = {
                "in": in_channels,
                "out": dim,
            }
            in_channels = dim

        self.layer_1, self.layer_2, self.layer_3, self.layer_4 = layers

        # For segmentation architectures, we need to disable striding at an output stride of
        # 8 or 16. Depending on the output stride value, we disable the striding in SwinTransformer
        if self.dilate_l5:
            for m in self.layer_3.modules():
                if isinstance(m, PatchMerging):
                    m.strided = False

        if self.dilate_l4:
            for m in self.layer_2.modules():
                if isinstance(m, PatchMerging):
                    m.strided = False

        self.layer_5 = nn.Sequential(
            *[
                get_normalization_layer(
                    opts=opts, norm_type=norm_layer, num_features=in_channels
                ),
                Permute([0, 3, 1, 2]),
            ]
        )

        self.conv_1x1_exp = Identity()
        self.model_conf_dict["exp_before_cls"] = {
            "in": in_channels,
            "out": in_channels,
        }

        self.classifier = nn.Sequential()
        self.classifier.add_module(
            name="global_pool", module=GlobalPool(pool_type=pool_type, keep_dim=False)
        )
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_module(
                name="classifier_dropout", module=Dropout(p=classifier_dropout)
            )
        self.classifier.add_module(
            name="classifier_fc",
            module=LinearLayer(
                in_features=in_channels, out_features=self.n_classes, bias=True
            ),
        )

        self.model_conf_dict["cls"] = {"in": in_channels, "out": self.n_classes}

        extract_enc_point_format = getattr(
            opts, "model.classification.swin.extract_end_point_format", "nchw"
        )
        if extract_enc_point_format not in ["nchw", "nhwc"]:
            logger.error(
                "End point extraction format should be either nchw or nhwc. Got: {}".format(
                    extract_enc_point_format
                )
            )
        self.extract_end_point_nchw_format = extract_enc_point_format == "nchw"

        # check model
        self.check_model()

        # weight initialization
        self.reset_parameters(opts=opts)

    def extract_end_points_all(
        self,
        x: Tensor,
        use_l5: Optional[bool] = True,
        use_l5_exp: Optional[bool] = False,
        *args,
        **kwargs
    ) -> Dict[str, Tensor]:
        # First conv layer in SwinTransformer down samples by a factor of 4, so we modify the end-point extraction
        # function, so that the model is compatible with down-stream heads (e.g., Mask-RCNN)

        out_dict = {}  # Use dictionary over NamedTuple so that JIT is happy

        if self.training and self.neural_augmentor is not None:
            x = self.neural_augmentor(x)
            out_dict["augmented_tensor"] = x

        # [N, C, H, W] --> [N, H/4, W/4, C]
        x = self.conv_1(x)

        # first layer down-samples by 4, so L1 and l2 should be identity
        if self.extract_end_point_nchw_format:
            x_nchw = torch.permute(x, dims=(0, 3, 1, 2))
            out_dict["out_l1"] = x_nchw
            out_dict["out_l2"] = x_nchw
        else:
            out_dict["out_l1"] = x
            out_dict["out_l2"] = x

        # [N, H/4, W/4, C] --> [N, H/8, W/8, C]
        x = self.layer_1(x)
        out_dict["out_l3"] = (
            torch.permute(x, dims=(0, 3, 1, 2))
            if self.extract_end_point_nchw_format
            else x
        )

        # [N, H/8, W/8, C] --> [N, H/16, W/16, C]
        x = self.layer_2(x)
        out_dict["out_l4"] = (
            torch.permute(x, dims=(0, 3, 1, 2))
            if self.extract_end_point_nchw_format
            else x
        )

        if use_l5:
            # [N, H/16, W/16, C] --> [N, H/32, W/32, C]
            x = self.layer_3(x)
            x = self.layer_4(x)
            # [N, H/32, W/32, C] --> [N, C, H/32, W/32]
            x = self.layer_5(x)
            out_dict["out_l5"] = (
                x
                if self.extract_end_point_nchw_format
                else torch.permute(x, dims=(0, 2, 3, 1))
            )

            if use_l5_exp:
                x = self.conv_1x1_exp(x)
                out_dict["out_l5_exp"] = x
        return out_dict

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.swin.mode",
            type=str,
            default="tiny",
            help="SwinTransformer mode. Default is swin_t",
        )

        group.add_argument(
            "--model.classification.swin.stochastic-depth-prob",
            type=float,
            default=None,
        )

        group.add_argument(
            "--model.classification.swin.extract-end-point-format",
            type=str,
            default="nchw",
            choices=["nchw", "nhwc"],
            help="End point extraction format in Swin Transformer. This is useful for down-stream tasks where "
            "task-specific heads are either in nhwc format or nchw format. Defaults to nchw.",
        )

        return parser
