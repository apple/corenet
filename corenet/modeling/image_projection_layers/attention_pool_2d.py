#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from corenet.modeling.image_projection_layers import (
    IMAGE_PROJECTION_HEAD_REGISTRY,
    BaseImageProjectionHead,
)
from corenet.modeling.layers import MultiHeadAttention, PositionalEmbedding
from corenet.utils import logger


@IMAGE_PROJECTION_HEAD_REGISTRY.register(name="attention_pool_nchw2nc")
class AttentionPool2dHead(BaseImageProjectionHead):
    """This class implements attention pooling layer, as
    described in `Clip <https://arxiv.org/pdf/2103.00020.pdf>`_, and should be
    used for CNN-style models, including MobileViTs"""

    def __init__(self, opts, in_dim: int, out_dim: int, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

        num_embeddings = getattr(
            opts,
            "model.image_projection_head.attention_pool_nchw2nc.num_pos_embeddings",
            None,
        )
        if num_embeddings is None:
            logger.error(
                "Number of embeddings can't be None in {}. Please specify using "
                "--model.image-projection.attention-pool-2d.num-pos-embeddings argument".format(
                    self.__class__.__name__
                )
            )
        sin_pos_emb = getattr(
            opts,
            "model.image_projection_head.attention_pool_nchw2nc.use_sinusoidal_pos_embeddings",
            False,
        )
        num_heads = getattr(
            opts, "model.image_projection_head.attention_pool_nchw2nc.num_attn_heads", 8
        )

        self.use_pytorch_mha = getattr(
            opts,
            "model.image_projection_head.attention_pool_nchw2nc.use_pytorch_mha",
            False,
        )

        self.positional_embedding = PositionalEmbedding(
            opts,
            num_embeddings=num_embeddings,
            embedding_dim=in_dim,
            padding_idx=None,
            is_learnable=not sin_pos_emb,
            sequence_first=self.use_pytorch_mha,
        )
        self.multi_head_attn = MultiHeadAttention(
            embed_dim=in_dim, num_heads=num_heads, output_dim=out_dim
        )

        self.embed_dim = in_dim
        self.projection_dim = out_dim if out_dim is not None else in_dim
        self.sin_pos_emb = sin_pos_emb
        self.normalize_features = not getattr(
            opts,
            "model.image_projection_head.attention_pool_nchw2nc.no_feature_normalization",
            False,
        )

        self.reset_parameters()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.image-projection-head.attention-pool-nchw2nc.num-pos-embeddings",
            type=int,
            default=None,
            help="Number of positional embeddings",
        )

        group.add_argument(
            "--model.image-projection-head.attention-pool-nchw2nc.use-sinusoidal-pos-embeddings",
            action="store_true",
            help="Use sinusoidal positional embeddings instead of learnable",
        )

        group.add_argument(
            "--model.image-projection-head.attention-pool-nchw2nc.num-attn-heads",
            type=int,
            default=8,
            help="Number of attention heads in {}".format(cls.__name__),
        )

        group.add_argument(
            "--model.image-projection-head.attention-pool-nchw2nc.no-feature-normalization",
            action="store_true",
            help="Don't normalize image features",
        )

        group.add_argument(
            "--model.image-projection-head.attention-pool-nchw2nc.use-pytorch-mha",
            action="store_true",
            help="Use Pytorch Multi-head attention",
        )

        return parser

    def reset_parameters(self):
        std = self.projection_dim**-0.5
        nn.init.normal_(self.multi_head_attn.qkv_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.multi_head_attn.out_proj.weight, mean=0.0, std=std)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:

        assert (
            x.dim() == 4
        ), "Input should be 4-dimensional (Batch, in_channels, height, width). Got: {}".format(
            x.shape
        )
        # x is [batch, in_channels, height, width]
        # For CNN-style architectures, including MobileViTs
        batch_size, in_channels, in_height, in_width = x.shape

        # Flatten the feature map
        # [batch, in_channels, height, width] --> [batch, in_channels, height*width]
        x = x.reshape(batch_size, in_channels, in_height * in_width)

        if self.use_pytorch_mha:
            # we need sequence first.
            # [batch, in_channels, height*width] --> [height*width, batch, in_channels]
            x = x.permute(2, 0, 1)

            # global pool
            # [height*width, batch, in_channels] --> [1, batch, in_channels]
            global_token = torch.mean(x, dim=0, keepdim=True)

            num_pixels = x.shape[0]

            # add positional embedding to pixels
            pos_emb = self.positional_embedding(num_pixels).to(
                device=x.device, dtype=x.dtype
            )
            x = x + pos_emb

            # concat the global token with pixel tokens
            # [1, batch, in_channels] || [height*width, batch, in_channels] --> [1 + height*width, batch, in_channels]
            x = torch.cat([global_token, x], dim=0)

            # do attention
            x = self.multi_head_attn(x, use_pytorch_mha=True)

            # extract embeddings corresponding to global token
            x = x[0]
        else:
            # [batch, in_channels, height*width] --> # [batch, height*width, in_channels]
            x = x.transpose(1, 2)

            # global pool
            # [batch, height*width, in_channels] --> [batch, 1, in_channels]
            global_token = torch.mean(x, dim=1, keepdim=True)

            num_pixels = x.shape[1]
            # add positional embedding to pixels
            pos_emb = self.positional_embedding(num_pixels).to(
                device=x.device, dtype=x.dtype
            )
            x = x + pos_emb

            # concat the global token with pixel tokens
            # [batch, 1, in_channels] || [batch, height*width, in_channels] --> [batch, 1 + height*width, in_channels]
            x = torch.cat([global_token, x], dim=1)

            # do attention
            x = self.multi_head_attn(x, use_pytorch_mha=False)

            # extract embeddings corresponding to global token
            x = x[:, 0]

        if self.normalize_features:
            x = F.normalize(x, dim=-1)
        return x
