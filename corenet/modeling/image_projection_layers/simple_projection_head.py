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
from corenet.utils import logger
from corenet.utils.ddp_utils import is_master


@IMAGE_PROJECTION_HEAD_REGISTRY.register(name="simple_projection_nc2nc")
class SimpleImageProjectionHead(BaseImageProjectionHead):
    """This class implements simple projection head"""

    def __init__(self, opts, in_dim: int, out_dim: int, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

        scale = in_dim**-0.5
        self.use_identity = (
            getattr(
                opts,
                "model.image_projection_head.simple_projection_nc2nc.identity_if_same_size",
            )
            and in_dim == out_dim
        )
        if not self.use_identity:
            self.proj = nn.Parameter(scale * torch.randn(size=(in_dim, out_dim)))
        else:
            if is_master(opts):
                logger.log(
                    f"Using identity projection for SimpleImageProjectionHead given input/out size = {in_dim}."
                )
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.feature_normalization = not getattr(
            opts,
            "model.image_projection_head.simple_projection_nc2nc.no_feature_normalization",
        )

        self.reset_parameters()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(title=cls.__name__)

        group.add_argument(
            "--model.image-projection-head.simple-projection-nc2nc.no-feature-normalization",
            action="store_true",
            help="Don't normalize image features. Defaults to False.",
        )

        group.add_argument(
            "--model.image-projection-head.simple-projection-nc2nc.identity-if-same-size",
            action="store_true",
            help="Use identity projection when projection input/output dims"
            " are the same. Defaults to False",
        )

        return parser

    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        # x is of shape [batch, in_dim]
        assert (
            x.dim() == 2
        ), "Input should be 2-dimensional (Batch x in_dim). Got: {}".format(x.shape)

        if not self.use_identity:
            # [batch, in_dim] x [in_dim, out_dim] --> [batch, out_dim]
            x = x @ self.proj
        if self.feature_normalization:
            x = F.normalize(x, dim=-1)
        return x
