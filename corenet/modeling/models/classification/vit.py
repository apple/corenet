#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import functools
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from corenet.modeling.layers import (
    ConvLayer2d,
    Dropout,
    Identity,
    LinearLayer,
    MaxPool2d,
    PositionalEmbedding,
    TransposeConvLayer2d,
    get_normalization_layer,
)
from corenet.modeling.misc.common import parameter_list
from corenet.modeling.misc.init_utils import initialize_conv_layer
from corenet.modeling.models import MODEL_REGISTRY
from corenet.modeling.models.classification.base_image_encoder import BaseImageEncoder
from corenet.modeling.models.classification.config.vit import get_configuration
from corenet.modeling.modules import FlashTransformerEncoder, TransformerEncoder
from corenet.utils import logger


@MODEL_REGISTRY.register(name="vit", type="classification")
class VisionTransformer(BaseImageEncoder):
    """Vision Transformer.

    This class defines the `Vision Transformer architecture <https://arxiv.org/abs/2010.11929>`_. Our model implementation
    is inspired from `Early Convolutions Help Transformers See Better <https://arxiv.org/abs/2106.14881>`_.

    Args:
        opts: Command-line arguments.

    .. note::
        Our implementation is different from the original implementation in two ways:
        1. Kernel size is odd.
        2. Our positional encoding implementation allows us to use ViT with any multiple input scales
        3. We do not use StochasticDepth
        4. We do not add positional encoding to class token (if enabled), as suggested in `DeiT-3 paper <https://arxiv.org/abs/2204.07118>`_
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        image_channels = 3

        super().__init__(opts, *args, **kwargs)

        vit_config = get_configuration(opts)

        # Typically, in the ImageNet dataset, we use 224x224 as a resolution.
        # For our ViT implementation, we use stride of 16. Therefore, total number of patch embeddings are (224 / 16)^2
        num_patch_embeddings = (224 // 16) ** 2

        embed_dim = vit_config["embed_dim"]
        ffn_dim = vit_config["ffn_dim"]
        pos_emb_drop_p = vit_config["pos_emb_drop_p"]
        n_transformer_layers = vit_config["n_transformer_layers"]
        num_heads = vit_config["n_attn_heads"]
        attn_dropout = vit_config["attn_dropout"]
        dropout = vit_config["dropout"]
        ffn_dropout = vit_config["ffn_dropout"]
        norm_layer = vit_config["norm_layer"]

        self.patch_emb = self._build_patch_embedding_layer(
            opts, image_channels=image_channels, embedding_dimension=embed_dim
        )

        use_cls_token = not getattr(opts, "model.classification.vit.no_cls_token")
        stochastic_dropout = getattr(
            opts, "model.classification.vit.stochastic_dropout"
        )
        per_layer_stochastic_drop_rate = [
            round(x, 3)
            for x in np.linspace(0, stochastic_dropout, n_transformer_layers)
        ]

        self.post_transformer_norm = get_normalization_layer(
            opts=opts, num_features=embed_dim, norm_type=norm_layer
        )

        use_flash_attn = getattr(opts, "model.classification.vit.use_flash_attention")

        if use_flash_attn:
            transformer_build_fn = self._build_transformer_layer_with_flash_attention
        else:
            transformer_build_fn = self._build_naive_transformer_layer

        self.transformer = transformer_build_fn(
            opts,
            embedding_dimension=embed_dim,
            ffn_dimension=ffn_dim,
            num_transformer_layers=n_transformer_layers,
            num_attention_heads=num_heads,
            dropout=dropout,
            attention_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            normalization_layer_name=norm_layer,
            per_layer_stochastic_drop_rate=per_layer_stochastic_drop_rate,
        )
        self.classifier = LinearLayer(embed_dim, self.n_classes)

        self.reset_parameters(opts=opts)

        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(size=(1, 1, embed_dim)))
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        else:
            self.cls_token = None

        self.pos_embed = PositionalEmbedding(
            opts=opts,
            num_embeddings=num_patch_embeddings,
            embedding_dim=embed_dim,
            sequence_first=False,
            padding_idx=None,
            is_learnable=not getattr(
                opts, "model.classification.vit.sinusoidal_pos_emb"
            ),
            interpolation_mode="bilinear",
        )
        self.emb_dropout = Dropout(p=pos_emb_drop_p)
        self.embed_dim = embed_dim
        self.use_flash_attn = use_flash_attn

        self.model_conf_dict = {
            "conv1": {"in": image_channels, "out": embed_dim},
            "layer1": {"in": embed_dim, "out": embed_dim},
            "layer2": {"in": embed_dim, "out": embed_dim},
            "layer3": {"in": embed_dim, "out": embed_dim},
            "layer4": {"in": embed_dim, "out": embed_dim},
            "layer5": {"in": embed_dim, "out": embed_dim},
            "exp_before_cls": {"in": embed_dim, "out": embed_dim},
            "cls": {"in": embed_dim, "out": self.n_classes},
        }

        use_simple_fpn = getattr(opts, "model.classification.vit.use_simple_fpn")
        self.simple_fpn = None
        if use_simple_fpn:
            # for object detection, we add Simple FPN on top of ViT backbone, so that it can
            # generate multi-scale representations. See https://arxiv.org/abs/2203.16527 for details
            self.simple_fpn = self._build_simple_fpn_layers(opts, embed_dim, norm_layer)
            self.reset_simple_fpn_params()

        self.update_layer_norm_eps()

    def _build_patch_embedding_layer(
        self, opts: argparse.Namespace, image_channels: int, embedding_dimension: int
    ) -> nn.Sequential:
        """Wrapper to build patch embedding layer.

        Args:
            opts: Command-line arguments.
            image_channels: Number of image channels.
            embedding_dimension: Embedding dimension.

        Returns:
            A sequential container with three convolution layers.
        """

        # For classification tasks, output stride is 16, while for dense prediction tasks
        # output stride is typically 8. We adjust default stride (i.e., [4, 2, 2]) in convolutional stem
        # to [2, 2, 2] to obtain a ViT model with an output stride of 8.
        kernel_sizes_conv_stem = [4, 2, 2]
        strides_conv_stem = [4, 2, 2]
        if self.output_stride is not None and self.output_stride not in [8, 16]:
            logger.error("Output stride should be 8 or 16")
        elif self.output_stride is not None and self.output_stride == 8:
            # For classification tasks,
            strides_conv_stem[0] = 2

        conv_stem_proj_dim = max(32, embedding_dimension // 4)
        patch_emb = [
            ConvLayer2d(
                opts=opts,
                in_channels=image_channels,
                out_channels=conv_stem_proj_dim,
                kernel_size=kernel_sizes_conv_stem[0],
                stride=strides_conv_stem[0],
                bias=False,
                use_norm=True,
                use_act=True,
                norm_layer=get_normalization_layer(
                    opts=opts, num_features=conv_stem_proj_dim, norm_type="batch_norm"
                ),
            ),
            ConvLayer2d(
                opts=opts,
                in_channels=conv_stem_proj_dim,
                out_channels=conv_stem_proj_dim,
                kernel_size=kernel_sizes_conv_stem[1],
                stride=strides_conv_stem[1],
                bias=False,
                use_norm=True,
                use_act=True,
                norm_layer=get_normalization_layer(
                    opts=opts, num_features=conv_stem_proj_dim, norm_type="batch_norm"
                ),
            ),
            ConvLayer2d(
                opts=opts,
                in_channels=conv_stem_proj_dim,
                out_channels=embedding_dimension,
                kernel_size=kernel_sizes_conv_stem[2],
                stride=strides_conv_stem[2],
                bias=True,
                use_norm=False,
                use_act=False,
            ),
        ]
        return nn.Sequential(*patch_emb)

    def _build_naive_transformer_layer(
        self,
        opts: argparse.Namespace,
        embedding_dimension: int,
        ffn_dimension: float,
        num_transformer_layers: int,
        num_attention_heads: int,
        dropout: float,
        attention_dropout: float,
        ffn_dropout: float,
        normalization_layer_name: str,
        per_layer_stochastic_drop_rate: List[float],
    ) -> nn.Sequential:
        """Wrapper to build transformer layer with unoptimized self-attention.

        Args:
            opts: Command-line arguments.
            embedding_dimension: Embedding dimension.
            ffn_dimension: Feed-forward network dimension.
            num_transformer_layers: Number of transformer layers.
            num_attention_heads: Number of attention heads.
            dropout: Standard dropout.
            attention_dropout: Attention dropout.
            ffn_dropout: FFN dropout.
            normalization_layer_name: Normalization layer name.
            per_layer_stochastic_drop_rate: Stochastic dropout rate for each transformer layer.

        Returns:
            A sequential container with 'num_transformer_layers' TransformerEncoder layers.
        """

        transformer_blocks = [
            TransformerEncoder(
                opts=opts,
                embed_dim=embedding_dimension,
                ffn_latent_dim=ffn_dimension,
                num_heads=num_attention_heads,
                attn_dropout=attention_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                transformer_norm_layer=normalization_layer_name,
                stochastic_dropout=per_layer_stochastic_drop_rate[layer_idx],
            )
            for layer_idx in range(num_transformer_layers)
        ]
        return nn.Sequential(*transformer_blocks)

    def _build_transformer_layer_with_flash_attention(
        self,
        opts: argparse.Namespace,
        embedding_dimension: int,
        ffn_dimension: float,
        num_transformer_layers: int,
        num_attention_heads: int,
        dropout: float,
        attention_dropout: float,
        ffn_dropout: float,
        normalization_layer_name: str,
        per_layer_stochastic_drop_rate: List[float],
    ) -> nn.Module:
        """Wrapper to build transformer layer with flash self-attention.

        Args:
            opts: Command-line arguments.
            embedding_dimension: Embedding dimension.
            ffn_dimension: Feed-forward network dimension.
            num_transformer_layers: Number of transformer layers.
            num_attention_heads: Number of attention heads.
            dropout: Standard dropout.
            attention_dropout: Attention dropout.
            ffn_dropout: FFN dropout.
            normalization_layer_name: Normalization layer name.
            per_layer_stochastic_drop_rate: Stochastic dropout rate for each transformer layer.

        Returns:
            A sequential container with 'num_transformer_layers' FlashTransformerEncoder layers.
        """

        if embedding_dimension % num_attention_heads != 0:
            logger.error(
                f"Embedding dimension should be divisble by num_attention_heads. Got: {embedding_dimension} embedding dimension and {num_attention_heads} heads."
            )
        head_dim = embedding_dimension // num_attention_heads

        if head_dim not in [64, 128]:
            logger.error(
                f"For flash attention, we want head dim to be 64 or 128 for better efficiency, but got head_dim as {head_dim}."
            )

        if ffn_dimension % embedding_dimension != 0:
            logger.error(
                f"FFN dimension should be divisble by embedding dimension. Got FFN dimension as {ffn_dimension} and embedding dimension as {embedding_dimension}."
            )

        ffn_multiplier = ffn_dimension // embedding_dimension
        transformer_blocks = [
            FlashTransformerEncoder(
                opts=opts,
                in_features=embedding_dimension,
                head_dim=head_dim,
                attn_dropout_prob=attention_dropout,
                qkv_features=embedding_dimension,
                bias=True,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                ffn_multiplier=ffn_multiplier,
                stochastic_dropout=per_layer_stochastic_drop_rate[layer_idx],
                norm_layer_name=normalization_layer_name,
                divisible_by=16,
            )
            for layer_idx in range(num_transformer_layers)
        ]
        return nn.Sequential(*transformer_blocks)

    def update_layer_norm_eps(self) -> None:
        # Most ViT models use LayerNorm with 10^-6 eps. So, we update it here
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                m.eps = 1e-6

    def reset_simple_fpn_params(self) -> None:
        # reset simple FPN parameters
        if self.simple_fpn is not None:
            for m in self.simple_fpn.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    initialize_conv_layer(m, init_method="kaiming_uniform")

    def _apply_layer_wise_lr(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Tuple[List, List]:
        """
        This function adjusts the learning rate of each layer in transformer module.
        Layer-wise learning is a bit involved and requires a knowledge of how each layer is consumed
        during the forward pass. We adjust the learning rate of patch embedding and transformer layers
        while keeping the classifier and SimpleFPN at 1.0. This is because layer_wise_lr is typically
        applied during fine-tuning for down-stream tasks.

        For ViT (classification tasks), the path is like this:
        Patch Embedding --> Transformer --> PostNorm --> Classifier

        For ViT (detection tasks), the path is like this:
        Patch Embedding --> Transformer --> PostNorm --> SimpleFPN

        """
        n_layers = 1 + len(self.transformer)
        layer_wise_lr = [
            round(self.layer_wise_lr_decay_rate ** (n_layers - i), 5)
            for i in range(n_layers)
        ]
        module_name = kwargs.pop("module_name", "")

        param_list = []
        param_lr_list = []

        if self.neural_augmentor:
            neural_aug_params = parameter_list(
                named_parameters=self.neural_augmentor.named_parameters,
                weight_decay=weight_decay,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + "neural_augmentor.",
                *args,
                **kwargs,
            )
            param_list.extend(neural_aug_params)
            param_lr_list.extend([layer_wise_lr[0]] * len(neural_aug_params))

        # Patch embedding related parameters
        embedding_params = parameter_list(
            named_parameters=self.patch_emb.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            module_name=module_name + "patch_emb.",
            *args,
            **kwargs,
        )
        param_list.extend(embedding_params)
        param_lr_list.extend([layer_wise_lr[0]] * len(embedding_params))

        # positional embedding parameters
        pos_emb_params = parameter_list(
            named_parameters=self.pos_embed.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            module_name=module_name + "pos_embed.",
            *args,
            **kwargs,
        )
        param_list.extend(pos_emb_params)
        param_lr_list.extend([layer_wise_lr[0]] * len(pos_emb_params))

        if self.cls_token is not None:
            # CLS token params
            cls_token_params = parameter_list(
                named_parameters=self.cls_token.named_parameters,
                weight_decay=0.0,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + "cls_token.",
                *args,
                **kwargs,
            )
            param_list.extend(cls_token_params)
            param_lr_list.extend([layer_wise_lr[0]] * len(cls_token_params))

        # transformer related parameters
        for layer_id, transformer_layer in enumerate(self.transformer):
            layer_lr = layer_wise_lr[layer_id + 1]
            transformer_layer_params = parameter_list(
                named_parameters=transformer_layer.named_parameters,
                weight_decay=weight_decay,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + f"transformer.{layer_id}.",
                *args,
                **kwargs,
            )
            param_list.extend(transformer_layer_params)
            param_lr_list.extend([layer_lr] * len(transformer_layer_params))

        # transformer post-norm params
        post_transformer_norm_params = parameter_list(
            named_parameters=self.post_transformer_norm.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
            module_name=module_name + "post_transformer_norm.",
            *args,
            **kwargs,
        )
        param_list.extend(post_transformer_norm_params)
        param_lr_list.extend([layer_wise_lr[-1]] * len(post_transformer_norm_params))

        if self.classifier is not None:
            # classifier parameters
            classifier_params = parameter_list(
                named_parameters=self.classifier.named_parameters,
                weight_decay=0.0,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + "classifier.",
                *args,
                **kwargs,
            )
            param_list.extend(classifier_params)
            param_lr_list.extend([1.0] * len(classifier_params))

        if self.simple_fpn is not None:
            # simple FPN parameters
            simple_fpn_params = parameter_list(
                named_parameters=self.simple_fpn.named_parameters,
                weight_decay=0.0,
                no_decay_bn_filter_bias=no_decay_bn_filter_bias,
                module_name=module_name + "simple_fpn.",
                *args,
                **kwargs,
            )
            param_list.extend(simple_fpn_params)
            param_lr_list.extend([1.0] * len(simple_fpn_params))
        return param_list, param_lr_list

    def _build_simple_fpn_layers(
        self,
        opts: argparse.Namespace,
        embedding_dimension: int,
        normalization_layer_name: str,
    ) -> nn.ModuleDict:
        """Wrapper to build simple FPN layer.

        Args:
            opts: Command-line arguments.
            embedding_dimension: Embedding dimension.
            normalization_layer_name: Normalization layer name.

        Returns:
            A module dictionary containing convolutional layers for building SimpleFPN.
        """
        layer_l2 = nn.Sequential(
            TransposeConvLayer2d(
                opts,
                in_channels=embedding_dimension,
                out_channels=embedding_dimension // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                groups=1,
                use_norm=True,
                use_act=True,
                norm_layer=get_normalization_layer(
                    opts=opts,
                    num_features=embedding_dimension // 2,
                    norm_type=normalization_layer_name,
                ),
            ),
            TransposeConvLayer2d(
                opts,
                in_channels=embedding_dimension // 2,
                out_channels=embedding_dimension // 4,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
                groups=1,
                use_norm=False,
                use_act=False,
                bias=True,
            ),
        )

        self.model_conf_dict["layer2"]["out"] = embedding_dimension // 4

        layer_l3 = TransposeConvLayer2d(
            opts,
            in_channels=embedding_dimension,
            out_channels=embedding_dimension // 2,
            kernel_size=2,
            stride=2,
            padding=0,
            output_padding=0,
            groups=1,
            use_norm=False,
            use_act=False,
            bias=True,
        )
        self.model_conf_dict["layer3"]["out"] = embedding_dimension // 2

        layer_l4 = Identity()
        layer_l5 = MaxPool2d(kernel_size=2, stride=2, padding=0)

        simple_fpn_layers = nn.ModuleDict(
            {
                "out_l2": layer_l2,
                "out_l3": layer_l3,
                "out_l4": layer_l4,
                "out_l5": layer_l5,
            }
        )

        return simple_fpn_layers

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls == VisionTransformer:
            group = parser.add_argument_group(cls.__name__)
            group.add_argument(
                "--model.classification.vit.mode",
                type=str,
                default="base",
                choices=["tiny", "small", "base", "large", "huge"],
                help="ViT mode. Default is base.",
            )
            group.add_argument(
                "--model.classification.vit.dropout",
                type=float,
                default=0.0,
                help="Dropout in Transformer layers. Defaults to 0.0.",
            )

            group.add_argument(
                "--model.classification.vit.stochastic-dropout",
                type=float,
                default=0.0,
                help="Stochastic Dropout in Transformer layers. Defaults to 0.0.",
            )

            group.add_argument(
                "--model.classification.vit.norm-layer",
                type=str,
                default="layer_norm",
                help="Normalization layer to be used in Transformer layer. Defaults to LayerNorm.",
            )

            group.add_argument(
                "--model.classification.vit.sinusoidal-pos-emb",
                action="store_true",
                default=False,
                help="Use sinusoidal instead of learnable positional embedding. Defaults to False.",
            )
            group.add_argument(
                "--model.classification.vit.no-cls-token",
                action="store_true",
                default=False,
                help="Do not use classification token. Defaults to False.",
            )

            group.add_argument(
                "--model.classification.vit.use-simple-fpn",
                action="store_true",
                default=False,
                help="Add simple FPN for down-stream tasks (e.g., detection). Defaults to False.",
            )
            group.add_argument(
                "--model.classification.vit.use-flash-attention",
                action="store_true",
                default=False,
                help="Use Transformer layers with flash attention for efficiently computing scaled dot-product attention. Defauls to False.",
            )

        return parser

    def extract_patch_embeddings(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Extract patch embeddings from input image tensor.

        Args:
            x: Input image tensor of size [batch, image channels, height, width]

        Returns:
            A tensor containing patch embeddings. The size of the tensor is [batch, number of patches, embedding dim].
        """
        # input is of shape [batch, image channels, height, width]. image channels is mostly 3 (for RGB images)
        batch_size = x.shape[0]

        # [batch, image channels, height, width] --> [batch, embedding dim, number of patches along height, number of patches along width]
        patch_emb = self.patch_emb(x)
        num_patches_height, num_patches_width = patch_emb.shape[-2:]

        # [batch, embedding dim, number of patches along height, number of patches along width] --> [batch, embedding dim, number of patches]
        patch_emb = patch_emb.flatten(2)
        # [batch, embedding dim, number of patches] --> [batch, number of patches, embedding dim]
        patch_emb = patch_emb.transpose(1, 2).contiguous()

        num_patches = patch_emb.shape[1]
        # we resize the positional encodings dynamically.
        pos_emb = self.pos_embed(num_patches).to(patch_emb.dtype)

        # add positional encodings
        patch_emb = pos_emb + patch_emb

        # add classification token
        if self.cls_token is not None:
            # [1, 1, embedding dim] --> [batch, 1, embedding dim]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            # Concat([batch, 1, embedding dim], [batch, number of patches, embedding dim]) --> [batch, number of patches + 1, embedding dim]
            patch_emb = torch.cat((cls_tokens, patch_emb), dim=1)

        # dropout
        patch_emb = self.emb_dropout(patch_emb)
        return patch_emb, (num_patches_height, num_patches_width)

    def _features_from_transformer(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """Helper function to extract patch embeddings and learn inter-patch representations using transformers.

        Args:
            x: Input image tensor of size [batch, image channels, Height, Width]

        Returns:
            A tensor containing contextualized patch embeddings.The size of the tensor is [batch, number of patches, embedding dimension]. It also
            returns a tuple containing the number of patches along height and width dimensions.
        """
        x, (n_h, n_w) = self.extract_patch_embeddings(x)
        x = self.transformer(x)
        x = self.post_transformer_norm(x)

        return x, (n_h, n_w)

    def extract_features(
        self, x: Tensor, return_image_embeddings: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Helper function for extraction features.

        Args:
            x: Input image tensor of size [batch, image channels, height, width].
            return_image_embeddings: When enabled, image embeddings are also returned.

        Returns:
            If 'return_image_embeddings=True', then both CLS_TOKEN and image embeddings are returned. Otherwise,
            CLS_TOKEN embedding and None are returned.

            The shape of CLS_TOKEN embedding is [batch, embedding dim] while the shape of image embeddings is
            [batch, embedding dim, num. patches height, num. patches width].
        """

        # [Batch, image channels, height, Width] --> [batch, CLS_TOKEN + number of patches, embedding dim]
        x, (n_h, n_w) = self._features_from_transformer(x)

        if self.cls_token is not None:
            # [batch, CLS_TOKEN + num. patches, embedding dim] --> [batch, embedding dim], [batch, number of patches, embedding dim]
            cls_embedding, image_embedding = torch.split(
                x, split_size_or_sections=[1, x.shape[1] - 1], dim=1
            )
            cls_embedding = cls_embedding.squeeze(1)
        else:
            # [batch, number of patches, embedding dim] -> [batch, embedding dim]
            cls_embedding = torch.mean(x, dim=1)
            # [batch, number of patches, embedding dim]
            image_embedding = x

        if return_image_embeddings:
            # reshape image embedding to 4-D tensor
            # [batch, number of patches, embedding dim] --> [batch, embedding dim, number of patches]
            image_embedding = image_embedding.transpose(1, 2).contiguous()
            # [batch, embedding dim, number of patches] --> [batch, embedding dim, number of patches along height, number of patches along width]
            image_embedding = image_embedding.reshape(
                image_embedding.shape[0], -1, n_h, n_w
            )

            return cls_embedding, image_embedding
        else:
            return cls_embedding, None

    def forward_classifier(
        self, x: Tensor, return_image_embeddings: bool = False
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward function for classification tasks.

        Args:
            x: Input image tensor of size [batch, image channels, height, width].
            return_image_embeddings: When enabled, image embeddings are also returned.

        Returns:
            The logits computed for CLS token are returned. If kwargs contain 'return_image_embeddings', then image embeddings
            are also returned.

            The shape of logits is [batch, number of classes] while the shape of image embeddings is
            [batch, embedding dim, num. patches height, num. patches width].
        """
        cls_embedding, image_embedding = self.extract_features(
            x, return_image_embeddings
        )
        # classify based on CLS token
        logits = self.classifier(cls_embedding)
        return logits, image_embedding

    def forward(
        self, x: Tensor, return_image_embeddings: bool = False
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """Forward function for ViT.

        Args:
            x: Input image tensor of shape [Batch, 3, Height, Width].
            return_image_embeddings: When enabled, image embeddings are also returned.

        Returns:
            The output of ViT model can be one of the following:
            1. If range augmentation is enabled, then a dictionary is returned with following keys
                'augmented_tensor': Contains the output after applying RangeAugment.
                'logits': Logit tensor
                'image_embeddings': Optionally tensor containing image embeddings
            2. If range augment is not enabled and return_image_embeddings is enabled, then a
               dictionary is returned with 'logits' and 'image_embeddings' keys.
            3. A logit tensor is returned.
        """

        if return_image_embeddings or self.neural_augmentor is not None:
            out_dict = {"augmented_tensor": None}
            if self.training and self.neural_augmentor is not None:
                # neural augmentor is applied during training  only
                x = self.neural_augmentor(x)
                out_dict.update({"augmented_tensor": x})
            logits, image_embedding = self.forward_classifier(
                x, return_image_embeddings
            )
            out_dict.update({"logits": logits})
            if image_embedding is not None:
                out_dict.update({"image_embeddings": image_embedding})
            return out_dict
        else:
            logits, _ = self.forward_classifier(x)
            return logits

    def extract_end_points_all(
        self,
        x: Tensor,
        use_l5: Optional[bool] = True,
        use_l5_exp: Optional[bool] = False,
    ) -> Dict[str, Tensor]:
        """Extract feature maps from different spatial levels in the model.

        This function is often used in down-stream applications (e.g., segmentation) where multi-scale features
        are required for prediction.

        Args:
            x: Input image tensor of shape [batch, image channels, height, width].
            use_l5: Extract features from spatial level 5.
            use_l5_exp: Extract features from the expansion layer in level 5.

        Returns:
            A mapping containing the output at each spatial level. The output keys are 'out_l1', 'out_l2',
            'out_l3', 'out_l4', 'out_l5', and 'out_l5_exp'.

        ...note:
            If 'use_l5_exp' is enabled, then features from expansion layer in level 5 are returned and 'out_l5' output
            is set to None. Otherwise, 'out_l5_exp' is set to None.
        """
        # this function is often used in down-stream applications (especially in segmentation and detection)
        if self.cls_token:
            logger.error("Please disable cls token for down-stream tasks")

        out_dict = {}
        if self.training and self.neural_augmentor is not None:
            x = self.neural_augmentor(x)
            out_dict["augmented_tensor"] = x

        cls_emb, x = self.extract_features(x, return_image_embeddings=True)
        out_dict["cls_embedding"] = cls_emb

        if self.simple_fpn is not None:
            # build simple FPN, as suggested in https://arxiv.org/abs/2203.16527
            for k, extra_layer in self.simple_fpn.items():
                out_dict[k] = extra_layer(x)
        else:
            # ViT does not have hierarchical structure by default.
            # Therefore, we set first four levels to None
            out_dict["out_l1"] = None
            out_dict["out_l2"] = None
            out_dict["out_l3"] = None
            out_dict["out_l4"] = None
            if use_l5_exp:
                out_dict["out_l5"] = None
                out_dict["out_l5_exp"] = x
            else:
                out_dict["out_l5"] = x
                out_dict["out_l5_exp"] = None
        return out_dict

    def get_activation_checkpoint_submodule_class(self) -> Callable:
        """Returns the activation checkpoint module class.

        For ViT, the activation checkpoint module class is TransformerEncoder or FlashTransformerEncoder.
        """
        return FlashTransformerEncoder if self.use_flash_attn else TransformerEncoder

    def get_fsdp_wrap_policy(
        self,
    ) -> Optional[Callable[[torch.nn.Module, bool, int], bool]]:
        """Returns the FSDP wrapping policy.

        For ViT, we use the Transformer's wrapping policy.
        """
        vit_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                FlashTransformerEncoder if self.use_flash_attn else TransformerEncoder
            },
        )
        return vit_auto_wrap_policy
