# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import math
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from corenet.modeling.layers import (
    Dropout,
    Embedding,
    PositionalEmbedding,
    get_normalization_layer,
)
from corenet.modeling.modules import TransformerEncoder
from corenet.modeling.text_encoders import TEXT_ENCODER_REGISTRY, BaseTextEncoder
from corenet.utils import logger


@TEXT_ENCODER_REGISTRY.register(name="transformer")
class TextTransformer(BaseTextEncoder):
    """Transformer-based text encoder.

    Args:
        opts: Command-line arguments.
        projection_dim: Projection dimension.
    """

    def __init__(self, opts, projection_dim: int, *args, **kwargs) -> None:
        model_dim = getattr(opts, "model.text.transformer.model_dim")
        no_scale_embedding = getattr(opts, "model.text.transformer.no_scale_embedding")
        no_pos_embedding = getattr(opts, "model.text.transformer.no_pos_embedding")
        embed_dropout = getattr(opts, "model.text.transformer.embed_dropout")
        dropout = getattr(opts, "model.text.transformer.dropout")
        attn_dropout = getattr(opts, "model.text.transformer.attn_dropout")
        ffn_dropout = getattr(opts, "model.text.transformer.ffn_dropout")
        norm_layer = getattr(opts, "model.text.transformer.norm_layer")

        if norm_layer is None:
            logger.error(
                "Normalization layer can not be None in {}".format(
                    self.__class__.__name__
                )
            )

        super().__init__(opts=opts, projection_dim=projection_dim, *args, **kwargs)

        # token embedding layer
        self.embedding_layer = Embedding(
            opts=opts,
            embedding_dim=model_dim,
            padding_idx=self.padding_index,
            num_embeddings=self.vocab_size,
        )
        self.embed_scale = 1.0 if no_scale_embedding else model_dim**-0.5

        self.positional_embedding = (
            None
            if no_pos_embedding
            else PositionalEmbedding(
                opts=opts,
                num_embeddings=self.context_length,
                embedding_dim=model_dim,
                padding_idx=self.padding_index,
                is_learnable=not getattr(
                    opts, "model.text.transformer.sinusoidal_pos_emb"
                ),
            )
        )

        self.embedding_dropout = Dropout(p=embed_dropout)

        n_transformer_layers = getattr(
            opts, "model.text.transformer.n_transformer_layers"
        )
        # FFN multipliers for transformer layer
        ffn_multipliers = getattr(
            opts, "model.text.transformer.ffn_multiplier_per_layer"
        )
        if isinstance(ffn_multipliers, (float, int)):
            ffn_multipliers = [ffn_multipliers] * n_transformer_layers

        if not isinstance(ffn_multipliers, Sequence):
            logger.error(
                "{} expects FFN multipliers as a list, whose length is the same as number of "
                "transformer layers. Got: {}".format(
                    self.__class__.__name__, type(ffn_multipliers)
                )
            )
        elif (
            isinstance(ffn_multipliers, Sequence)
            and len(ffn_multipliers) != n_transformer_layers
        ):
            logger.error(
                "We need FFN multiplier for each transformer layer. Got {} ffn multipliers while number of "
                "transformer layers = {}".format(
                    len(ffn_multipliers), n_transformer_layers
                )
            )
        ffn_dims = [
            int(math.ceil(model_dim * ffn_mult / 16.0) * 16.0)
            for ffn_mult in ffn_multipliers
        ]

        # Heads for transformer layers
        mha_heads = getattr(opts, "model.text.transformer.n_heads_per_layer")
        if isinstance(mha_heads, int):
            mha_heads = [mha_heads] * n_transformer_layers

        if not isinstance(mha_heads, Sequence):
            logger.error(
                "{} expects MHA heads as a list, whose length is the same as number of "
                "transformer layers. Got: {}".format(
                    self.__class__.__name__, type(mha_heads)
                )
            )
        elif isinstance(mha_heads, Sequence) and len(mha_heads) != n_transformer_layers:
            logger.error(
                "{} needs MHA heads for each transformer layer. Got {} mha heads while number of "
                "transformer layers = {}".format(
                    self.__class__.__name__, len(mha_heads), n_transformer_layers
                )
            )

        self.transformer = nn.ModuleList(
            [
                TransformerEncoder(
                    opts=opts,
                    embed_dim=model_dim,
                    num_heads=mha_heads[layer_idx],
                    ffn_latent_dim=ffn_dims[layer_idx],
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                    dropout=dropout,
                    transformer_norm_layer=norm_layer,
                )
                for layer_idx in range(n_transformer_layers)
            ]
        )
        self.final_layer_norm = get_normalization_layer(
            opts, num_features=model_dim, norm_type=norm_layer
        )

        self.projection_layer = nn.Parameter(
            torch.empty(model_dim, self.projection_dim)
        )
        self.model_dim = model_dim
        self.reset_parameters_clip_style()

        self.classes_per_split_zero_shot = max(
            1, int(getattr(opts, "model.text.transformer.classes_per_split_zero_shot"))
        )
        self.cached_attn_mask = None

    def build_causal_attention_mask(
        self,
        context_length: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[Tensor]:
        """Builds the causal attention mask.

        Args:
            context_length: Context length.
            batch_size: Batch size.
            device: Device on which mask should be created.
            dtype: Data type of the mask.

        Returns:
            An output tensor of the [batch size, context length, context length] if causal masking is enabled
            using 'model.text.transformer.causal_masking'. Otherwise, None is returned.
        """
        if getattr(self.opts, "model.text.transformer.causal_masking"):
            if self.cached_attn_mask is None:
                assert context_length <= self.context_length
                mask = torch.empty(
                    self.context_length, self.context_length, device=device, dtype=dtype
                )
                mask.fill_(float("-inf"))
                mask.triu_(1)
                self.cached_attn_mask = mask
            mask = self.cached_attn_mask[:context_length, :context_length]
            # add a dummy batch dimension and repeat it.
            mask = mask.unsqueeze(0)
            mask = mask.expand(batch_size, -1, -1)
            return mask.to(device=device, dtype=dtype)
        return None

    def reset_parameters_clip_style(self) -> None:
        """This function resets the weights of Transformer model as done in the CLIP paper"""

        # reset the weights of the embedding and positional embedding layers
        nn.init.normal_(self.embedding_layer.weight, mean=0.0, std=0.02)

        # compute standard deviation for different linear layers in transformer model
        attn_std = self.model_dim**-0.5
        proj_std = attn_std * ((2 * len(self.transformer)) ** -0.5)
        fc_std = (2 * self.model_dim) ** -0.5

        for block in self.transformer:
            # multi-head attention QKV projection layer
            nn.init.normal_(
                block.pre_norm_mha[1].qkv_proj.weight, mean=0.0, std=attn_std
            )
            # multi-head attention output projection layer
            nn.init.normal_(
                block.pre_norm_mha[1].out_proj.weight, mean=0.0, std=proj_std
            )
            # FFN expansion layer
            nn.init.normal_(block.pre_norm_ffn[1].weight, mean=0.0, std=fc_std)
            # FFN reduction layer
            nn.init.normal_(block.pre_norm_ffn[4].weight, mean=0.0, std=proj_std)

        nn.init.normal_(self.projection_layer, mean=0.0, std=attn_std)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != TextTransformer:
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.text.transformer.model-dim",
            type=int,
            default=512,
            help="Model dimension of the transformer model",
        )

        group.add_argument(
            "--model.text.transformer.no-scale-embedding",
            action="store_true",
            help="Do not scale the output of embedding layer in {}".format(
                cls.__name__
            ),
        )

        group.add_argument(
            "--model.text.transformer.no-pos-embedding",
            action="store_true",
            help="Do not add positional embeddings to the output of embedding layer in {}".format(
                cls.__name__
            ),
        )

        group.add_argument(
            "--model.text.transformer.embed-dropout",
            type=float,
            default=0.0,
            help="Dropout in embedding layer",
        )

        # transformer layer parameters
        default_layers = 6
        group.add_argument(
            "--model.text.transformer.n-transformer-layers",
            type=int,
            default=default_layers,
            help="Number of transformer layers in {}".format(cls.__name__),
        )
        group.add_argument(
            "--model.text.transformer.n-heads-per-layer",
            type=int,
            default=[8] * default_layers,
            nargs="+",
            help="Number of transformer heads per transformer layer",
        )

        group.add_argument(
            "--model.text.transformer.ffn-multiplier-per-layer",
            type=float,
            default=[4.0] * default_layers,
            nargs="+",
            help="FFN multiplier for each transformer layer",
        )
        group.add_argument(
            "--model.text.transformer.attn-dropout",
            type=float,
            default=0.0,
            help="Dropout in multi-head attention",
        )
        group.add_argument(
            "--model.text.transformer.ffn-dropout",
            type=float,
            default=0.0,
            help="Dropout between linear layers in FFN",
        )
        group.add_argument(
            "--model.text.transformer.dropout",
            type=float,
            default=0.0,
            help="Dropout in transformer",
        )

        group.add_argument(
            "--model.text.transformer.norm-layer",
            type=str,
            default="layer_norm",
            help="Normalization layer",
        )

        group.add_argument(
            "--model.text.transformer.sinusoidal-pos-emb",
            action="store_true",
            help="Use sinusoidal positional embedding",
        )

        group.add_argument(
            "--model.text.transformer.causal-masking",
            action="store_true",
            help="Use causal masking",
        )

        group.add_argument(
            "--model.text.transformer.classes-per-split-zero-shot",
            type=int,
            default=20,
            help="Divide zero-shot classes into these many chunks, for faster processing",
        )

        return parser

    def forward_embedding(self, text_tokens: Tensor) -> Tensor:
        """Converts the token indexes into vectors.

        Args:
            text_tokens: A tensor containing token indices. The shape  of the
                tensor is [batch, sequence length].

        Returns:
            An output tensor whose shape is [batch, sequence length, hidden dimension].
        """
        token_emb = self.embedding_layer(text_tokens)
        seq_len = token_emb.shape[1]
        if self.positional_embedding is not None:
            token_emb = token_emb + self.positional_embedding(seq_len).to(
                token_emb.dtype
            )
        token_emb = self.embedding_dropout(token_emb)
        return token_emb

    def encode_text(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_all_tokens: bool = False,
    ) -> Tensor:
        """
        Returns token embeddings.

        Args:
            text_tokens: A tensor containing token indicies. The shape of tensor is
                [batch size, sequence length].
            key_padding_mask: A boolean tensor indicating padding token indices.
            return_all_tokens: A boolean flag to return all tokens. Defaults to False
                to return end-of-text embedding.

        Returns:
            A tensor of shape [batch size, sequence length, hidden dimension] if 'return_all_tokens'
            is True. Otherwise, a tensor containing end-of-text token embedding is returned. The shape
            is [batch size, sequence length].
        """
        token_emb = self.forward_embedding(text_tokens)

        attn_mask = self.build_causal_attention_mask(
            context_length=text_tokens.shape[1],
            batch_size=text_tokens.shape[0],
            device=token_emb.device,
            dtype=token_emb.dtype,
        )

        for layer in self.transformer:
            token_emb = layer(
                token_emb,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

        token_emb = self.final_layer_norm(token_emb)

        if return_all_tokens:
            return token_emb
        else:
            # return the token embedding corresponding to end-of-text token.
            token_emb = token_emb[
                torch.arange(text_tokens.shape[0]), text_tokens.argmax(dim=-1)
            ]
            token_emb = token_emb @ self.projection_layer
            token_emb = F.normalize(token_emb, dim=-1)
            return token_emb

    def forward_zero_shot(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward function for computing text features for zero-shot image classification.

        Args:
            text_tokens: A tensor containing token indicies. The shape of tensor is
                [batch size, number of classes, number of captions, sequence length].
            key_padding_mask: A boolean tensor indicating padding token indices.

        Returns:
            A tensor of shape [number of classes, sequence length].
        """
        if self.training:
            raise NotImplementedError(
                "Zero-shot evaluation is only supported with eval mode"
            )

        if text_tokens.ndim != 4:
            logger.error(
                f"For zero-shot evaluation, expected a 4D tensor whose shape is [batch size, number of classes, number of captions, sequence length]. Got: {text_tokens.shape}."
            )

        batch_size, num_classes, num_captions, context_len = text_tokens.shape

        if batch_size > 1:
            text_tokens = text_tokens[0:1]
            batch_size = 1
            logger.warning(
                "For zero-shot evaluation, text templates are the same across all images in the batch."
                "Got: {}. Please consider adjusting collate function.".format(
                    batch_size
                )
            )

        text_features = []

        # The input 4D tensor could be very large and lead to out of memory issues. As an example,
        # the ImageNet dataset has 1000 classes and 80 captions per class. Processing such a large tensor
        # is very expensive. For efficiency, we split the tensor along the class dimension
        # and then compute features.
        for start_idx in range(0, num_classes, self.classes_per_split_zero_shot):
            end_idx = min(start_idx + self.classes_per_split_zero_shot, num_classes)

            text_tokens_split = text_tokens[0, start_idx:end_idx, ...]
            num_classes_split = text_tokens_split.shape[0]
            text_tokens_split = text_tokens_split.reshape(
                num_classes_split * num_captions, context_len
            )

            key_padding_mask_split = None
            if key_padding_mask is not None:
                key_padding_mask_split = key_padding_mask[0, start_idx:end_idx, ...]
                key_padding_mask_split = key_padding_mask_split.reshape(
                    num_classes_split * num_captions, context_len
                )

            # [num_classes_per_split * num_captions, sequence_length] --> [num_classes_per_split * num_captions, hidden_dim]
            class_embedding_split = self.encode_text(
                text_tokens=text_tokens_split, key_padding_mask=key_padding_mask_split
            )

            # [num_classes_per_split * num_captions, hidden_dim] --> [num_classes_per_split, num_captions, hidden_dim]
            class_embedding_split = class_embedding_split.reshape(
                num_classes_split, num_captions, class_embedding_split.shape[-1]
            )

            # Compute mean of all captions for a given class.
            # [num_classes_per_split, num_captions, hidden_dim] --> [num_classes_per_split, hidden_dim]
            mean_class_embedding_split = class_embedding_split.mean(dim=1)

            # Normalize the embeddings
            mean_class_embedding_split = F.normalize(mean_class_embedding_split, dim=-1)

            text_features.append(mean_class_embedding_split)

        # [num_classes_per_split, hidden_dim] * num_splits --> [num_classes, hidden_dim]
        text_features = torch.cat(text_features, dim=0)
        # [num_classes, hidden_dim] --> [hidden_dim, num_classes]
        text_features = text_features.transpose(0, 1)
        return text_features.contiguous()

    def forward(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_all_tokens: bool = False,
    ) -> Tensor:
        """Forward function for text encoder.

        Args:
            text_tokens: A tensor containing token indicies. The shape of tensor could be:
                1. [batch size, sequence length] -> This input size typically corresponds to
                    pre-training tasks (e.g., Image-Text pretraining in CLIP).
                2. [batch size, number of classes, number of captions, sequence length] -> This input size
                    typically corresponds to zero-shot image classification tasks.
                3. [batch size, number of captions, sequence length] -> This input size typically corresponds
                    to captioning tasks.
            key_padding_mask: A boolean tensor indicating padding token indices.
            return_all_tokens: A boolean flag to return all tokens.

        Returns:
            An output tensor. The shape of the output tensor is one of the following:
            1. When input tensor is 4D, then the shape of the output is [hidden_dim, number of classes].
            2. When input tensor is 2D, then the shape of the output is [batch size, hidden dim]. If 'return_all_tokens' is
                enabled, then shape of the output is [batch size, sequence length, hidden dim].
            3. When input tensor is 3D, then the shape of the output is [batch size, number of captions, hidden dim].
                If 'return_all_tokens' is enabled, then shape of the output is [batch size, number of captions, sequence length, hidden dim].
        """

        if text_tokens.dim() == 4:
            # Example use case is zero-shot image-classification evaluation
            return self.forward_zero_shot(
                text_tokens=text_tokens,
                key_padding_mask=key_padding_mask,
            )
        elif text_tokens.dim() == 2:
            # Example use case is image-text pre-training where each image
            # has single caption.
            text_tokens = self.encode_text(
                text_tokens=text_tokens,
                key_padding_mask=key_padding_mask,
                return_all_tokens=return_all_tokens,
            )
            return text_tokens
        elif text_tokens.dim() == 3:
            # Example use case is image-text pre-training where each image
            # has multiple captions.
            batch_size, num_captions, _ = text_tokens.shape
            text_tokens = text_tokens.reshape(batch_size * num_captions, -1)
            if key_padding_mask:
                key_padding_mask = key_padding_mask.reshape(
                    batch_size * num_captions, -1
                )
            text_tokens = self.encode_text(
                text_tokens=text_tokens,
                key_padding_mask=key_padding_mask,
                return_all_tokens=return_all_tokens,
            )
            text_tokens = text_tokens.reshape(batch_size, num_captions, -1)
            return text_tokens
        else:
            raise NotImplementedError(
                f"Only 2-, 3-, and 4-D tensors are supported. Got: {text_tokens.dim()}-D tensor."
            )

    def get_activation_checkpoint_submodule_class(self) -> Callable:
        """Returns TextTransformer's submodule, TransformerEncoder, class needs to be checkpointed."""
        return TransformerEncoder
