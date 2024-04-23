#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import math
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import init

from corenet.modeling.layers import (
    LinearLayer,
    embedding,
    get_normalization_layer,
    normalization,
    positional_embedding,
    token_merging,
)
from corenet.modeling.models import MODEL_REGISTRY, BaseAnyNNModel
from corenet.modeling.models.classification.config.byteformer import get_configuration
from corenet.modeling.modules import WindowedTransformerEncoder


def unfold_tokens(t: Tensor, kernel_size: int) -> Tensor:
    """
    Group tokens from tensor @t using torch.Tensor.unfold, using the given
    kernel size. This amounts to windowing @t using overlapping windows
    of size @kernel_size, with overlap of @kernel_size // 2.

    Args:
        t: A tensor of shape [batch_size, sequence_length, num_channels].
        kernel_size: The kernel size.

    Returns:
        A tensor of shape [batch_size * (sequence_length - kernel_size)
        // (kernel_size // 2) + 1, kernel_size, num_channels].
    """
    t = t.unfold(dimension=1, size=kernel_size, step=kernel_size // 2)
    B, L, C, _ = t.shape
    t = t.reshape(B * L, C, kernel_size)
    t = t.transpose(1, 2)
    return t


@MODEL_REGISTRY.register(name="byteformer", type="classification")
class ByteFormer(BaseAnyNNModel):
    """
    This class defines the `ByteFormer <https://arxiv.org/abs/2306.00238>`_ architecture.
    """

    def __init__(self, opts: argparse.Namespace, *args, **kwargs) -> None:
        super().__init__(opts, *args, **kwargs)

        byteformer_config = get_configuration(opts)
        embed_dim = byteformer_config["embed_dim"]
        ffn_dim = byteformer_config["ffn_dim"]
        n_transformer_layers = byteformer_config["n_transformer_layers"]
        num_heads = byteformer_config["n_attn_heads"]
        attn_dropout = byteformer_config["attn_dropout"]
        dropout = byteformer_config["dropout"]
        ffn_dropout = byteformer_config["ffn_dropout"]
        norm_layer = byteformer_config["norm_layer"]

        # This is usually 257 in the case of byte inputs (2**8 + 1 mask token).
        vocab_size = getattr(opts, "model.classification.byteformer.vocab_size")
        self.embeddings = embedding.Embedding(
            opts, num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=-1
        )
        # Reinitialize everything except the padding index.
        init.trunc_normal_(self.embeddings.weight[:-1], std=math.sqrt(1.0 / embed_dim))

        self.dummy_input_token_length = getattr(
            opts, "model.classification.byteformer.dummy_input_token_length"
        )

        # Add token reduction convolution.
        self.conv_kernel_size = getattr(
            opts, "model.classification.byteformer.conv_kernel_size"
        )
        if self.conv_kernel_size == 0:
            # We skip the convolution.
            self.token_reduction_net = None
        if self.conv_kernel_size is not None:
            self.token_reduction_net = nn.Conv1d(
                embed_dim,
                get_configuration(opts)["embed_dim"],
                kernel_size=self.conv_kernel_size,
                stride=self.conv_kernel_size // 2,
                bias=False,
            )

        # Add the positional embeddings.
        self.max_num_tokens = getattr(
            opts, "model.classification.byteformer.max_num_tokens"
        )
        self.sinusoidal_pos_embed = getattr(
            opts, "model.classification.byteformer.sinusoidal_pos_emb"
        )
        self.pos_embed = positional_embedding.PositionalEmbedding(
            opts=opts,
            num_embeddings=self.max_num_tokens,
            embedding_dim=embed_dim,
            sequence_first=False,
            padding_idx=None,
            is_learnable=not self.sinusoidal_pos_embed,
            interpolation_mode="bilinear",
        )

        pos_emb_drop_p = getattr(opts, "model.classification.byteformer.dropout")
        self.emb_dropout = nn.Dropout(p=pos_emb_drop_p)

        # Build the transformer backbone.
        window_sizes = getattr(opts, "model.classification.byteformer.window_sizes")
        window_shifts = getattr(opts, "model.classification.byteformer.window_shifts")
        downsample = getattr(opts, "model.classification.byteformer.downsample")

        if len(window_sizes) == 1:
            window_sizes = window_sizes * n_transformer_layers

        for x in [window_sizes, window_shifts, downsample]:
            if len(x) != n_transformer_layers:
                raise ValueError(
                    f"Invalid argument length {len(x)} != {n_transformer_layers}"
                )

        stochastic_dropout = getattr(
            opts, "model.classification.byteformer.stochastic_dropout"
        )
        per_layer_stochastic_drop_rate = [
            round(x, 3)
            for x in np.linspace(0, stochastic_dropout, n_transformer_layers)
        ]

        blocks = []
        self.downsamplers = nn.ModuleDict()
        for layer_idx in range(n_transformer_layers):
            blocks.append(
                WindowedTransformerEncoder(
                    opts=opts,
                    embed_dim=embed_dim,
                    ffn_latent_dim=ffn_dim,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    ffn_dropout=ffn_dropout,
                    transformer_norm_layer=norm_layer,
                    stochastic_dropout=per_layer_stochastic_drop_rate[layer_idx],
                    window_size=window_sizes[layer_idx],
                    window_shift=window_shifts[layer_idx],
                )
            )
            if downsample is not None and downsample[layer_idx]:
                self.downsamplers[self.get_downsampler_name(layer_idx)] = (
                    token_merging.TokenMerging(embed_dim)
                )
        self.transformer = nn.Sequential(*blocks)

        self.post_transformer_norm = get_normalization_layer(
            opts=opts, num_features=embed_dim, norm_type=norm_layer
        )

        num_classes = getattr(opts, "model.classification.n_classes")
        self.classifier = LinearLayer(embed_dim, num_classes)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        if cls != ByteFormer:
            # Don't re-register arguments in subclasses that don't override `add_arguments()`.
            return parser
        group = parser.add_argument_group(title=cls.__name__)
        group.add_argument(
            "--model.classification.byteformer.dropout",
            type=float,
            default=0.0,
            help="Dropout in Byteformer layers. Defaults to 0.0.",
        )
        group.add_argument(
            "--model.classification.byteformer.stochastic-dropout",
            type=float,
            default=0.0,
            help="Probability of applying stochastic dropout to "
            "TransformerEncoder submodules. Defaults to 0.0.",
        )
        group.add_argument(
            "--model.classification.byteformer.norm-layer",
            type=str,
            default="layer_norm",
            help="Normalization layer in Byteformer. Defaults to LayerNorm.",
            choices=list(normalization.NORM_LAYER_REGISTRY.keys()),
        )
        group.add_argument(
            "--model.classification.byteformer.sinusoidal-pos-emb",
            action="store_true",
            default=False,
            help="Use sinusoidal instead of learnable positional encoding. Defaults to False.",
        )
        group.add_argument(
            "--model.classification.byteformer.use-pytorch-mha",
            action="store_true",
            default=False,
            help="Use PyTorch's native multi-head attention. Defaults to False.",
        )
        group.add_argument(
            "--model.classification.byteformer.mode",
            type=str,
            default="tiny",
            help="Byteformer mode, which determines the model size. Defaults to tiny.",
            choices=("tiny", "small", "base", "huge"),
        )
        group.add_argument(
            "--model.classification.byteformer.vocab-size",
            type=int,
            help="The vocab size of the token embedding. Defaults to 257,"
            "corresponding to the number of unique bytes (256) plus 1 "
            "more for the mask token.",
            default=257,
        )
        group.add_argument(
            "--model.classification.byteformer.max-num-tokens",
            type=int,
            help="The maximum number of tokens that can be input to the network. Defaults to 10000.",
            default=10000,
        )
        group.add_argument(
            "--model.classification.byteformer.conv-kernel-size",
            type=int,
            default=16,
            help="The size of the kernel of the initial downsampling conv1d. Defaults to 16.",
        )
        group.add_argument(
            "--model.classification.byteformer.window-sizes",
            type=int,
            nargs="*",
            default=[128],
            help="A list of window sizes used in shifted window attention. If the "
            "list is length 1, the same window size is used for all windows. "
            "Defaults to 128 for all windows.",
        )
        group.add_argument(
            "--model.classification.byteformer.window-shifts",
            type=int,
            nargs="*",
            default=[0, 64] * 6,
            help="A list of shifts used in shifted window attention. Defaults to values that alternate between 0 and 64.",
        )
        default_downsampling = [True, True] + ([False, True] * 4) + [False, False]
        group.add_argument(
            "--model.classification.byteformer.downsample",
            type=bool,
            nargs="*",
            default=default_downsampling,
            help="A list of boolean values, where the i'th element specifies "
            "whether to downsample after the transformer block with index i. "
            f"Defaults to {default_downsampling}.",
        )
        group.add_argument(
            "--model.classification.byteformer.padding-index",
            default=-1,
            type=int,
            help="The index used for padding tokens. Defaults to -1.",
        )
        group.add_argument(
            "--model.classification.byteformer.dummy-input-token-length",
            default=48564,
            type=int,
            help="The token length to use for dummy inputs. Defaults to 48564, "
            "corresponding to the average length of 224x224 JPEG images from "
            "ImageNet.",
        )
        return parser

    def dummy_input_and_label(self, batch_size: int) -> Dict:
        """
        Get a dummy input and label that could be passed to the model.

        Args:
            batch_size: The batch size to use for the generated inputs.

        Returns:
            A dict with
                {
                    "samples": tensor of shape [batch_size, sequence_length],
                    "targets": tensor of shape [batch_size],
                }
        """
        n_labels = 10
        max_value = 257

        samples = torch.randint(
            0, max_value, [batch_size, self.dummy_input_token_length]
        )
        targets = torch.randint(low=0, high=n_labels, size=(batch_size,)).long()
        return {"samples": samples, "targets": targets}

    def apply_token_reduction_net(
        self, x: Tensor, x_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply the portion of the network used to reduce sequence lengths before
        the transformer backbone.

        Args:
            x: The input token embeddings of shape [batch_size, sequence_length,
                embed_dim].
            x_mask: The input mask of shape [batch_size, sequence_length].

        Returns:
            New versions of @x and @x_mask, downsampled along the sequence
            dimension by the token reduction net.
        """
        B, N, C = x.shape
        if self.token_reduction_net is None:
            return x, x_mask

        x = self.token_reduction_net(x.permute(0, 2, 1)).permute(0, 2, 1)
        if x_mask is not None:
            x_mask = unfold_tokens(
                x_mask.reshape(B, N, 1).float(), self.conv_kernel_size
            )
            # The mask is now [B * N, kernel_size, 1]. It contains values in {0, -inf}.
            x_mask = x_mask.max(dim=1).values.view(x.shape[0], x.shape[1])

            assert x.shape[:2] == x_mask.shape
        return x, x_mask

    def get_backbone_inputs(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Convert input bytes into embeddings to be passed to the network's
        transformer backbone.

        Args:
            x: The input bytes as an integer tensor of shape [batch_size,
                sequence_length]. Integer tensors are expected (rather than byte
                tensors) since -1 is usually used for padding.

        Returns:
            The embeddings of shape [batch_size, new_sequence_length] and a
            mask tensor of shape [batch_size, new_sequence_length]. The mask
            contains 0 at unmasked positions and float(-inf) at masked
            positions.
        """
        mask = torch.zeros_like(x, dtype=torch.float)
        mask[x == -1].fill_(float("-inf"))
        mask = mask.detach().requires_grad_(False)
        x[x == -1] = self.embeddings.padding_idx
        x = self.embeddings(x)

        x, mask = self.apply_token_reduction_net(x, mask)
        x = x + self.pos_embed(self.max_num_tokens)[:, : x.shape[1]]

        x = self.emb_dropout(x)
        return x, mask

    def backbone_forward(
        self, x: Tensor, key_padding_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Execute the forward pass of the network's transformer backbone.

        Args:
            x: The input embeddings as a [batch_size, sequence_length, embed_dim] tensor.
            key_padding_mask: The mask tensor of shape [batch_size, sequence_length].

        Returns:
            The outputs of the backbone as a tuple. The first element is the feature
            tensor, and the second element is the updated key_padding_mask.
        """
        B, S, _ = x.shape
        assert key_padding_mask.shape == (B, S)

        for layer_idx, elem in enumerate(self.transformer):
            x = elem(x, key_padding_mask=key_padding_mask)
            if self.get_downsampler(layer_idx) is not None:
                x, key_padding_mask = self.get_downsampler(layer_idx)(
                    x, key_padding_mask
                )
        x = self.post_transformer_norm(x)
        return x, key_padding_mask

    def get_downsampler_name(self, idx: int) -> str:
        """
        Get the name of the downsampling layer with index @idx.

        Args:
            idx: The index of the downsampling layer.

        Returns:
            A string representing the name of the donwsampling layer.
        """
        return f"downsample_{idx}"

    def get_downsampler(self, idx: int) -> Optional[nn.Module]:
        """
        Get the module that performs downsampling after transformer layer @idx.
        If no downsampling occurs after that layer, return None.

        Args:
            idx: The desired index.

        Returns:
            The downsampling layer, or None.
        """
        name = self.get_downsampler_name(idx)
        if name not in self.downsamplers:
            return None
        return self.downsamplers[name]

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """
        Perform a forward pass on input bytes. The tensor is
        stored as an integer tensor of shape [batch_size, sequence_length].
        Integer tensors are used because @x usually contains mask tokens.

        Args:
            x: The input tensor of shape [batch_size, sequence_length].

        Returns:
            The output logits.
        """
        x, key_padding_mask = self.get_backbone_inputs(x)
        x, attn_mask = self.backbone_forward(x, key_padding_mask)

        attn_mask = attn_mask.view(x.shape[0], x.shape[1], 1)
        x[(attn_mask == float("-inf")).expand(-1, -1, x.shape[-1])] = 0
        norms = (attn_mask == 0).sum(dim=1)
        x = torch.sum(x, dim=1) / norms
        x = self.classifier(x)
        return x

    @classmethod
    def build_model(cls, opts: argparse.Namespace, *args, **kwargs) -> BaseAnyNNModel:
        """
        Helper function to build a model.

        Args:
            opts: Command-line arguments.

        Returns:
            An instance of `corenet.modeling.models.BaseAnyNNModel`.
        """
        model = cls(opts, *args, **kwargs)

        if getattr(opts, "model.classification.freeze_batch_norm"):
            cls.freeze_norm_layers(opts=opts, model=model)
        return model
