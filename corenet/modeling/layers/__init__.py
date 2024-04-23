#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import importlib
import inspect
import os

from corenet.modeling.layers.adaptive_pool import AdaptiveAvgPool2d
from corenet.modeling.layers.base_layer import BaseLayer
from corenet.modeling.layers.conv_layer import (
    ConvLayer1d,
    ConvLayer2d,
    ConvLayer3d,
    NormActLayer,
    SeparableConv1d,
    SeparableConv2d,
    SeparableConv3d,
    TransposeConvLayer2d,
)
from corenet.modeling.layers.dropout import Dropout, Dropout2d
from corenet.modeling.layers.embedding import Embedding
from corenet.modeling.layers.flash_multi_head_attention import (
    FlashMultiHeadSelfAttention,
)
from corenet.modeling.layers.flatten import Flatten
from corenet.modeling.layers.global_pool import GlobalPool
from corenet.modeling.layers.identity import Identity
from corenet.modeling.layers.linear_attention import LinearSelfAttention
from corenet.modeling.layers.linear_layer import GroupLinear, LinearLayer
from corenet.modeling.layers.multi_head_attention import MultiHeadAttention
from corenet.modeling.layers.normalization_layers import (
    get_normalization_layer,
    norm_layers_tuple,
)
from corenet.modeling.layers.pixel_shuffle import PixelShuffle
from corenet.modeling.layers.pooling import AvgPool2d, MaxPool2d
from corenet.modeling.layers.positional_embedding import PositionalEmbedding
from corenet.modeling.layers.rotary_embeddings import RotaryEmbedding
from corenet.modeling.layers.single_head_attention import SingleHeadAttention
from corenet.modeling.layers.softmax import Softmax
from corenet.modeling.layers.stochastic_depth import StochasticDepth
from corenet.modeling.layers.upsample import UpSample

__all__ = [
    "ConvLayer1d",
    "ConvLayer2d",
    "ConvLayer3d",
    "SeparableConv1d",
    "SeparableConv2d",
    "SeparableConv3d",
    "NormActLayer",
    "TransposeConvLayer2d",
    "LinearLayer",
    "GroupLinear",
    "GlobalPool",
    "Identity",
    "PixelShuffle",
    "UpSample",
    "MaxPool2d",
    "AvgPool2d",
    "Dropout",
    "Dropout2d",
    "Flatten",
    "MultiHeadAttention",
    "SingleHeadAttention",
    "Softmax",
    "LinearSelfAttention",
    "Embedding",
    "PositionalEmbedding",
    "norm_layers_tuple",
    "StochasticDepth",
    "get_normalization_layer",
    "RotaryEmbedding",
    "FlashMultiHeadSelfAttention",
]


# iterate through all classes and fetch layer specific arguments
def layer_specific_args(parser: argparse.ArgumentParser):
    layer_dir = os.path.dirname(__file__)
    parsed_layers = []
    for file in os.listdir(layer_dir):
        path = os.path.join(layer_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            layer_name = file[: file.find(".py")] if file.endswith(".py") else file
            module = importlib.import_module("corenet.modeling.layers." + layer_name)
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, BaseLayer) and name not in parsed_layers:
                    parser = cls.add_arguments(parser)
                    parsed_layers.append(name)
    return parser


def arguments_nn_layers(parser: argparse.ArgumentParser):
    # Retrieve layer specific arguments
    parser = layer_specific_args(parser)

    # activation and normalization arguments
    from corenet.modeling.layers.activation import arguments_activation_fn

    parser = arguments_activation_fn(parser)

    from corenet.modeling.layers.normalization import arguments_norm_layers

    parser = arguments_norm_layers(parser)

    return parser
