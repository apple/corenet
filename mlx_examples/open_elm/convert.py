#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import copy
import json
import pathlib
import shutil
import sys
from typing import Any, Dict, Tuple

import numpy as np
import safetensors
import torch
import yaml

from mlx_examples.open_elm import open_elm

try:
    import mlx
    from mlx import core as mx
    from mlx import nn
except ImportError:
    sys.exit("You must install Apple MLX to use this program.")


def torch_to_mlx(x: torch.Tensor) -> mx.array:
    """Converts a PyTorch tensor to an MLX tensor with the same dtype.

    Args:
        x: PyTorch tensor to convert

    Returns:
        An MLX version with the same dtype and contents.
    """
    x = x.detach()
    torch_dtype = str(x.dtype).split(".")[-1]
    mlx_dtype = getattr(mx, torch_dtype)
    # MLX mentions that converting to bfloat16 under NumPy could result in
    # precision loss, so we first up-cast to fp32.
    if torch_dtype == "bfloat16":
        x = x.to(torch.float32)
    return mx.array(x.cpu().numpy(), dtype=mlx_dtype)


def quantize_weights(
    weights: Dict[str, mx.array],
    model_config: Dict[str, Any],
    bits: int = 4,
    group_size: int = 64,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Quantizes model weights to a given bit depth and group size.

    Args:
        weights: weights dictionary for the model
        model_config: model configuration from the JSON config file
        bits: quantization depth
        group_size: quantization group size

    Returns:
        Quantized weights dictionary and the updated model config.
    """
    model_config = copy.deepcopy(model_config)
    model = open_elm.OpenELM(**model_config)
    weights = mlx.utils.tree_map(mx.array, weights)
    model.update(mlx.utils.tree_unflatten(list(weights.items())))

    nn.QuantizedLinear.quantize_module(model, group_size=group_size, bits=bits)

    quantized_weights = dict(mlx.utils.tree_flatten(model.parameters()))
    model_config["quantization"] = {
        "group_size": group_size,
        "bits": bits,
    }

    return quantized_weights, model_config


def convert_open_elm(
    torch_checkpoint_path: pathlib.Path,
    tokenizer_path: pathlib.Path,
    config_yaml_path: pathlib.Path,
    output_dir: pathlib.Path,
    quantize: bool = False,
    quantization_bits: int = 4,
    quantization_group_size: int = 64,
    dtype: mx.Dtype = mx.float16,
) -> None:
    """Converts Apple OpenELM LLM checkpoint and configuration from
    PyTorch/CoreNet format to MLX format.


    Args:
        torch_checkpoint_path: path to the input *.pt checkpoint file
        tokenizer_path: path to the tokenizer file to copy to the output
        config_yaml_path: path to the training config *.yaml file
        output_dir: output directory to write the checkpoint
        quantize: set to true to enable quantization (default 4 bit, group size 64)
        quantization_bits: number of bits to quantize to
        quantization_group_size: quantization group size

    Returns:
        None
    """
    assert torch_checkpoint_path.is_file(), torch_checkpoint_path
    assert config_yaml_path.is_file(), config_yaml_path

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract the relevant part of YAML config.
    with config_yaml_path.open("r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]["language_modeling"]["general_gpt"]
    # Padding index is used in CoreNet for training, and is not required for inference.
    model_config.pop("padding_index")

    # Load PyTorch checkpoint.
    ckpt = torch.load(torch_checkpoint_path, map_location="cpu")
    weights = {k: torch_to_mlx(v).astype(dtype) for k, v in ckpt.items()}

    if quantize:
        weights, model_config = quantize_weights(
            weights,
            model_config,
            bits=quantization_bits,
            group_size=quantization_group_size,
        )

    checkpoint_file = output_dir / "weights.safetensors"
    with checkpoint_file.open("wb") as f:
        mx.save_safetensors(f, weights)
    print(f"Wrote converted checkpoint to {checkpoint_file}.")

    config_file = output_dir / "config.json"
    with config_file.open("w") as f:
        json.dump(model_config, f, indent=2)
    print(f"Wrote config to {config_file}.")

    shutil.copy2(tokenizer_path, output_dir / "tokenizer.model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts OpenELM checkpoints from PyTorch to MLX.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-checkpoint",
        type=pathlib.Path,
        required=True,
        help="Input PyTorch / CoreNet checkpoint for Apple OpenELM model.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=pathlib.Path,
        required=True,
        help="SentencePiece tokenizer model file to copy into the output directory.",
    )
    parser.add_argument(
        "--config-yaml",
        type=pathlib.Path,
        required=True,
        help="Path to the YAML file containing the CoreNet training configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Output directory for MLX checkpoint.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        help="Data type to use for the converted model.",
    )
    parser.add_argument(
        "--quantize",
        help="Generate a quantized model.",
        action="store_true",
    )
    parser.add_argument(
        "--quantization-group-size",
        help="Group size for quantization.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--quantization-bits",
        help="Bits per weight for quantization.",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    dtype = args.dtype.lower()
    if dtype == "float16":
        dtype = mx.float16
    elif dtype == "bfloat16":
        dtype = mx.bfloat16
    elif dtype == "float32":
        dtype = mx.float32
    else:
        raise ValueError(f"Unsupported dtype {dtype}.")

    convert_open_elm(
        args.input_checkpoint,
        args.tokenizer_path,
        args.config_yaml,
        args.output_dir,
        quantize=args.quantize,
        quantization_bits=args.quantization_bits,
        quantization_group_size=args.quantization_group_size,
        dtype=dtype,
    )
