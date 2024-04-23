#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import json
import platform
import shutil
from pathlib import Path
from typing import Any, Dict, List, Union

try:
    import mlx.core as mx
    from huggingface_hub import snapshot_download
except ModuleNotFoundError:
    pass

import numpy as np
import torch

from corenet.constants import TMP_RES_FOLDER
from corenet.modeling import get_model
from corenet.modeling.models.classification.config.vit import get_configuration
from corenet.options.opts import get_training_arguments
from corenet.utils import logger
from corenet.utils.import_utils import ensure_library_is_available


def is_apple_silicon_macos() -> bool:
    return platform.machine() == "arm64" and platform.system() == "Darwin"


def mlx_naming_remap(corenet_k: str) -> str:
    """Translate CoreNet's CLIP weight mapping to MLX's naming scheme.

    Args: a single string represents the name of the parameter from CoreNet's
        state dict.
    Returns:
        a remapped string that will be honored by the MLX CLIP example.
    """
    trivial_mapping = {
        "text_encoder": "text_model",
        "image_encoder": "vision_model",
        "transformer": "encoder.layers",
        "positional_embedding.pos_embed.pos_embed": "embeddings.position_embedding.weight",
        "embedding_layer.weight": "embeddings.token_embedding.weight",
        "pre_norm_ffn.0": "layer_norm2",
        # NOTE: pre_norm_ffn.{2, 3} are act layer that do not have weights, and dropout layer,
        # it will not be part of the state dict.
        "pre_norm_ffn.1": "mlp.fc1",
        "pre_norm_ffn.4": "mlp.fc2",
        "pre_norm_mha.0": "layer_norm1",
        "pre_norm_mha.1.out_proj": "self_attn.out_proj",
        "pre_norm_mha.1": "self_attn",
        "cls_token": "embeddings.class_embedding",
        "post_encoder.layers_norm": "post_layernorm",
        "pos_embed.pos_embed.pos_embed": "embeddings.position_embedding.weight",
        "patch_emb.0.block.conv.weight": "embeddings.patch_embedding.layers.0.weight",
        "patch_emb.0.block.norm.bias": "embeddings.patch_embedding.layers.1.bias",
        "patch_emb.0.block.norm.running_mean": "embeddings.patch_embedding.layers.1.running_mean",
        "patch_emb.0.block.norm.running_var": "embeddings.patch_embedding.layers.1.running_var",
        "patch_emb.0.block.norm.weight": "embeddings.patch_embedding.layers.1.weight",
        "patch_emb.1.block.conv.weight": "embeddings.patch_embedding.layers.3.weight",
        "patch_emb.1.block.norm.bias": "embeddings.patch_embedding.layers.4.bias",
        "patch_emb.1.block.norm.running_mean": "embeddings.patch_embedding.layers.4.running_mean",
        "patch_emb.1.block.norm.running_var": "embeddings.patch_embedding.layers.4.running_var",
        "patch_emb.1.block.norm.weight": "embeddings.patch_embedding.layers.4.weight",
        "patch_emb.2.block.conv.weight": "embeddings.patch_embedding.layers.6.weight",
        "patch_emb.2.block.conv.bias": "embeddings.patch_embedding.layers.6.bias",
    }
    for k, v in trivial_mapping.items():
        if k in corenet_k:
            corenet_k = corenet_k.replace(k, v)

    # Depended on the trivial mapping to be done.
    non_trivial_mapping = {
        "text_model.projection_layer": "text_projection.weight",
        "vision_model.classifier.proj": "visual_projection.weight",
    }
    for k, v in non_trivial_mapping.items():
        if k in corenet_k:
            corenet_k = corenet_k.replace(k, v)

    return corenet_k


def make_shards(
    weights: Dict[str, Any], *, max_file_size_gb: int
) -> List[Dict[str, Any]]:
    """Split weights into separate shards given @max_file_size_gb limits for each shard.
    Each shard contains a mapping between the tensor name and its weights.
    """
    max_file_size_bytes = max_file_size_gb << 30
    shards = []
    shard, shard_size = {}, 0
    for k, v in weights.items():
        if shard_size + v.nbytes > max_file_size_bytes:
            shards.append(shard)
            shard, shard_size = {}, 0
        shard[k] = v
        shard_size += v.nbytes
    shards.append(shard)
    return shards


def save_weights(save_path: Union[str, Path], weights: Dict[str, Any]) -> None:
    """Save model weights into specified directory with MLX's safetensors format.

    Provide the given @weights, it will save it in MLX's safetensors format to @save_path.
    Shards will be created in 5GB chunk.
    """
    if isinstance(save_path, str):
        save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    shards = make_shards(weights, max_file_size_gb=5)
    shards_count = len(shards)
    shard_file_format = (
        "model-{:05d}-of-{:05d}.safetensors"
        if shards_count > 1
        else "model.safetensors"
    )

    total_byte_size = sum(v.nbytes for v in weights.values())
    # A json string that will encode the tensor info.
    index_data = {
        "metadata": {"total_size": total_byte_size},  # Provide the total size
        # Provide the mapping between weight tensor's name and the shard it
        # was put to.
        "weight_map": {},
    }

    # Partition weights into shard
    for i, shard in enumerate(shards):
        shard_name = shard_file_format.format(i + 1, shards_count)
        shard_path = save_path / shard_name

        # Save the tensor
        mx.save_safetensors(str(shard_path), shard)

        for weight_name in shard.keys():
            index_data["weight_map"][weight_name] = shard_name

    # Sort
    index_data["weight_map"] = {
        k: index_data["weight_map"][k] for k in sorted(index_data["weight_map"])
    }

    # Save the index.
    with open(save_path / "model.safetensors.index.json", "w") as f:
        json.dump(
            index_data,
            f,
            indent=4,
        )


def get_hf_clip_model_path() -> Path:
    """Get the configuration files and vocab using OpenAI's clip model, since
    MLX digests the model configurations and vocab through those files.

    We will need to modify the preprocessing and config, so it matches the definition
    of CoreNet's CLIP base's/huge's implementation.
    """
    HF_REPO = "openai/clip-vit-base-patch32"
    model_path = Path(
        snapshot_download(
            repo_id=HF_REPO,
            allow_patterns=[
                "*.json",
                "*.txt",
            ],
        )
    )
    return model_path


def ascontiguousarray(mx_array: mx.array) -> mx.array:
    """Convert a mx array to a contiguous mx array"""
    # save_safetensors requires row_contiguous array (mlx==0.8.1), while mlx does
    # not provide the API to do that, which we could only convert to numpy and and
    # copy back to mlx to acheive this.
    # This is expensive, but currently there are no way around it.
    # TODO(Frank): revisit to see if newer version of mlx will have
    # `ascontiguousarray` exposed.
    np_arr = np.ascontiguousarray(np.array(mx_array))
    return mx.array(np_arr, dtype=mx_array.dtype)


def sanitize(state_dict: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Perform some cleaning and transformations so MLX could load the sanitized
    weights faithfully under `strict=True` mode.
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in [
            "text_model.embeddings.position_embedding.weight",
            "vision_model.embeddings.position_embedding.weight",
            "vision_model.embeddings.class_embedding",
        ]:
            new_state_dict[k] = ascontiguousarray(v.squeeze(0).squeeze(0))
        elif k in ["visual_projection.weight", "text_projection.weight"]:
            # NOTE: There are more weights require transposition, but they will be
            # handled within MLX's `sanitize` function, similar to how OpenAI's
            # code was ported:
            # https://github.com/ml-explore/mlx-examples/blob/main/clip/model.py#L412
            new_state_dict[k] = ascontiguousarray(v.T)
        elif "qkv_proj" in k:
            q_proj_name = k.replace("qkv_proj", "q_proj")
            k_proj_name = k.replace("qkv_proj", "k_proj")
            v_proj_name = k.replace("qkv_proj", "v_proj")

            dim_size = v.shape[0] // 3
            if v.ndim == 1:  # Bias
                new_state_dict[q_proj_name] = ascontiguousarray(v[0:dim_size])
                new_state_dict[k_proj_name] = ascontiguousarray(
                    v[dim_size : 2 * dim_size]
                )
                new_state_dict[v_proj_name] = ascontiguousarray(v[2 * dim_size :])
            else:  # Weights
                new_state_dict[q_proj_name] = ascontiguousarray(v[0:dim_size, :])
                new_state_dict[k_proj_name] = ascontiguousarray(
                    v[dim_size : 2 * dim_size, :]
                )
                new_state_dict[v_proj_name] = ascontiguousarray(v[2 * dim_size :, :])
        elif "num_batches_tracked" in k or "neural_augmentor" in k:
            # parameters we don't need for the conversion
            continue
        else:
            new_state_dict[k] = v
    return new_state_dict


def torch_to_mx(a: torch.Tensor, *, dtype: str) -> mx.array:
    """Convert torch tensor to MLX tensor"""
    # bfloat16 is not numpy convertible. Upcast to float32 to avoid precision loss
    a = a.to(torch.float32) if dtype == "bfloat16" else a.to(getattr(torch, dtype))
    return mx.array(a.numpy(), getattr(mx, dtype))


def main() -> None:
    if not is_apple_silicon_macos():
        raise ValueError(
            "Expected to install MLX dependencies while on non-Apple Silicon MacOS. "
            "MLX is only available on Apple Silicon MacOS"
        )
    ensure_library_is_available("mlx")
    ensure_library_is_available("huggingface_hub")

    opts = get_training_arguments()

    pretrained_ckpt_loc = getattr(opts, "model.multi_modal_image_text.pretrained")
    config_file = getattr(opts, "common.config_file")
    logger.info(f"Loading config: {config_file} with ckpt: {pretrained_ckpt_loc}")

    model = get_model(opts)

    results_folder = getattr(opts, "common.results_loc")
    results_path = Path(results_folder)
    if results_path.is_file():
        raise ValueError(
            f"Result location specified is a regular file: {results_folder}"
        )
    results_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Save converted model to path: {results_folder}")
    logger.info("Converting")

    state_dict = model.state_dict()
    mlx_weights = {
        mlx_naming_remap(k): torch_to_mx(v, dtype="float32")
        for k, v in state_dict.items()
    }
    mlx_weights = sanitize(mlx_weights)

    logger.info("Saving weights")
    save_weights(results_path, mlx_weights)

    logger.info("Saving configs")
    hf_path = get_hf_clip_model_path()
    # Direct copy
    for fn in ["merges.txt", "vocab.json"]:
        shutil.copyfile(
            str(hf_path / f"{fn}"),
            str(results_path / f"{fn}"),
        )

    # Save the configuration about the input preprocessing.
    with open(str(hf_path / "preprocessor_config.json"), "r") as f:
        preprocessor_config = json.load(f)
        # 1. Set `do_normalize` to `false`.
        preprocessor_config["do_normalize"] = False
        # 2. Add entry `"use_clip_corenet_variant": true`
        preprocessor_config["use_clip_corenet_variant"] = True
    with open(str(results_path / "preprocessor_config.json"), "w") as f:
        json.dump(preprocessor_config, f, indent=2)

    vit_config = get_configuration(opts)
    text_n_header_per_layer = getattr(opts, "model.text.transformer.n_heads_per_layer")
    text_model_dim = getattr(opts, "model.text.transformer.model_dim")

    # Convert our yaml file to HF style config for model architecture.
    with open(str(hf_path / "config.json"), "r") as f:
        config = json.load(f)
        # 1. `"model_type": "clip_corenet",`  (originally configured as `"clip"`)
        config["model_type"] = "clip_corenet"
        # 2. `"hidden_act": "gelu",` (originally configured as `"quick_gelu"`)
        config["text_config"]["hidden_act"] = "gelu"
        config["vision_config"]["hidden_act"] = "gelu"
        # 3. `"layer_norm_eps": 1e-06,` (originally configured as `1e-05`)
        config["text_config"]["layer_norm_eps"] = 1e-06
        config["vision_config"]["layer_norm_eps"] = 1e-06
        # 4. `"patch_size": 16,` (originally configured as `32`)
        config["vision_config"]["patch_size"] = 16
        # 5. `"num_attention_heads": 8/16` for `"text_config"` depending on base/huge model
        config["text_config"]["num_attention_heads"] = text_n_header_per_layer
        # 6. Configure the text's dimension since base/huge variant has different cfg
        config["text_config"]["hidden_size"] = text_model_dim
        config["text_config"]["intermediate_size"] = text_model_dim * 4
        # 7. Configure the vision's parameters since base/huge variant has different cfg
        config["vision_config"]["hidden_size"] = vit_config["embed_dim"]
        config["vision_config"]["num_attention_heads"] = vit_config["n_attn_heads"]
        config["vision_config"]["intermediate_size"] = vit_config["ffn_dim"]
        config["vision_config"]["num_hidden_layers"] = vit_config[
            "n_transformer_layers"
        ]
    with open(str(results_path / "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    logger.info("Done")


if __name__ == "__main__":
    main()
