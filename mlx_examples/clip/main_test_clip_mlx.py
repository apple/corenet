# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

"""
Test that MLX output match with CoreNet's output.

The test requires a conversion using `main_clip_to_mlx.py` script to
dump the converted model to `results/mlx_model/` directory

Example invokation:

python main_test_clip_mlx.py \
    --common.config-file "../../projects/range_augment/clip/clip_vit_base.yaml" \
    --model.multi-modal-image-text.pretrained https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/clip_vit_base_16.pt
"""

import argparse
from os import path
from typing import Dict, List, Tuple

try:
    import mlx.core as mx
except ModuleNotFoundError:
    pass
import numpy as np
import torch

# Clip specific
from model import CLIPModel, CLIPModelOutput
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

# CoreNet specific
from corenet.data.text_tokenizer import build_tokenizer
from corenet.modeling.models import get_model
from corenet.modeling.models.base_model import BaseAnyNNModel
from corenet.options.opts import get_training_arguments
from corenet.utils import logger
from corenet.utils.import_utils import ensure_library_is_available


def load_mlx_model(model_dir: str) -> CLIPModel:
    """Load the MLX model from @model_dir"""
    if not path.exists(model_dir):
        raise ValueError(f"Model does not exist at: {model_dir}")
    model = CLIPModel.from_pretrained(model_dir)
    return model


def get_mlx_output(
    tokenized_input_ids: mx.array, pixel_values: mx.array
) -> CLIPModelOutput:
    """Load the MLX model and get the forward output given the inputs"""
    model = load_mlx_model("results/mlx_model")
    model.eval()

    inputs = {
        "input_ids": tokenized_input_ids,
        "pixel_values": pixel_values,
    }
    return model(**inputs)


def transform_image(img: Image.Image, size: int = 224) -> torch.Tensor:
    """Convert the PIL image to Tensor with pre-processing."""

    input_image_transform = Compose(
        [Resize(size=size), CenterCrop(size=size), ToTensor()]
    )
    input_img_tensor = input_image_transform(img)
    return input_img_tensor


def text_transform(tokeized_text_tensor: torch.Tensor) -> torch.Tensor:
    """Transform the text.

    No need to transform the text given the example.
    """
    return tokeized_text_tensor


def get_tokenized_input(
    opts: argparse.Namespace, class_names: List[str]
) -> torch.Tensor:
    """Build the tokenizer and get the tokenized input."""
    text_tokenizer = build_tokenizer(opts)

    input_text_templates = [
        "a photo of a {}".format(class_name) for class_name in class_names
    ]
    input_tokenized_templates = [
        text_transform(text_tokenizer(inp_template))
        for inp_template in input_text_templates
    ]
    input_tokenized_templates = torch.stack(input_tokenized_templates, dim=0)
    return input_tokenized_templates


def get_preprocessed_img(class_names: List[str]) -> torch.Tensor:
    """Build the image process pipeline and get preprocessed image."""
    input_img_batch = []
    for class_name in class_names:
        input_img = Image.open(f"assets/{class_name}.jpeg").convert("RGB")
        input_img_tensor = transform_image(input_img)
        input_img_batch.append(input_img_tensor)
    # stack input images to form a batch
    input_img_batch = torch.stack(input_img_batch, dim=0)
    return input_img_batch


def get_input_images_and_tokenized_text(
    opts: argparse.Namespace, class_names: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get the example pre-processed/tokenized text/images"""
    torch_images = get_preprocessed_img(class_names)
    torch_tokens = get_tokenized_input(opts, class_names)
    return torch_tokens, torch_images


def get_torch_output(
    opts: argparse.Namespace,
    torch_tokens: torch.Tensor,
    torch_images: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Perform the forward function and get the result given tokens and images"""
    mps_device = torch.device("mps")

    logger.disable_printing()
    clip_model = get_model(opts)
    clip_model.to(torch.float32).to(mps_device)
    clip_model.eval()
    logger.enable_printing()

    return clip_model.forward(
        input={
            "text": torch_tokens.to(mps_device),
            "image": torch_images.to(mps_device),
        }
    )


def torch_to_mx(a: torch.Tensor, *, dtype: str) -> mx.array:
    """Convert torch tensor to MLX tensor"""
    # bfloat16 is not numpy convertible. Upcast to float32 to avoid precision loss
    a = a.to(torch.float32) if dtype == "bfloat16" else a.to(getattr(torch, dtype))
    return mx.array(a.numpy(), getattr(mx, dtype))


def main_test() -> None:
    ensure_library_is_available("mlx")

    class_names = ["cat", "dog"]
    opts = get_training_arguments()

    torch_tokens, torch_images = get_input_images_and_tokenized_text(opts, class_names)
    torch_output = get_torch_output(opts, torch_tokens, torch_images)
    mlx_output = get_mlx_output(
        tokenized_input_ids=torch_to_mx(torch_tokens, dtype="int32"),
        pixel_values=torch_to_mx(torch_images, dtype="float32").transpose((0, 2, 3, 1)),
    )

    assert np.all(
        np.argmax(mlx_output.text_embeds, axis=1)
        == torch.argmax(torch_output["text"], axis=1).cpu().numpy()
    )
    assert np.all(
        np.argmax(mlx_output.image_embeds, axis=1)
        == torch.argmax(torch_output["image"], axis=1).cpu().numpy()
    )
    logger.info("Results matched.")


if __name__ == "__main__":
    main_test()
