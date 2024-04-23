# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

# Taken from https://github.com/ml-explore/mlx-examples/blob/main/clip/image_processor.py
# with modifications about imports and default model loading location.

import os
from typing import Tuple

import mlx.core as mx
from image_processor import CLIPImageProcessor
from model import CLIPModel
from PIL import Image
from tokenizer import CLIPTokenizer


def load(model_dir: str) -> Tuple[CLIPModel, CLIPTokenizer, CLIPImageProcessor]:
    model = CLIPModel.from_pretrained(model_dir)
    tokenizer = CLIPTokenizer.from_pretrained(model_dir)
    img_processor = CLIPImageProcessor.from_pretrained(model_dir)
    return model, tokenizer, img_processor


def main() -> None:
    example_class_names = ["cat", "dog"]

    model, tokenizer, img_processor = load("results/mlx_model")
    assert os.getcwd().endswith(
        "mlx_examples/clip"
    ), "Please run this script from 'mlx_examples/clip' folder."
    inputs = {
        "input_ids": tokenizer(
            [f"a photo of a {class_name}" for class_name in example_class_names]
        ),
        "pixel_values": img_processor(
            [
                Image.open(f"../../assets/{class_name}.jpeg")
                for class_name in example_class_names
            ]
        ),
    }
    output = model(**inputs)

    # Get text and image embeddings:
    text_embeds = output.text_embeds
    image_embeds = output.image_embeds
    logits = image_embeds @ text_embeds.T
    logits = mx.softmax(logits, axis=-1)
    predicted_class_prob = mx.max(logits, axis=-1).tolist()
    predicted_class_id = mx.argmax(logits, axis=-1).tolist()

    for batch_id in range(len(predicted_class_prob)):
        print(
            f"Predicted class for sample {batch_id} is"
            f" {example_class_names[predicted_class_id[batch_id]]} "
            f"p=({predicted_class_prob[batch_id]:.3f})"
        )


if __name__ == "__main__":
    main()
