# MLX port of OpenELM

This directory contains an MLX port of OpenELM model trained with CoreNet. MLX
is an Apple deep learning framework similar in spirit to PyTorch, which is
optimized for Apple Silicon based hardware.

This code requires the MLX-specific dependencies from `../requirements.txt` to
be installed. We assume that the main requirements.txt is already installed.


## Downloading pre-converted checkpoints

The pre-converted checkpoints are available at the following URLs.

| Model | Weights | Config |
| ---- | ---- | ---- |
| 270M | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/270M/weights.safetensors) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/270M/config.json) |
| 270M - 4bit | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/270M-4bit/weights.safetensors) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/270M-4bit/config.json) |
| 450M | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/450M/weights.safetensors) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/450M/config.json) |
| 450M - 4bit | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/450M-4bit/weights.safetensors) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/450M-4bit/config.json) |
| 1.1B | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/1.1B/weights.safetensors) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/1.1B/config.json) |
| 1.1B - 4bit | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/1.1B-4bit/weights.safetensors) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/1.1B-4bit/config.json) |
| 3B | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/3B/weights.safetensors) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/3B/config.json) |
| 3B - 4bit | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/3B-4bit/weights.safetensors) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/mlx/3B-4bit/config.json) |

 Note that these checkpoints do not contain a tokenizer model file, which is
 required for inference with `inference.py`. Simply place Meta LLaMA2's
 `tokenizer.model` into the directories to load model using our provided
 `inference.py`, or if you prefer to use the models directly, use the
 corresponding tokenizer from Huggingface Transformers.

## Running the model

In order to run the model, the `inference.py` script is provided. It also
provides documentation for how to load and use the model if you are not
familiar with language modeling in MLX.

Here's a usage example:

```
PYTHONPATH=. mlx_examples/open_elm/inference.py \
    --model-dir <MLX model directory> \
    --prompt "Once upon a time in a land far away" \
    --max-tokens=1024
```

This should produce a completion for your prompt.

## Converting the weights

This port includes a conversion script, which can also do quantization. We have
tested this script with fp16/bf16 and 4-bit quantized models with group size 32
and 64. Because of the similarities between MLX and PyTorch the naming of all
variables in checkpoints is identical.

A note on the tokenizer model: OpenELM uses Meta LLaMA tokenizer, which you will
need to obtain from Meta.

To run a fp16 conversion, download the training YAML configuration with which
the model was trained, and the `*.pt` checkpoint that corresponds to that
configuration. Then, execute the following command from the root of this
repository:

```
PYTHONPATH=. mlx_examples/open_elm/convert.py \
    --input-checkpoint <PyTorch/CoreNet checkpoint> \
    --config-yaml <CoreNet training configuration YAML> \
    --tokenizer-path <path to tokenizer.model> \
    --dtype="float16" \
    --output-dir <output dir>
```

This will produce two files: an `*.npz` checkpoint, and `config.json` configuration
file necessary to load the checkpoint.

In order to convert to a 4-bit quantized checkpoint, simply add the required
flags like so:

```
PYTHONPATH=. python3 mlx_examples/open_elm/convert.py \
    --input-checkpoint <PyTorch checkpoint> \
    --config-yaml <CoreNet training configuration YAML> \
    --tokenizer-path <path to tokenizer.model> \
    --dtype="float16" \
    --quantize \
    --output-dir <output dir>
```

Both of these commands will produce self-contained model directories with
weights, configuration and tokenizer files inside.

Note that OpenELM 3B should use BFloat16 for both 16-bit and quantized
inference. It requires a greater activation range than the other model sizes.
