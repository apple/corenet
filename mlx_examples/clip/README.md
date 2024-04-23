# MLX port of CLIP

This is an example to convert CoreNet's CLIP model implementation to
[MLX](https://github.com/ml-explore/mlx)'s CLIP example with some customized modification. MLX is a machine learning framework that provides native Apple Silicon hardware support.

## Conversion

To convert an example CoreNet's CLIP model to the example MLX CLIP using the files in this directory:

```bash
cd mlx_examples/clip/

# Install required dependencies
# We assume that the main requirements.txt is already installed.
pip install -r requirements.txt

# Convert the model
python main_clip_to_mlx.py \
    --common.config-file "../../projects/range_augment/clip/clip_vit_base.yaml" \
    --model.multi-modal-image-text.pretrained https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/clip_vit_base_16.pt \
    --common.results-loc results/mlx_model/

# Try example inference
python clip.py
```

## Benchmarking results

Comparing to PyTorch, given the input as `["a photo of cat", "a photo of dog"]` prompt
and the `assets/{cat,dog}.jpeg` images. The results are the following on a M2 Ultra:


| Model | PyTorch time 100iters (s) | MLX time 100iters (s) | Speedup (%) |
| :-----| :----------------------------- | :------------------------- | :---------- |
| FP16 Base variant  | 2.7322 | 1.0743 | 60.68% |
| FP16 Huge variant  | 4.9098 | 4.3189 | 12.04% |
