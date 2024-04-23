# Training CatLIP Models on the Single-label Object Classification Datasets

Below are instructions for [training](#training) a pre-trained CatLIP model on the ImageNet dataset and [evaluating](#evaluation) its accuracy. Similarly, this fine-tuning process can be applied to the Places365 dataset. 

We also provide [pre-trained model weights](#pretrained-model-weights) for different classification models.

## Training

To train ViT-B, pretrained using CatLIP, on [ImageNet](https://image-net.org) using a single A100 GPU, run the following command:

```bash
export CFG_FILE=projects/catlip/image_classification/imagenet/vit_base.yaml
corenet-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

We assume that the training and validation data is located at `/mnt/imagenet/training` and `/mnt/imagenet/validation` folders, respectively. 

## Evaluation

To evaluate the finetuned `ViT-B` model on the validation set of the ImageNet, run the following command:

```bash
export CFG_FILE=projects/catlip/image_classification/imagenet/vit_base.yaml
export DATASET_PATH="/mnt/vision_datasets/imagenet/validation/" # change to the ImageNet validation path
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/image-classification/imagenet-1k/vit_base.pt
CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CFG_FILE --common.override-kwargs dataset.root_val=$DATASET_PATH model.classification.pretrained=$MODEL_WEIGHTS model.resume_exclude_scopes=''
```

This should give
```
top1={'logits': 84.158} || top5={'logits': 97.006}
```

## Pretrained Model Weights

### ImageNet-1k

| Model | Resolution | Top-1 | Pretrained weights |
| ---- | ---- | ---- | ---- |
| ViT-B/16 | 224x224 | 84.2 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/image-classification/imagenet-1k/vit_base.pt) |
| ViT-L/16 | 224x224 | 86.8 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/image-classification/imagenet-1k/vit_large.pt) |
| ViT-H/16 | 224x224 | 87.1 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/image-classification/imagenet-1k/vit_huge.pt) |
| ---- | ---- | ---- | ---- |
| ViT-B/16 | 512x512 | 86.1 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/image-classification/imagenet-1k/vit_base_512x512.pt) |
| ViT-L/16 | 512x512 | 88.6 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/image-classification/imagenet-1k/vit_large_512x512.pt) |
| ViT-H/16 | 512x512 | 88.7 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/image-classification/imagenet-1k/vit_huge_512x512.pt) |

### Places365

| Model | Resolution | Top-1 | Pretrained weights |
| ---- | ---- | ---- | ---- |
| ViT-B/16 | 224x224 | 59.3 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/image-classification/places365/vit_base.pt) |
| ViT-L/16 | 224x224 | 60.4 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/image-classification/places365/vit_large.pt) |
| ViT-H/16 | 224x224 | 60.3 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/image-classification/places365/vit_huge.pt) |
| ---- | ---- | ---- | ---- |
| ViT-B/16 | 512x512 | 59.4 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/image-classification/places365/vit_base_512x512.pt) |
| ViT-L/16 | 224x224 | 60.7 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/image-classification/places365/vit_large_512x512.pt) |
| ViT-H/16 | 224x224 | 61.1 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/image-classification/places365/vit_huge_512x512.pt) |
