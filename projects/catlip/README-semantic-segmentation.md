# Training and Evaluating CatLIP Pre-trained Models on the ADE20k dataset for Semantic Segmentation

Below are instructions for [training](#training-deeplabv3) a pre-trained CatLIP model with DeepLabv3 on the ADE20k dataset and [evaluating](#evaluating-deeplabv3) its accuracy.

We also provide [pre-trained model weights](#pre-trained-models) for different segmentation models.

## Training DeepLabv3

Single node `8 A100 GPU` training of DeepLabv3 with CatLIP's ViT-B/16 image backbone can be done using below command:

```bash
export CFG_FILE=projects/catlip/semantic_segmentation/deeplabv3_vit_base.yaml
corenet-train --common.config-file $CFG_FILE --common.results-loc detection_results
```

Please see [semantic segmentation](./semantic_segmentation) folder for configuration files of different ViT variants.

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

## Evaluating DeepLabv3

Evaluation on the validation set can be done using the below command:

```bash
export DATASET_PATH="/mnt/vision_datasets/ADEChallengeData2016/" # Change the path
export CFG_FILE=projects/catlip/semantic_segmentation/deeplabv3_vit_base.yaml
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/semantic-segmentation/ade20k/deeplabv3_vit_base.pt
CUDA_VISIBLE_DEVICES=0 corenet-eval-seg --common.config-file $CFG_FILE \
--model.segmentation.pretrained $MODEL_WEIGHTS \
--common.override-kwargs dataset.root_val=$DATASET_PATH
```

This should give

```
mean IoU: 50.12
```

## Pretrained Model Weights

| Model | mIoU | Pretrained weights |
| ---- | ---- | ---- |
| Mask R-CNN w/ ViT-B/16 | 50.1 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/semantic-segmentation/ade20k/deeplabv3_vit_base.pt) |
| Mask R-CNN w/ ViT-L/16 | 54.8 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/semantic-segmentation/ade20k/deeplabv3_vit_large.pt) |
| Mask R-CNN w/ ViT-H/16 | 55.6 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/semantic-segmentation/ade20k/deeplabv3_vit_huge.pt) |
