# Multi-label Object Classification using CatLIP

Below are instructions for [training](#training-on-coco) a pre-trained CatLIP model on the COCO dataset and [evaluating](#evaluation) its accuracy. 

We also provide [pre-trained model weights](#pretrained-model-weights-on-coco) for different multi-label classification models.

## Training on COCO

To finetune ViT-B, pretrained using CatLIP, on COCO using four A100 GPU, run the following command:

```bash
export CFG_FILE=projects/catlip/multi_label_image_classification/vit_base.yaml
corenet-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

We assume that the training and validation data is located at `/mnt/vision_datasets/coco`. 

## Evaluation

To evaluate the finetuned `ViT-B` model on the validation set of the COCO, run the following command:

```bash
export CFG_FILE=projects/catlip/multi_label_image_classification/vit_base.yaml
export DATASET_PATH="/mnt/vision_datasets/coco" # change to the COCO validation path
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/multi-label-classification/coco/vit_base.pt
CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CFG_FILE --common.override-kwargs dataset.root_val=$DATASET_PATH model.classification.pretrained=$MODEL_WEIGHTS model.resume_exclude_scopes=''
```

This should give
```
'micro': 0.9118, 'macro': 0.8806, 'weighted': 0.8907
```

## Pretrained Model Weights on COCO

| Model | Macro mAP | Pretrained weights |
| ---- | ---- | ---- |
| ViT-B/16 | 88.06 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/multi-label-classification/coco/vit_base.pt) |
| ViT-L/16 | 90.75 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/multi-label-classification/coco/vit_large.pt) |
