# MobileViTv2: Separable Self-attention for Mobile Vision Transformers

[MobileViTv2](https://arxiv.org/abs/2206.02680) is an enhancement of MobileViT that adds separable self-attention. See [paper](https://arxiv.org/abs/2206.02680) for details.

We provide training and evaluation code of MobileViTv2, along with pretrained models and configuration files for the following tasks:

## Image classification on the ImageNet dataset

### Training
To train `MobileViTv2-2.0` model on [ImageNet](https://image-net.org) using a single node with 8 A100 GPUs, run the following command:

```bash
export CFG_FILE="projects/mobilevit_v2/classification/mobilevitv2_2.0_in1k.yaml"
corenet-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

We assume that the training and validation data is located in `/mnt/imagenet/training` and `/mnt/imagenet/validation` folders, respectively. 

### Evaluation

To evaluate the pre-trained MobileViTv2 2.0 model on the validation set of the ImageNet, run the following command:

```bash
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-2.0.pt
export CFG_FILE="projects/mobilevit_v2/classification/mobilevitv2_2.0_in1k.yaml"
export DATASET_PATH="/mnt/vision_datasets/imagenet/validation/" # change to the ImageNet validation path
CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CFG_FILE --model.classification.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.root_val=$DATASET_PATH
```

This should give
```
top1=81.17 || top5=95.378
```

To evaluate the fine-tuned model at higher resolution, run the following command:
```bash
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-2.0.pt
export CFG_FILE="projects/mobilevit_v2/classification/mobilevitv2_2.0_ft_384x384.yaml"
export DATASET_PATH="/mnt/vision_datasets/imagenet/validation/" # change to the ImageNet validation path
CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CFG_FILE --common.override-kwargs dataset.root_val=$DATASET_PATH model.classification.pretrained=$MODEL_WEIGHTS
```

This should give
```
top1=82.18 || top5=95.928
```

## Object detection using SSD on COCO

### Training

To train the MobileViTv2 2.0 with [SSD](https://arxiv.org/abs/1512.02325) as a detection backbone on the [COCO](https://cocodataset.org/#home) dataset using a single node with 4 A100 GPUs, run the following command:

```bash
export CFG_FILE="projects/mobilevit_v2/detection/mobilevitv2_2.0_ssd_coco.yaml"
corenet-train --common.config-file $CFG_FILE --common.results-loc detection_results
```

We assume that the training and validation datasets are located in `/mnt/vision_datasets/coco` directory. 

### Evaluation

To evaluate the pre-trained detection model on the validation set of the COCO, run the following command:

```bash
export CFG_FILE="projects/mobilevit_v2/detection/mobilevitv2_2.0_ssd_coco.yaml"
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-2.0.pt
CUDA_VISIBLE_DEVICES=0 corenet-eval-det --common.config-file $CFG_FILE --common.results-loc seg_results --model.detection.pretrained $MODEL_WEIGHTS --evaluation.detection.resize-input-images --evaluation.detection.mode validation_set 
```

This should give
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.302
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.501
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.308
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.092
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.319
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.514
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.266
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.402
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.425
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.153
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.477
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.663
```

## Semantic segmentation on the ADE20k dataset

### Training

To train the MobileViTv2 1.0 with [DeepLabv3](https://arxiv.org/abs/1706.05587) as a segmentation head on the [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/) dataset using a single A100 GPUs, run the following command:

```bash
export CFG_FILE="projects/mobilevit_v2/segmentation/deeplabv3_mobilevitv2_1.0_ade20k.yaml"
corenet-train --common.config-file $CFG_FILE --common.results-loc segmentation_results
```

We assume that the training and validation datasets are located in `/mnt/vision_datasets/ADEChallengeData2016/` directory. 

### Evaluation

To evaluate the pre-trained segmentation model on the validation set of the ADE20k dataset, run the following command:

```bash
export CFG_FILE="projects/mobilevit_v2/segmentation/deeplabv3_mobilevitv2_1.0_ade20k.yaml"
export DATASET_PATH="/mnt/vision_datasets/ADEChallengeData2016/" # change to the ADE20k's path
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.0.pt
CUDA_VISIBLE_DEVICES=0 corenet-eval-seg --common.config-file $CFG_FILE --model.segmentation.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.root_val=$DATASET_PATH
```

This should give
```
mean IoU: 37.06
```

## Pretrained Models

<details>
  <summary>Exapnd the section to see available pre-trained models across different tasks.</summary>

### Classification
#### MobileViTv2 (256x256)
| Model | Parameters | Top-1 | Pretrained weights | Config file | Logs |
| ---  | --- | --- | --- | --- |  --- |
| MobileViTv2-0.5 | 1.4 M | 70.18 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-0.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-0.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-0.5.logs) |
| MobileViTv2-0.75 | 2.9 M | 75.56 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-0.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-0.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-0.75.logs) |
| MobileViTv2-1.0 | 4.9 M | 78.09 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.0.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.0.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.0.logs) |
| MobileViTv2-1.25 | 7.5 M | 79.65 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.25.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.25.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.25.logs) |
| MobileViTv2-1.5 | 10.6 M | 80.38 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.5.logs) |
| MobileViTv2-1.75 | 14.3 M | 80.84 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-1.75.logs) |
| MobileViTv2-2.0 | 18.4 M | 81.17 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-2.0.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-2.0.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/256x256/mobilevitv2-2.0.logs) |

#### MobileViTv2 (Trained on 256x256 and Finetuned on 384x384)
| Model | Parameters | Top-1 | Pretrained weights | Config file | Logs |
| ---  | --- | --- | --- | --- |  --- |
| MobileViTv2-0.5 | 1.4 M | 72.14 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-0.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-0.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-0.5.logs) |
| MobileViTv2-0.75 | 2.9 M | 76.98 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-0.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-0.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-0.75.logs) |
| MobileViTv2-1.0 | 4.9 M | 79.68 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-1.0.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-1.0.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-1.0.logs) |
| MobileViTv2-1.25 | 7.5 M | 80.94 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-1.25.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-1.25.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-1.25.logs) |
| MobileViTv2-1.5 | 10.6 M | 81.50 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-1.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-1.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-1.5.logs) |
| MobileViTv2-1.75 | 14.3 M | 82.04 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-1.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-1.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-1.75.logs) |
| MobileViTv2-2.0 | 18.4 M | 82.17 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-2.0.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-2.0.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet1k/384x384/mobilevitv2-2.0.logs) |


#### MobileViTv2 (Trained on ImageNet-21k and Finetuned on ImageNet-1k 256x256)
| Model | Parameters | Top-1 | Pretrained weights | Config file | Logs |
| ---  | --- | --- | --- | --- |  --- |
| MobileViTv2-1.5 | 10.6 M | 81.46 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/256x256/mobilevitv2-1.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/256x256/mobilevitv2-1.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/256x256/mobilevitv2-1.5.logs) |
| MobileViTv2-1.75 | 14.3 M | 81.94 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/256x256/mobilevitv2-1.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/256x256/mobilevitv2-1.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/256x256/mobilevitv2-1.75.logs) |
| MobileViTv2-2.0 | 18.4 M | 82.36 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/256x256/mobilevitv2-2.0.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/256x256/mobilevitv2-2.0.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/256x256/mobilevitv2-2.0.logs) |

#### MobileViTv2 (Trained on ImageNet-21k, Finetuned on ImageNet-1k 256x256, and Finetuned on ImageNet-1k 384x384)
| Model | Parameters | Top-1 | Pretrained weights | Config file | Logs |
| ---  | --- | --- | --- | --- |  --- |
| MobileViTv2-1.5 | 10.6 M | 82.60 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/384x384/mobilevitv2-1.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/384x384/mobilevitv2-1.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/384x384/mobilevitv2-1.5.logs) |
| MobileViTv2-1.75 | 14.3 M | 82.93 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/384x384/mobilevitv2-1.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/384x384/mobilevitv2-1.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/384x384/mobilevitv2-1.75.logs) |
| MobileViTv2-2.0 | 18.4 M | 83.41 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/384x384/mobilevitv2-2.0.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/384x384/mobilevitv2-2.0.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilevitv2/imagenet21k_to_1k/384x384/mobilevitv2-2.0.logs) |

### Object Detection (MS-COCO)

| Model | Parameters | MAP | Pretrained weights | Config file | Logs |
| ---  | --- | --- | --- | --- |  --- |
| SSD MobileViTv2-0.5 | 2.0 M | 21.24 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-0.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-0.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-0.5.logs) |
| SSD MobileViTv2-0.75 | 3.6 M | 24.57 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-0.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-0.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-0.75.logs) |
| SSD MobileViTv2-1.0 | 5.6 M | 26.47 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.0.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.0.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.0.logs) |
| SSD MobileViTv2-1.25 | 8.2 M | 27.85 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.25.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.25.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.25.logs) |
| SSD MobileViTv2-1.5 | 11.3 M | 28.83 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.5.logs) |
| SSD MobileViTv2-1.75 | 14.9 M | 29.52 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-1.75.logs) |
| SSD MobileViTv2-2.0 | 19.1 M | 30.21 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-2.0.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-2.0.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/mobilevitv2/coco-ssd-mobilevitv2-2.0.logs) |

### Segmentation (ADE 20K)
Note: The number of parameters reported does not include the auxiliary branches.

| Model | Parameters | mIoU | Pretrained weights | Config file | Logs |
| ---  | --- | --- | --- | --- |  --- |
| PSPNet MobileViTv2-0.5 | 3.6 M | 31.77 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-0.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-0.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-0.5.logs) |
| PSPNet MobileViTv2-0.75 | 6.2 M | 35.22 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-0.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-0.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-0.75.logs) |
| PSPNet MobileViTv2-1.0 | 9.4 M | 36.57 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-1.0.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-1.0.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-1.0.logs) |
| PSPNet MobileViTv2-1.25 | 13.2 M | 38.76 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-1.25.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-1.25.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-1.25.logs) |
| PSPNet MobileViTv2-1.5 | 17.6 M | 38.74 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-1.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-1.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-1.5.logs) |
| PSPNet MobileViTv2-1.75 | 22.5 M | 39.82 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-1.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-1.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/pspnet-mobilevitv2-1.75.logs) |
| DeepLabv3 MobileViTv2-0.5 | 6.3 M | 31.93 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-0.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-0.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-0.5.logs) |
| DeepLabv3 MobileViTv2-0.75 | 9.6 M | 34.70 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-0.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-0.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-0.75.logs) |
| DeepLabv3 MobileViTv2-1.0 | 13.4 M | 37.06 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.0.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.0.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.0.logs) |
| DeepLabv3 MobileViTv2-1.25 | 17.7 M | 38.42 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.25.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.25.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.25.logs) |
| DeepLabv3 MobileViTv2-1.5 | 22.6 M | 38.91 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.5.logs) |
| DeepLabv3 MobileViTv2-1.75 | 28.1 M | 39.53 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-1.75.logs) |
| DeepLabv3 MobileViTv2-2.0 | 34.0 M | 40.94 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-2.0.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-2.0.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/mobilevitv2/deeplabv3-mobilevitv2-2.0.logs) |

### Segmentation (Pascal VOC 2012)
| Model | Parameters | mIoU | Pretrained weights | Config file | Logs |
| ---  | --- | --- | --- | --- |  --- |
| PSPNet MobileViTv2-0.5 | 3.6 M | 74.62 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-0.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-0.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-0.5.logs) |
| PSPNet MobileViTv2-0.75 | 6.2 M | 77.44 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-0.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-0.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-0.75.logs) |
| PSPNet MobileViTv2-1.0 | 9.4 M | 78.92 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-1.0.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-1.0.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-1.0.logs) |
| PSPNet MobileViTv2-1.25 | 13.2 M | 79.40 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-1.25.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-1.25.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-1.25.logs) |
| PSPNet MobileViTv2-1.5 | 17.5 M | 79.93 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-1.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-1.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/pspnet-mobilevitv2-1.5.logs) |
| DeepLabv3 MobileViTv2-0.5 | 6.2 M | 75.07 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/deeplabv3-mobilevitv2-0.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/deeplabv3-mobilevitv2-0.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/deeplabv3-mobilevitv2-0.5.logs) |
| DeepLabv3 MobileViTv2-1.0 | 13.3 M | 78.94 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/deeplabv3-mobilevitv2-1.0.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/deeplabv3-mobilevitv2-1.0.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/deeplabv3-mobilevitv2-1.0.logs) |
| DeepLabv3 MobileViTv2-1.25 | 17.7 M | 79.68 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/deeplabv3-mobilevitv2-1.25.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/deeplabv3-mobilevitv2-1.25.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/deeplabv3-mobilevitv2-1.25.logs) |
| DeepLabv3 MobileViTv2-1.5 | 22.6 M | 80.30 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/deeplabv3-mobilevitv2-1.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/deeplabv3-mobilevitv2-1.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/pascalvoc/mobilevitv2/deeplabv3-mobilevitv2-1.5.logs) |

</details>

## Citation

If you find our work useful, please cite:

```BibTex
@article{mehta2023separable,
  title={Separable Self-attention for Mobile Vision Transformers},
  author={Sachin Mehta and Mohammad Rastegari},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023},
  url={https://openreview.net/forum?id=tBl4yBEjKi},
  note={}
}

@inproceedings{mehta2022cvnets, 
     author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad}, 
     title = {CVNets: High Performance Library for Computer Vision}, 
     year = {2022}, 
     booktitle = {Proceedings of the 30th ACM International Conference on Multimedia}, 
     series = {MM '22} 
}
```
