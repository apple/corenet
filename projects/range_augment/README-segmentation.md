# Training Segmentations Models with RangeAugment

## Training DeepLabv3 on the ADE20k and PASCAL VOC datasets

Single node `1 A100 GPU` training of different mobile and non-mobile backbones with Deeplabv3 segmentation head 
can be done using below command:

``` 
export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
corenet-train --common.config-file $CFG_FILE --common.results-loc segmentation_results
```

Please see [ade20k](./segmentation/ade20k) and [pascal_voc](./segmentation/pascal_voc) folder for configuration files of different models.

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

## Evaluating the segmentation model

Evaluation on the validation set can be done using the below command:

```bash
export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
export MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS_FILE"
CUDA_VISIBLE_DEVICES=0 corenet-eval-seg --common.config-file $CFG_FILE --common.results-loc seg_results --model.segmentation.pretrained $MODEL_WEIGHTS
```

## Results on the ADE20k dataset

| Backbone             | Top-1 | Config                                                                     | Weights                                                                                                                     | Logs                                                                                                                               |
|----------------------|-------|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| MobileNetv1-1.0      | 39.4  | [DeepLabv3-MV1-config](segmentation/ade20k/deeplabv3_mobilenet_v1.yaml)    | [DeepLabv3-MV1-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_mobilenet_v1.pt)    | [DeepLabv3-MV1-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_mobilenet_v1_logs.txt)    |
| MobileNetv2-1.0      | 38.6  | [DeepLabv3-MV2-config](segmentation/ade20k/deeplabv3_mobilenet_v2.yaml)    | [DeepLabv3-MV2-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_mobilenet_v2.pt)    | [DeepLabv3-MV2-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_mobilenet_v2_logs.txt)    |
| MobileNetv3-Large    | 38.9  | [DeepLabv3-MV3-config](segmentation/ade20k/deeplabv3_mobilenet_v3.yaml)    | [DeepLabv3-MV3-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_mobilenet_v3.pt)    | [DeepLabv3-MV3-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_mobilenet_v3_logs.txt)    |
| MobileViTv1-Small    | 38.5  | [DeepLabv3-MViT-config](segmentation/ade20k/deeplabv3_mobilevit.yaml)      | [DeepLabv3-MViT-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_mobilevit.pt)      | [DeepLabv3-MViT-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_mobilevit_logs.txt)      |
| EfficientNet-B3      | 43.9  | [DeepLabv3-EB3-config](segmentation/ade20k/deeplabv3_efficientnet_b3.yaml) | [DeepLabv3-EB3-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_efficientnet_b3.pt) | [DeepLabv3-R50-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_efficientnet_b3_logs.txt) |
| ResNet-50            | 44.0  | [DeepLabv3-R50-config](segmentation/ade20k/deeplabv3_resnet_50.yaml)       | [DeepLabv3-R50-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_resnet_50.pt)       | [DeepLabv3-R50-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_resnet_50_logs.txt)       |
| ResNet-101           | 46.5  | [DeepLabv3-R101-config](segmentation/ade20k/deeplabv3_resnet_101.yaml)     | [DeepLabv3-R101-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_resnet_101.pt)     | [DeepLabv3-R101-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_resnet_101_logs.txt)     |


## Results on the PASCAL VOC 2012 dataset

| Backbone             | Top-1 | Config                                                                         | Weights                                                                                                                         | Logs                                                                                                                               |
|----------------------|-------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| MobileNetv1-1.0      | 77.2  | [DeepLabv3-MV1-config](segmentation/pascal_voc/deeplabv3_mobilenet_v1.yaml)    | [DeepLabv3-MV1-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/pascal_voc/deeplabv3_mobilenet_v1.pt)    | [DeepLabv3-MV1-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/pascal_voc/deeplabv3_mobilenet_v1_log.txt) |
| MobileNetv2-1.0      | 76.7  | [DeepLabv3-MV2-config](segmentation/pascal_voc/deeplabv3_mobilenet_v2.yaml)    | [DeepLabv3-MV2-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/pascal_voc/deeplabv3_mobilenet_v2.pt)    | [DeepLabv3-MV2-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/pascal_voc/deeplabv3_mobilenet_v2_log.txt) |
| MobileNetv3-Large    | 77.0  | [DeepLabv3-MV3-config](segmentation/pascal_voc/deeplabv3_mobilenet_v3.yaml)    | [DeepLabv3-MV3-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/pascal_voc/deeplabv3_mobilenet_v3.pt)    | [DeepLabv3-MV3-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/pascal_voc/deeplabv3_mobilenet_v3_log.txt) |
| EfficientNet-B3      | 82.0  | [DeepLabv3-EB3-config](segmentation/pascal_voc/deeplabv3_efficientnet_b3.yaml) | [DeepLabv3-EB3-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/pascal_voc/deeplabv3_efficientnet_b3.pt) | [DeepLabv3-EB3-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/pascal_voc/deeplabv3_efficientnet_b3_log.txt) |
| ResNet-50            | 81.2  | [DeepLabv3-R50-config](segmentation/pascal_voc/deeplabv3_resnet_50.yaml)       | [DeepLabv3-R50-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/pascal_voc/deeplabv3_resnet_50.pt)       | [DeepLabv3-R50-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/pascal_voc/deeplabv3_resnet_50_logs.txt)   |
| ResNet-101           | 84.0  | [DeepLabv3-R101-config](segmentation/pascal_voc/deeplabv3_resnet_101.yaml)     | [DeepLabv3-R101-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/pascal_voc/deeplabv3_resnet_101.pt)     | [DeepLabv3-R101-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/pascal_voc/deeplabv3_resnet_101_logs.txt) |

 
## Demo

```bash
export IMG_PATH="http://farm2.staticflickr.com/1173/678795137_bb1a91f659_z.jpg"
export CFG_FILE="projects/range_augment/segmentation/pascal_voc/deeplabv3_resnet_50.yaml"
export MODEL_WEIGHTS="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/pascal_voc/deeplabv3_resnet_50.pt"
corenet-eval-seg --common.config-file $CFG_FILE \
 --common.results-loc results \
 --model.segmentation.pretrained $MODEL_WEIGHTS \
 --evaluation.segmentation.mode single_image \
 --evaluation.segmentation.path "${IMG_PATH}" \
 --evaluation.segmentation.apply-color-map \
 --evaluation.segmentation.save-overlay-rgb-pred
```

***Notes***
   * If running on CPU, please disable mixed precision by adding following arguments to above command `--common.override-kwargs common.mixed_precision=false`
