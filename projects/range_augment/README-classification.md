# Training Classification Models with RangeAugment

## Training on the ImageNet dataset

Single node 8 A100 GPU training of different mobile and non-mobile classification backbones studied in RangeAugment paper 
can be done using below command:

```bash 
export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
corenet-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

Please see [classification](./classification) folder for configuration files of different models.

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

## Evaluating the classification model

Evaluation can be done using the below command:

```bash
 export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
 export MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS_FILE"
 export DATASET_PATH="PATH_TO_DATASET"
 CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CFG_FILE --common.results-loc classification_results --model.classification.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.root_val=$DATASET_PATH
```

## Results on the ImageNet dataset

| Model             | Top-1 | Config                                                            | Weights                                                                                                   | Logs                                                                                                              | 
|-------------------|-------|-------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| MobileNetv1-1.0   | 73.8  | [MV1-1.0-config](classification/mobilenet_v1.yaml)              | [MV1-1.0-WT](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/mobilenetv1_1.0.pt)     | [MV1-1.0-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/mobilenetv1_1.0_logs.txt)     |
| MobileNetv2-1.0   | 73.0  | [MV2-1.0-config](classification/mobilenet_v2.yaml)              | [MV2-1.0-WT](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/mobilenetv2_1.0.pt)     | [MV2-1.0-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/mobilenetv2_1.0_logs.txt)     |
| MobileNetv3-Large | 75.1  | [MV3-Large-config](classification/mobilenet_v3.yaml)            | [MV3-Large-WT](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/mobilenetv3_large.pt) | [MV3-Large-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/mobilenetv3_large_logs.txt) |
| MobileViTv1-Small | 78.2  | [MViT-Small-config](classification/mobilevit_v1.yaml)           | [MViT-Small-WT](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/mobilevit_small.pt)  | [MViT-Small-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/mobilevit_small_logs.txt)  |
| EfficientNet-B0      | 77.3  | [EB0-config](classification/efficientnet_b0.yaml)               | [EB0-WT](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/efficientnet_b0.pt)         | [EB0-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/efficientnet_b0_logs.txt)         |
| EfficientNet-B1      | 79.5  | [EB1-config](classification/efficientnet_b1.yaml)               | [EB1-WT](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/efficientnet_b1.pt)         | [EB1-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/efficientnet_b1_logs.txt)         |
| EfficientNet-B2      | 81.3  | [EB2-config](classification/efficientnet_b2.yaml)               | [EB2-WT](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/efficientnet_b2.pt)         | [EB2-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/efficientnet_b2_logs.txt)         |
| EfficientNet-B3      | 81.9  | [EB3-config](classification/efficientnet_b3.yaml)               | [EB3-WT](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/efficientnet_b3.pt)         | [EB3-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/efficientnet_b3_logs.txt)         |
| ResNet-50            | 80.2  | [R50-config](classification/resnet_50.yaml)                     | [R50-WT](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/resnet_50.pt)               | [R50-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/resnet_50_logs.txt)               |
| ResNet-101           | 81.9  | [R101-config](classification/resnet_101.yaml)                   | [R101-WT](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/resnet_101.pt)             | [R50-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/resnet_101_logs.txt)              |
| SwinTransformer-Tiny | 81.1  | [Swin-Tiny-config](classification/swin_transformer_tiny.yaml)   | [Swin-Tiny-WT](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/swin_tiny.pt)         | [Swin-Tiny-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/swin_tiny_logs.txt)         |
| SwinTransformer-Small | 82.8  | [Swin-Small-config](classification/swin_transformer_small.yaml) | [Swin-Small-WT](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/swin_small.pt)      | [Swin-Small-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/swin_small_logs.txt)       |

 ***Note:*** For MobileViT, we report results with EMA (as suggested in the paper) while for other models, we use the best checkpoint.
