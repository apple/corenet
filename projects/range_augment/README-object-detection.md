# Training Mask R-CNN for object detection and instance segmentation with RangeAugment

## Training Mask R-CNN on the MS-COCO dataset

Single node `8 A100 GPU` training of different mobile and non-mobile backbones with Mask R-CNN 
can be done using below command:

``` 
export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
corenet-train --common.config-file $CFG_FILE --common.results-loc detection_results
```

Please see [detection](./detection) folder for configuration files of different models.

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

## Evaluating the detection model

Evaluation on the validation set can be done using the below command:

```bash
 export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
 export MODEL_WEIGHTS="PATH_TO_MODEL_WEIGHTS_FILE"
 export DATASET_PATH="PATH_TO_DATASET"
 CUDA_VISIBLE_DEVICES=0 corenet-eval-det --common.config-file $CFG_FILE \
 --common.results-loc seg_results \
 --model.detection.pretrained $MODEL_WEIGHTS --evaluation.detection.resize-input-images \
 --evaluation.detection.mode validation_set \
 --common.override-kwargs dataset.root_val=$DATASET_PATH
```

## Results on the MS-COCO dataset

| Backbone             | BBox mAP | Seg mAP | Config                                                         | Weights                                                                                                                         | Logs                                                                                                                   |
|----------------------|----------|---------|----------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| MobileNetv1-1.0      | 39.4     | 35.6    | [MaskRCNN-MV1-config](detection/maskrcnn_mobilenet_v1.yaml)    | [MaskRCNN-MV1-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_mobilenet_v1.pt)    | [MaskRCNN-MV1-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_mobilenet_v1_logs.txt)    |
| MobileNetv2-1.0      | 38.4     | 34.7    | [MaskRCNN-MV2-config](detection/maskrcnn_mobilenet_v2.yaml)    | [MaskRCNN-MV2-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_mobilenet_v2.pt)    | [MaskRCNN-MV2-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_mobilenet_v2_logs.txt)    |
| MobileNetv3-Large    | 35.6     | 32.5    | [MaskRCNN-MV3-config](detection/maskrcnn_mobilenet_v3.yaml)    | [MaskRCNN-MV3-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_mobilenet_v3.pt)    | [MaskRCNN-MV3-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_mobilenet_v3_logs.txt)    |
| MobileViT-Small      | 42.0     | 37.7    | [MaskRCNN-MViT-config](detection/maskrcnn_mobilevit.yaml)      | [MaskRCNN-MViT-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_mobilevit.pt)    | [MaskRCNN-MV3-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_mobilevit_logs.txt)       |
| EfficientNet-B3      | 44.5     | 39.5    | [MaskRCNN-EB3-config](detection/maskrcnn_efficientnet_b3.yaml) | [MaskRCNN-EB3-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_efficientnet_b3.pt) | [MaskRCNN-EB3-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_efficientnet_b3_logs.txt) |
| ResNet-50            | 44.0     | 39.5    | [MaskRCNN-R50-config](detection/maskrcnn_resnet_50.yaml)       | [MaskRCNN-R50-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_resnet_50.pt)  | [MaskRCNN-R50-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_resnet_50_logs.txt)       |
| ResNet-101           | 46.1     | 41.1    | [MaskRCNN-R101-config](detection/maskrcnn_resnet_101.yaml)     | [MaskRCNN-R101-Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_resnet_101.pt)     | [MaskRCNN-R101-Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_resnet_101_logs.txt)     |


## Demo

```bash
 export IMG_PATH="http://farm2.staticflickr.com/1173/678795137_bb1a91f659_z.jpg"
 export CFG_FILE="projects/range_augment/detection/maskrcnn_resnet_50.yaml"
 export MODEL_WEIGHTS="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_resnet_50.pt"
 corenet-eval-det --common.config-file $CFG_FILE \
 --common.results-loc results \
 --model.detection.pretrained $MODEL_WEIGHTS \
 --model.detection.n-classes 81 \
 --evaluation.detection.resize-input-images \
 --evaluation.detection.mode single_image \
 --evaluation.detection.path "${IMG_PATH}" \
 --model.detection.mask-rcnn.box-score-thresh 0.7
```

***Notes***
   * Adjust the value of bounding box threshold using `--model.detection.mask-rcnn.box-score-thresh` to control the number of boxes and mask instances to be displayed. 
   * If running on CPU, please disable mixed precision by adding following arguments to above command `--common.override-kwargs common.mixed_precision=false`
