# Training and Evaluating CatLIP Pre-trained Models on the MS-COCO dataset for Object Detection

Below are instructions for [training](#training-mask-r-cnn) a pre-trained CatLIP model with Mask R-CNN on the COCO dataset and [evaluating](#evaluating-mask-r-cnn) its accuracy.

We also provide [pre-trained model weights](#pre-trained-models) for different detection model variants.

## Training Mask R-CNN

The following command trains Mask R-CNN with CatLIP's ViT-B/16 image backbone assuming 8 nodes each with 8 80GB GPUs (verified on A100s):

```bash
export CFG_FILE=projects/catlip/object_detection/maskrcnn_vit_base.yaml
corenet-train --common.config-file $CFG_FILE --common.results-loc detection_results
```

Please see [detection](./object_detection) folder for configuration files of different ViT variants.

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

## Evaluating Mask R-CNN

Evaluation on the validation set can be done using the command below:

```bash
export CFG_FILE=projects/catlip/object_detection/maskrcnn_vit_base.yaml
export DATASET_PATH="/mnt/vision_datasets/coco" # Change the path
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/object-detection/coco/maskrcnn_vit_base.pt
CUDA_VISIBLE_DEVICES=0 corenet-eval-det --common.config-file $CFG_FILE \
--model.detection.pretrained $MODEL_WEIGHTS \
--common.override-kwargs dataset.root_val=$DATASET_PATH
```

This should give

```shell
Evaluate annotation type *bbox* 
DONE (t=25.77s). 
Accumulating evaluation results...             
DONE (t=4.29s). 
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.499 
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.727 
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.546 
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.339 
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.532 
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.652 
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.374 
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.598 
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.629 
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.474 
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.668 
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.771 

Running per image evaluation... 
Evaluate annotation type *segm* 
DONE (t=29.50s). 
Accumulating evaluation results... 
DONE (t=4.24s). 
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.436 
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.694 
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.464 
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.243 
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.466 
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.638 
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.340 
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.531 
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.556 
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.387 
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.595 
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.724
```

## Pretrained Model Weights

| Model | mAP (bbox) | mAP (segm) | Pretrained weights |
| ---- | ---- | ---- | ---- |
| Mask R-CNN w/ ViT-B/16 | 49.9 | 43.6 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/object-detection/coco/maskrcnn_vit_base.pt) |
| Mask R-CNN w/ ViT-L/16 | 52.8 | 46.0 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/object-detection/coco/maskrcnn_vit_large.pt) |
| Mask R-CNN w/ ViT-H/16 | 53.3 | 46.4 | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/object-detection/coco/maskrcnn_vit_huge.pt) |
