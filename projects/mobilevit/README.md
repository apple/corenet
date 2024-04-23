# MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer

[MobileViT](https://arxiv.org/abs/2110.02178) is a light-weight vision transformer that combines the strengths of CNNs and transformers. See [paper](https://arxiv.org/abs/2110.02178) for details.

We provide training and evaluation code of MobileViT, trained with [RangeAugment](https://arxiv.org/abs/2212.10553), along with pretrained models and configuration files for the following tasks:

## Image classification on the ImageNet dataset

### Training
To train `MobileViT-Small` model on the [ImageNet](https://image-net.org) using a single node with 8 A100 GPUs, run the following command:

```bash
export CFG_FILE="projects/range_augment/classification/mobilevit_v1.yaml"
corenet-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

We assume that the training and validation data is located in `/mnt/imagenet/training` and `/mnt/imagenet/validation` folders, respectively. 

### Evaluation

To evaluate the pre-trained `MobileViT-Small` model on the validation set of the ImageNet on a single GPU, run the following command:

```bash
export CFG_FILE="projects/range_augment/classification/mobilevit_v1.yaml"
export DATASET_PATH="/mnt/vision_datasets/imagenet/validation/" # change to the ImageNet validation path
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/classification/mobilevit_small.pt
CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CFG_FILE --model.classification.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.root_val=$DATASET_PATH
```

This should give
```
top1={'logits': 78.194} || top5={'logits': 94.064}
```

## Object detection and instance segmentation using Mask R-CNN on COCO

### Training

To train the MobileViT-Small with [Mask R-CNN](https://arxiv.org/abs/1703.06870) as a detection backbone on the [COCO](https://cocodataset.org/#home) dataset using a single node with 8 A100 GPUs, run the following command:

```bash 
export CFG_FILE="projects/range_augment/detection/maskrcnn_mobilevit.yaml"
corenet-train --common.config-file $CFG_FILE --common.results-loc detection_results
```

We assume that the training and validation datasets are located in `/mnt/vision_datasets/coco` directory. 

### Evaluation

To evaluate the pre-trained detection model on the validation set of the COCO on a single GPU, run the following command:

```bash
 export CFG_FILE="projects/range_augment/detection/maskrcnn_mobilevit.yaml"
 export MODEL_WEIGHTS="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/detection/maskrcnn_mobilevit.pt"
 CUDA_VISIBLE_DEVICES=0 corenet-eval-det --common.config-file $CFG_FILE --model.detection.pretrained $MODEL_WEIGHTS --evaluation.detection.resize-input-images --evaluation.detection.mode validation_set 
```

This should give for annotation type *bbox*
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.640
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.456
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.277
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.452
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.537
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.336
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.549
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.581
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.411
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.712
```
and for annotation type *segm*
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.377
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.606
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.403
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.206
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.404
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.543
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.313
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.524
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.348
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.559
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.669
```

## Semantic segmentation on the ADE20k dataset

### Training

To train the MobileViT-Small with [DeepLabv3](https://arxiv.org/abs/1706.05587) as a segmentation head on the [ADE20k](https://groups.csail.mit.edu/vision/datasets/ADE20K/) dataset using a single A100 GPUs, run the following command:

```bash
export CFG_FILE="projects/range_augment/segmentation/ade20k/deeplabv3_mobilevit.yaml"
corenet-train --common.config-file $CFG_FILE --common.results-loc segmentation_results
```

We assume that the training and validation datasets are located in `/mnt/vision_datasets/ADEChallengeData2016/` directory. 

### Evaluation

To evaluate the pre-trained segmentation model on the validation set of the ADE20k dataset on a single GPU, run the following command:

```bash
export CFG_FILE="projects/range_augment/segmentation/ade20k/deeplabv3_mobilevit.yaml"
export DATASET_PATH="/mnt/vision_datasets/ADEChallengeData2016/" # change to the ADE20k's path
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/segmentation/ade20k/deeplabv3_mobilevit.pt
CUDA_VISIBLE_DEVICES=0 corenet-eval-seg --common.config-file $CFG_FILE --model.segmentation.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.root_val=$DATASET_PATH
```

This should give
```
mean IoU: 38.49
```

## Citation

If you find our work useful, please cite:

```BibTex
@inproceedings{mehta2022mobilevit,
    title={MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer},
    author={Sachin Mehta and Mohammad Rastegari},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=vh-0sUt8HlG}
}

@inproceedings{mehta2022cvnets, 
     author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad}, 
     title = {CVNets: High Performance Library for Computer Vision}, 
     year = {2022}, 
     booktitle = {Proceedings of the 30th ACM International Conference on Multimedia}, 
     series = {MM '22} 
}
```

## Code and pre-trained models released with the paper

For code and pre-trained models released with the paper, please check [CVNets v0.1](https://github.com/apple/ml-cvnets/tree/cvnets-v0.1).
