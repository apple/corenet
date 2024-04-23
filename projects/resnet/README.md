# ResNet

[ResNet](https://arxiv.org/abs/1512.03385) introduces skip connections to develop a more accurate visual recognition backbone.

We provide training and evaluation code of ResNet, along with pretrained models and configuration files for the following tasks:

## Image classification on the ImageNet dataset

### Training
To train ResNet50 on [ImageNet 1k](https://image-net.org) with the advanced recipe, using a single node with 8 A100 GPUs, run the following command:

```bash
export CFG_FILE="projects/resnet/classification/resnet50_in1k.yaml"
corenet-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

We assume that the training and validation data is located in `/mnt/imagenet/training` and `/mnt/imagenet/validation` folders, respectively. 

### Evaluation

To evaluate the pre-trained ResNet50 model on the validation set of the ImageNet, run the following command:

```bash
export CFG_FILE="projects/resnet/classification/resnet50_in1k.yaml"
export MODEL_WEIGHTS="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/advanced/resnet-50-adv.pt"
export DATASET_PATH="/mnt/vision_datasets/imagenet/validation/" # change to the ImageNet validation path
CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CFG_FILE --model.classification.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.root_val=$DATASET_PATH
```

This should give
```
top1=80.37 || top5=95.056
```

## Object detection on the MS-COCO dataset

### Training
To train ResNet50 on MS-COCO using a single node with 8 A100 GPUs, run the following command:

```bash
export CFG_FILE="projects/resnet/detection/ssd_resnet50_coco.yaml"
corenet-train --common.config-file $CFG_FILE --common.results-loc detection_results
```

### Evaluation
To evaluate the pre-trained detection model on the validation set of the COCO, run the following command:

```bash
export CFG_FILE="projects/resnet/detection/ssd_resnet50_coco.yaml"
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/coco-ssd-resnet-50.pt
CUDA_VISIBLE_DEVICES=0 corenet-eval-det --common.config-file $CFG_FILE --common.results-loc detection_results --model.detection.pretrained $MODEL_WEIGHTS --evaluation.detection.resize-input-images --evaluation.detection.mode validation_set 
```

This should give

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.300
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.482
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.073
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.315
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.271
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.402
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.426
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.141
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.475
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.680
```


## Pretrained Models
### Classification (ImageNet-1k)

| Model | Parameters | Top-1 | Pretrained weights | Config file | Logs |
| ---  | --- | --- | --- | --- | --- |
| ResNet-34 | 21.8 M | 74.85 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/resnet-34.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/resnet-34.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/resnet-34.logs) |
| ResNet-50 | 25.6 M | 78.44 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/resnet-50.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/resnet-50.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/resnet-50.logs) |
| ResNet-101 | 44.5 M | 79.81 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/resnet-101.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/resnet-101.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/resnet-101.logs) |
| ResNet-34 (advanced recipe) | 21.8 M | 76.91 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/advanced/resnet-34-adv.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/advanced/resnet-34-adv.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/advanced/resnet-34-adv.logs) |
| ResNet-50 (advanced recipe) | 25.6 M | 80.36 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/advanced/resnet-50-adv.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/advanced/resnet-50-adv.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/advanced/resnet-50-adv.logs) |
| ResNet-101 (advanced recipe) | 44.5 M | 81.68 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/advanced/resnet-101-adv.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/advanced/resnet-101-adv.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/advanced/resnet-101-adv.logs) |

### Object Detection (MS-COCO)
| Model | Parameters | MAP | Pretrained weights | Config file | Logs |
| ---  | --- | --- | --- | --- |  --- |
| SSD ResNet-50 | 28.5 M | 30.0 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/coco-ssd-resnet-50.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/coco-ssd-resnet-50.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/detection/coco-ssd-resnet-50.logs) |

## Citation
If you find our work useful, please cite following papers:

```BibTex
@article{He2015DeepRL,
  title={Deep Residual Learning for Image Recognition},
  author={Kaiming He and X. Zhang and Shaoqing Ren and Jian Sun},
  journal={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2015},
  pages={770-778},
  url={https://api.semanticscholar.org/CorpusID:206594692}
}

@inproceedings{mehta2022cvnets, 
     author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad}, 
     title = {CVNets: High Performance Library for Computer Vision}, 
     year = {2022}, 
     booktitle = {Proceedings of the 30th ACM International Conference on Multimedia}, 
     series = {MM '22} 
}
```
