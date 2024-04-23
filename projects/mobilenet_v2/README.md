# MobileNetv2

[MobileNetv2](https://arxiv.org/abs/1801.04381) leverages an inverted residual structure to build an efficient backbone.

We provide training and evaluation code of MobileNetv2, along with pretrained models and configuration files for the following tasks:

## Image classification on the ImageNet dataset

### Training
To train MobileNetv2 model on [ImageNet](https://image-net.org) using a single node with 4 A100 GPUs, run the following command:

```bash
export CFG_FILE="projects/mobilenet_v2/classification/mobilenetv2_1.0_in1k.yaml"
corenet-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

We assume that the training and validation data is located in `/mnt/imagenet/training` and `/mnt/imagenet/validation` folders, respectively. 

### Evaluation

To evaluate the pre-trained MobileNetv2 model on the validation set of the ImageNet, run the following command:

```bash
export CFG_FILE="projects/mobilenet_v2/classification/mobilenetv2_1.0_in1k.yaml"
export DATASET_PATH="/mnt/vision_datasets/imagenet/validation/" # change to the ImageNet validation path
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv2-1.00.pt
CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CFG_FILE --model.classification.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.root_val=$DATASET_PATH
```

This should give
```
top1=72.926 || top5=91.174
```

## Semantic Segmentation using DeepLabv3 on the ADE20K dataset

### Training
To train DeepLabv3-MobileNetv2 model on ADE20K using a single node with 4 A100 GPUs, run the following command:

```bash
export CFG_FILE="projects/mobilenet_v2/segmentation/deeplabv3_ade20k.yaml"
corenet-train --common.config-file $CFG_FILE --common.results-loc segmentation_results
```

We assume that the training and validation datasets are located in `/mnt/vision_datasets/ADEChallengeData2016/` directory. 

### Evaluation

To evaluate the pre-trained DeepLabv3-MobileNetv2 model on the validation set of ADE20k, run the following command:

```bash
export CFG_FILE="projects/mobilenet_v2/segmentation/deeplabv3_ade20k.yaml"
export DATASET_PATH="/mnt/vision_datasets/ADEChallengeData2016/" # change to the ADE20k's path
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/deeplabv3-mobilenetv2.pt
CUDA_VISIBLE_DEVICES=0 corenet-eval-seg --common.config-file $CFG_FILE --model.segmentation.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.root_val=$DATASET_PATH
```

This should give
```
mean IoU: 35.20
```

## Pretrained Models
### Classification (ImageNet-1k)
| Model | Parameters | Top-1 | Pretrained weights | Config file | Logs |
| ---  | --- | --- | --- | --- | --- |
| MobileNetv2-0.25 | 1.5 M | 53.57 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv2-0.25.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv2-0.25.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv2-0.25.logs) |
| MobileNetv2-0.5 | 2.0 M | 65.28 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv2-0.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv2-0.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv2-0.5.logs) |
| MobileNetv2-0.75 | 2.6 M | 70.42 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv2-0.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv2-0.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv2-0.75.logs) |
| MobileNetv2-1.00 | 3.5 M | 72.93 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv2-1.00.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv2-1.00.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv2-1.00.logs) |

### Segmentation (ADE20k)

Note: The number of parameters reported does not include the auxiliary branches.

| Model | Parameters | mIoU | Pretrained weights | Config file | Logs |
| ---  | --- | --- | --- | --- |  --- |
| DeepLabv3 MobileNetv2 | 8.0 M | 35.20 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/deeplabv3-mobilenetv2.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/deeplabv3-mobilenetv2.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/segmentation/ade20k/deeplabv3-mobilenetv2.logs) |

## Citation
If you find our work useful, please cite following papers:

```BibTeX
@article{Sandler2018MobileNetV2IR,
  title={MobileNetV2: Inverted Residuals and Linear Bottlenecks},
  author={Mark Sandler and Andrew G. Howard and Menglong Zhu and Andrey Zhmoginov and Liang-Chieh Chen},
  journal={2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2018},
  pages={4510-4520},
  url={https://api.semanticscholar.org/CorpusID:4555207}
}

@inproceedings{mehta2022cvnets, 
     author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad}, 
     title = {CVNets: High Performance Library for Computer Vision}, 
     year = {2022}, 
     booktitle = {Proceedings of the 30th ACM International Conference on Multimedia}, 
     series = {MM '22} 
}
```
