# MobileNetv1

[MobileNetv1](https://arxiv.org/abs/1704.04861) introduces an efficient backbone that leverages depthwise separable convolutions.

We provide training and evaluation code of MobileNetv1, along with pretrained models and configuration files for image classification on the ImageNet dataset.

## Image classification on the ImageNet dataset

### Training
To train MobileNetv1-1.0 on [ImageNet](https://image-net.org) using a single node with 4 A100 GPUs, run the following command:

```bash
export CFG_FILE=projects/mobilenet_v1/classification/mobilenetv1_1.0_in1k.yaml
corenet-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

We assume that the training and validation data is located at `/mnt/imagenet/training` and `/mnt/imagenet/validation` folders, respectively. 

### Evaluation

To evaluate the pre-trained `MobileNetv1-1.0` model on the validation set of the ImageNet, run the following command:

```bash
export CFG_FILE=projects/mobilenet_v1/classification/mobilenetv1_1.0_in1k.yaml
export DATASET_PATH="/mnt/vision_datasets/imagenet/validation/" # change to the ImageNet validation path
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv1-1.00.pt
CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CFG_FILE --model.classification.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.root_val=$DATASET_PATH
```

This should give
```
top1=74.044 || top5=91.578
```

## Pretrained Models on ImageNet-1k

| Model | Parameters | Top-1 | Pretrained weights | Config file | Logs |
| ---  | --- | --- | --- | --- | --- |
| MobileNetv1-0.25 | 0.5 M | 54.45 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv1-0.25.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv1-0.25.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv1-0.25.logs) |
| MobileNetv1-0.5 | 1.3 M | 65.93 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv1-0.5.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv1-0.5.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv1-0.5.logs) |
| MobileNetv1-0.75 | 2.6 M | 71.44 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv1-0.75.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv1-0.75.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv1-0.75.logs) |
| MobileNetv1-1.00 | 4.2 M | 74.04 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv1-1.00.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv1-1.00.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv1-1.00.logs) |

## Citation
If you find our work useful, please cite following papers:

```BibTeX
@article{Howard2017MobileNetsEC,
  title={MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications},
  author={Andrew G. Howard and Menglong Zhu and Bo Chen and Dmitry Kalenichenko and Weijun Wang and Tobias Weyand and Marco Andreetto and Hartwig Adam},
  journal={ArXiv},
  year={2017},
  volume={abs/1704.04861},
  url={https://api.semanticscholar.org/CorpusID:12670695}
}

@inproceedings{mehta2022cvnets, 
     author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad}, 
     title = {CVNets: High Performance Library for Computer Vision}, 
     year = {2022}, 
     booktitle = {Proceedings of the 30th ACM International Conference on Multimedia}, 
     series = {MM '22} 
}
```
