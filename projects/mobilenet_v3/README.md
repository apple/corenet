# MobileNetv3

[MobileNetv3](https://arxiv.org/abs/1905.02244) uses an architecture search to design an efficient vision backbone.

We provide training and evaluation code of MobileNetv3, along with pretrained models and configuration files for the following tasks:

## Image classification on the ImageNet dataset

### Training
To train `MobileNetv3-Large` on the [ImageNet](https://image-net.org) using a single node with 4 A100 GPUs, run the following command:

```bash
export CFG_FILE="projects/mobilenet_v3/classification/mobilenetv3_large_in1k.yaml"
corenet-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

We assume that the training and validation data is located in `/mnt/imagenet/training` and `/mnt/imagenet/validation` folders, respectively. 

### Evaluation

To evaluate the pre-trained `MobileNetv3-Large` model on the validation set of the ImageNet dataset, run the following command:

```bash
export CFG_FILE="projects/mobilenet_v3/classification/mobilenetv3_large_in1k.yaml"
export DATASET_PATH="/mnt/vision_datasets/imagenet/validation/" # change to the ImageNet validation path
export MODEL_WEIGHTS="https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv3-large.pt"
CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CFG_FILE --model.classification.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.root_val=$DATASET_PATH
```

This should give:
```
top1=75.138 || top5=92.424
```

## Pretrained Models

### Classification (ImageNet)
| Model | Parameters | Top-1 | Pretrained weights | Config file | Logs |
| ---  | --- | --- | --- | --- | --- |
| MobileNetv3-small | 2.5 M | 66.65 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv3-small.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv3-small.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv3-small.logs) |
| MobileNetv3-large | 5.4 M | 75.13 | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv3-large.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv3-large.yaml) | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/classification/mobilenetv3-large.logs) |


## Citation
If you find our work useful, please cite following papers:

```BibTeX 
@article{Howard2019SearchingFM,
  title={Searching for MobileNetV3},
  author={Andrew G. Howard and Mark Sandler and Grace Chu and Liang-Chieh Chen and Bo Chen and Mingxing Tan and Weijun Wang and Yukun Zhu and Ruoming Pang and Vijay Vasudevan and Quoc V. Le and Hartwig Adam},
  journal={2019 IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2019},
  pages={1314-1324},
  url={https://api.semanticscholar.org/CorpusID:146808333}
}

@inproceedings{mehta2022cvnets, 
     author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad}, 
     title = {CVNets: High Performance Library for Computer Vision}, 
     year = {2022}, 
     booktitle = {Proceedings of the 30th ACM International Conference on Multimedia}, 
     series = {MM '22} 
}
```
