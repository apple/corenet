# MobileOne: An Improved One millisecond Mobile Backbone

[MobileOne](https://arxiv.org/abs/2206.04040) is an efficient CNN architecture that attains SOTA accuracy to latency tradeoff.

We provide training and evaluation code of MobileOne, along with pretrained models and configuration files for the following tasks:

## ImageNet classification

### Training
Single node training with 4 A100 GPUs of `MobileOne-S1` model can be done using below command:

```bash
export CFG_FILE="projects/mobileone/classification/mobileone_s1_in1k.yaml"
corenet-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

### Evaluation

We evaluate the model on a single GPU using following command:

```bash
 export MODEL_WEIGHTS="https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/mobileone/imagenet-1k/mobileone-s1.pt"
 export CFG_FILE="projects/mobileone/classification/mobileone_s1_in1k.yaml"
 export DATASET_PATH="/mnt/vision_datasets/imagenet/validation/" # change to the ImageNet validation path
 CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CFG_FILE --model.classification.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.root_val=$DATASET_PATH
```

This should give:
```
top1=75.316 || top5=92.544
```

## Citation
If you find our work useful, please cite following papers:

```BibTeX
@article{mobileone2022,
  title={An Improved One millisecond Mobile Backbone},
  author={Vasu, Pavan Kumar Anasosalu and Gabriel, James and Zhu, Jeff and Tuzel, Oncel and Ranjan, Anurag},
  journal={arXiv preprint arXiv:2206.04040},
  year={2022}
}

@inproceedings{mehta2022cvnets,
     author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad},
     title = {CVNets: High Performance Library for Computer Vision},
     year = {2022},
     booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
     series = {MM '22}
}
```
