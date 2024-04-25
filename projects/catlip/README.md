# CatLIP: CLIP-level Visual Recognition Accuracy with 2.7x Faster Pre-training on Web-scale Image-Text Data 
[![arXiv](https://img.shields.io/badge/arXiv-2404.15653-a6dba0.svg)](https://arxiv.org/abs/2404.15653)

`CatLIP` introduces a novel weakly supervised pre-training approach for vision models on web-scale noisy image-text data, *reframing pre-training as a classification task to circumvent computational challenges associated with pairwise similarity computations in contrastive learning*, resulting in a significant 2.7x acceleration in training speed while maintaining high representation quality across various vision tasks.

We provide training and evaluation code along with pretrained models and configuration files for the following tasks:

1. [CatLIP Pre-training](./README-pretraining.md)
2. [Single-label Object Classification with CatLIP](./README-single-label-object-classification.md)
3. [Multi-label Object Classification with CatLIP](./README-multi-label-object-classification.md)
4. [Object Detection with CatLIP](./README-object-detection.md)
4. [Semantic Segmentation with CatLIP](./README-semantic-segmentation.md)

## Citation

If you find our work useful, please cite:

```BibTex 
@article{mehta2024catlip,
  title={CatLIP: CLIP-level Visual Recognition Accuracy with 2.7x Faster Pre-training on Web-scale Image-Text Data}, 
  author={Sachin Mehta and Maxwell Horton and Fartash Faghri and Mohammad Hossein Sekhavat and Mahyar Najibi and Mehrdad Farajtabar and Oncel Tuzel and Mohammad Rastegari},
  year={2024},
  eprint={2404.15653},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@inproceedings{mehta2022cvnets, 
     author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad}, 
     title = {CVNets: High Performance Library for Computer Vision}, 
     year = {2022}, 
     booktitle = {Proceedings of the 30th ACM International Conference on Multimedia}, 
     series = {MM '22} 
}
```
