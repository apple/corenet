# Training CLIP Models on Image-Text Dataset

Below is an example for [training](#training-clip) a CLIP model on image-text dataset, and evaluating it's [zero-shot image classification](#zero-shot-image-classification) accuracy on the ImageNet dataset. In our experiments, we used [DataComp-1.3B](https://arxiv.org/abs/2304.14108).

## Image-text dataset preparation

To prepare the dataset, see the documentation in the [img_text_tar_dataset.py](../../corenet/data/datasets/multi_modal_img_text/img_text_tar_dataset.py) file.

## Training CLIP

CLIP leverages our custom ViT implementation that can be used with multi-scale variable batch sampler. CLIP models are 
trained with [RangeAugment](https://arxiv.org/abs/2212.10553) on multiple nodes, each node with multiple GPUs. Please see comments in configuration files for exact number of GPUs and nodes used in our experiments.


An example command for training on `i-th` node is
```bash
export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
export RANK=<NODE_ID> * <NUM_GPUS_PER_NODE> # For Node-0, RANK=0; For Node-1, Rank=8, For Node-2, RANK=16, and so on.
export WORLD_SIZE=<NUM_NODES> * <NUM_GPUS_PER_NODE> # WORLD_SIZE=32 nodes * 8 GPUS per node = 256
corenet-train --common.config-file $CFG_FILE --common.results-loc results_clip --ddp.rank $RANK --ddp.world-size $WORLD_SIZE --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT'
```

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

## Zero-shot image classification

CLIP model, pretrained on DataComp-1.3B, with ViT-B image backbone can be evaluated at an input resolution of 224x224 using below shell script:

```bash
export CONFIG_FILE="projects/clip/clip_vit_base.yaml"
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/clip/clip_vit_base.pt
export DATASET_PATH="/mnt/vision_datasets/imagenet/validation/" # change to the ImageNet validation path
CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CONFIG_FILE --model.multi-modal-image-text.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.multi_modal_img_text.zero_shot_img_cls_dataset_name=imagenet dataset.root_val=$DATASET_PATH
```

This should give
```
top1={'zero_shot_image_logits': 75.042}
```

## Citation

If you find our work useful, please cite:

```BibTex 

@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International conference on machine learning},
  pages={8748--8763},
  year={2021},
  organization={PMLR}
}

@article{mehta2022rangeaugment,
  title={RangeAugment: Efficient Online Augmentation with Range Learning},
  author = {Mehta, Sachin and Naderiparizi, Saeid and Faghri, Fartash and Horton, Maxwell and Chen, Lailin and Farhadi, Ali and Tuzel, Oncel and Rastegari, Mohammad},
  journal={arXiv preprint arXiv:2212.10553},
  year={2022},
  url={https://arxiv.org/abs/2212.10553},
}

@inproceedings{mehta2022cvnets, 
     author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad}, 
     title = {CVNets: High Performance Library for Computer Vision}, 
     year = {2022}, 
     booktitle = {Proceedings of the 30th ACM International Conference on Multimedia}, 
     series = {MM '22} 
}
```


