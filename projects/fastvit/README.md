# FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization

[FastViT](https://arxiv.org/abs/2303.14189) is an efficient hybrid ViT architecture that attains state-of-the-art accuracy to latency tradeoff.

We provide training and evaluation code of FastVit, along with pretrained models and configuration files for image classification on the imagenet dataset.

## ImageNet classification

### Training
Single node 8 A100 GPU training of `FastVit-T8` model can be done using below command:

```bash
export CFG_FILE=projects/fastvit/classification/fastvit_t8_in1k.yaml
corenet-train --common.config-file $CFG_FILE --common.results-loc classification_results
```

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

### Evaluation and Results

We evaluate the model on a single GPU using following command:

```bash
 export CFG_FILE=projects/fastvit/classification/fastvit_t8_in1k.yaml
 export MODEL_WEIGHTS="https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/fastvit/imagenet-1k/fastvit-t8.pt"
 export DATASET_PATH="/mnt/vision_datasets/imagenet/validation/" # change to the ImageNet validation path
 CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CFG_FILE --model.classification.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.root_val=$DATASET_PATH
```

This should give
```
top1=76.284 || top5=93.244
```

### Citation
If you find the work useful, please cite following papers:

```BibTeX
@inproceedings{vasufastvit2023,
  author = {Pavan Kumar Anasosalu Vasu and James Gabriel and Jeff Zhu and Oncel Tuzel and Anurag Ranjan},
  title = {FastViT:  A Fast Hybrid Vision Transformer using Structural Reparameterization},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year = {2023}
}

@inproceedings{mehta2022cvnets, 
     author = {Mehta, Sachin and Abdolhosseini, Farzad and Rastegari, Mohammad}, 
     title = {CVNets: High Performance Library for Computer Vision}, 
     year = {2022}, 
     booktitle = {Proceedings of the 30th ACM International Conference on Multimedia}, 
     series = {MM '22} 
}
```

