# Training CLIP Models with RangeAugment

## Image-text dataset preparation

To prepare the dataset, see the documentation in the [img_text_tar_dataset.py](../../corenet/data/datasets/multi_modal_img_text/img_text_tar_dataset.py) file.

## Training CLIP on image-text pair dataset

CLIP leverages our custom ViT implementation that can be used with multi-scale variable batch sampler. CLIP models are 
trained on multiple nodes, each node with multiple GPUs. Please see comments in configuration files for exact number of 
GPUs and nodes used in our experiments.


An example command for training on `i-th` node is
```bash
export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
export RANK=<NODE_ID> * <NUM_GPUS_PER_NODE> # For Node-0, RANK=0; For Node-1, Rank=8, For Node-2, RANK=16, and so on.
export WORLD_SIZE=<NUM_NODES> * <NUM_GPUS_PER_NODE> # WORLD_SIZE=32 nodes * 8 GPUS per node = 256
corenet-train --common.config-file $CFG_FILE --common.results-loc results_clip --ddp.rank $RANK --ddp.world-size $WORLD_SIZE --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT'
```

***Note***: Do not forget to change the training and validation dataset locations in configuration files.

## Zero-shot evaluation of CLIP models on the ImageNet dataset

CLIP model with ViT-B image backbone can be evaluated at an input resolution of 224x224 using below shell script:

```bash
export CONFIG_FILE="projects/range_augment/clip/clip_vit_base.yaml"
export MODEL_WEIGHTS=https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/clip_vit_base_16.pt
export DATASET_PATH="/mnt/vision_datasets/imagenet/validation/" # change to the ImageNet validation path
CUDA_VISIBLE_DEVICES=0 corenet-eval --common.config-file $CONFIG_FILE --model.multi-modal-image-text.pretrained $MODEL_WEIGHTS --common.override-kwargs dataset.multi_modal_img_text.zero_shot_img_cls_dataset_name="imagenet" dataset.root_val=$DATASET_PATH
```

## Results

We should get the following zero-shot top-1 accuracy when CLIP models are evaluated on the ImageNet at different input resolutions.

| Model            | 160   | 192   | 224   | 256   | 288   | Config                                          | Weights                                                                                             | Logs                                                                                                       |
|------------------|-------|-------|-------|-------|-------|-------------------------------------------------|-----------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| CLIP w/ ViT-B/16 | 69.26 | 71.07 | 71.84 | 72.34 | 72.82 | [CLIP-ViT-B/16_Config](clip/clip_vit_base.yaml) | [CLIP-ViT-B/16_Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/clip_vit_base_16.pt) | [CLIP-ViT-B/16_Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/clip_vit_base_16_logs.txt) |
| CLIP w/ ViT-H/16 | 76.13 | 77.35 | 77.92 | 78.41 | 78.56 | [CLIP-ViT-H/16_Config](clip/clip_vit_huge.yaml) | [CLIP-ViT-H/16_Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/clip_vit_huge_16.pt) | [CLIP-ViT-H/16_Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/clip_vit_huge_16_logs.txt) |

***Note:*** For CLIP models, we found EMA and best checkpoints deliver similar performance. Here, we report the results for best checkpoint.

## Fine-tuning CLIP on the ImageNet dataset

Configuration files for fine-tuning clip model are [here](./clip_finetune_imagenet/). Please follow instructions for training and evaluation in the [classification readme file](README-classification.md).

We finetune the ViT backbone from CLIP model on the ImageNet dataset for 10 epochs. Below are the results:

| Model    | Top-1 @ 224x224 | Config                                                               | Weights         | Logs                                                                                                             |
|----------|-----------------|----------------------------------------------------------------------|-----------------|------------------------------------------------------------------------------------------------------------------|
| ViT-B/16 | 84.31           | [CLIP-ViT-B/16_FT_Config](clip_finetune_imagenet/clip_vit_base.yaml) | [CLIP-ViT-B/16_FT_Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/vit_base_16_ft_in1k.pt) | [CLIP-ViT-B/16_FT_Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/vit_huge_16_ft_in1k_logs.txt) |
| ViT-H/16 | 86.90           | [CLIP-ViT-H/16_FT_Config](clip_finetune_imagenet/clip_vit_huge.yaml) | [CLIP-ViT-H/16_FT_Wts](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/vit_huge_16_ft_in1k.pt) | [CLIP-ViT-H/16_FT_Logs](https://docs-assets.developer.apple.com/ml-research/models/cvnets-v2/examples/range_augment/clip/vit_base_16_ft_in1k_logs.txt) |
