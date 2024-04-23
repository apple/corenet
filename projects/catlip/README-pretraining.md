# Training CatLIP Models on Noisy Image-text Datasets

## Setup instructions

Besides set-up instructions in CoreNet, please install NLTK library for CatLIP pre-training using below commands:

```bash
python3 -m pip install nltk==3.8.1 
python3 -m nltk.downloader all
```

## Dataset preparation

To prepare the dataset, see the documentation in the [img_text_tar_dataset.py](../../corenet/data/datasets/multi_modal_img_text/img_text_tar_dataset.py) file. In our experiments, we produce multi-class labels from text captions `on-the-fly`. However, they can be cached for efficiency. See [wordnet_tagged_classification.py](../../corenet/data/datasets/classification/wordnet_tagged_classification.py) for details.

## Training CatLIP on image-text pair dataset

CatLIP leverages our custom ViT implementation that can be used with multi-scale variable batch sampler. CatLIP models are 
trained on multiple nodes, each node with multiple GPUs. Please see comments in configuration files for exact number of 
GPUs and nodes used in our experiments.


An example command for training on `i-th` node is
```bash
export CFG_FILE="PATH_TO_MODEL_CONFIGURATION_FILE"
export RANK=<NODE_ID> * <NUM_GPUS_PER_NODE> # For Node-0, RANK=0; For Node-1, Rank=8, For Node-2, RANK=16, and so on.
export WORLD_SIZE=<NUM_NODES> * <NUM_GPUS_PER_NODE> # WORLD_SIZE=32 nodes * 8 GPUS per node = 256
corenet-train --common.config-file $CFG_FILE --common.results-loc results_catlip --ddp.rank $RANK --ddp.world-size $WORLD_SIZE --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT'
```

***Note***: Do not forget to change the `metadata_file` and `vocab_file` paths in the configuration files.

## Pretraining checkpoints and vocabulary file

The pre-training checkpoints and vocabulary files for DataComp-1.3B can be downloaded using below links:

| Name | Link |
| ---- | ---- |
| Vocabulary file | [Vocabulary](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/pretraining_vocab/datacomp_1_2B_vocab.pkl) |
| ViT-B/16 | [ViT-B/16 Pre-trained weights](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/pretrained_models/vit_base.pt) |
| ViT-L/16 | [ViT-L/16 Pre-trained weights](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/pretrained_models/vit_large.pt) |
| ViT-H/16 | [ViT-H/16 Pre-trained weights](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/catlip/pretrained_models/vit_huge.pt) |

The pre-training checkpoints are the checkpoints obtained with EMA at the end of training.
