# Pretraining OpenELM

## Dataset preparation

OpenELM was pretrained on public datasets. Specifically, our pre-training dataset contains RefinedWeb, PILE, a subset of RedPajama, and a subset of Dolma v1.6. 

> **NOTE:**  Please accept the license of each individual dataset before using it. 

Please process the dataset according to the instructions provided in the following YAML snippet. We used the same structure during our experiments.

```yaml
dataset:
  language_modeling:
    shuffle_data: true
    general_lm:
      # path to HF dataset
      train_data_info: [
        ####### RefinedWeb ########
        # The dataset can be downloaded from HuggingFace and contains 5535 parquet files.
        # https://huggingface.co/datasets/tiiuae/falcon-refinedweb/tree/main/data
        # We renamed the files to 'refinedweb-{file_id:05d}-of-05534.parquet' format.
        {
          # "file_name": "REPLACE_WITH_LOCATION_OF_DOWNLOADED_REFINED_WEB_DATA/refinedweb-{file_id:05d}-of-05534.parquet",
          "text_key": "content",
          "file_id_range": [0, 5534]
        },
        ####### RedPAJAMA ########
        # The urls for Redpajama data are available at https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt.
        # The data can be downloaded from the above URL.
        # The name of the files is different from the format that is expected in CoreNet, so we recommend to rename the files # to the expected format. In general, we expect the redpajama data in following format:
        #     LOCATION_OF_DOWNLOADED_REDPAJAMA_DATA/REDPAJAMA_SUBSET_NAME/REDPAJAMA_SUBSET_NAME-FILE_ID-TOTAL_FILES.jsonl
        # where 
        #   LOCATION_OF_DOWNLOADED_REDPAJAMA_DATA corresponds to the root location (local or s3) of downloaded RedPajama dataset
        #   REDPAJAMA_SUBSET_NAME corresponds to the subset name (e.g., arxiv)
        #   FILE_ID: Uniqiue identifier for the file (e.g., 00001) for the first file in the subset
        #   TOTAL_FILES: Total files in the subset (e.g., 00099)

        # Assume that the subset name is 'arxiv' and it has 100 files (e.g., arxiv-00000-00099.jsonl, arxiv-00001-00099.jsonl, and so on). We can pass it as a below dictionary.
        ### Arxiv
        {
          # "file_name": "LOCATION_OF_DOWNLOADED_REDPAJAMA_DATA/arxiv/arxiv-{file_id:05d}-00099.jsonl", 
          "text_key": "text",
          "file_id_range": [0, 100]
        },
        ### Books
        {
          # Note that there is only one book file in the RedPajama dataset.
          # "file_name": "LOCATION_OF_DOWNLOADED_REDPAJAMA_DATA/book/book-{file_id:05d}-00001.jsonl",
          "text_key": "text",
          "file_id_range": [0, 1]
        },
        ### Github
        {
          # "file_name": "LOCATION_OF_DOWNLOADED_REDPAJAMA_DATA/github/github-{file_id:05d}-00098.jsonl",
          "text_key": "text",
          "file_id_range": [0, 98]
        },
        ### Stackexchange
        {
          # "file_name": "LOCATION_OF_DOWNLOADED_REDPAJAMA_DATA/stackexchange/stackexchange-{file_id:05d}-00001.jsonl",
          "text_key": "text",
          "file_id_range": [0, 1]
        },
        ### wikipedia
        {
          # "file_name": "LOCATION_OF_DOWNLOADED_REDPAJAMA_DATA/wikipedia/wiki-{file_id:05d}-00001.jsonl",
          "text_key": "text",
          "file_id_range": [0, 1]
        },
        ### C4
        {
          # "file_name": "LOCATION_OF_DOWNLOADED_REDPAJAMA_DATA/c4/c4-{file_id:05d}-01024.jsonl",
          "text_key": "text",
          "file_id_range": [0, 1024]
        },
        ####### PILE ########
        # The PILE dataset can be downloaded from HuggingFace. THe dataset contains 1650 parquet files.
        # https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated
        # Similar to RefinedWeb, please rename the files to the expected format.
        {
          # "file_name": "LOCATION_OF_DOWNLOADED_PILE_DATA/pile-{file_id:05d}-of-01650.parquet",
          "text_key": "text",
          "file_id_range": [0, 1650]
        },
        ####### Dolma v1.6 ########
        # The urls for Dolma v1.6 are available at: https://huggingface.co/datasets/allenai/dolma/blob/main/urls/v1_6.txt
        # The dataset files are available as 'json.gz' format. We follow similar steps as RedPajama to rename the files.
        # We recommend users to follow the similar steps and rename the files. Expected format for each subset is 
        # mentioned below.

        ### Gutenberg books
        {
          # "file_name": "LOCATION_OF_DOWNLOADED_DOLMAv1.6_DATA/books/books-{file_id:04d}.json.gz",
          "text_key": "text",
          "file_id_range": [0, 3]
        },
        ### peS2o
        {
          # "file_name": "LOCATION_OF_DOWNLOADED_DOLMAv1.6_DATA/pes2o/pes2o_v2-{file_id:04d}.json.gz",
          "text_key": "text",
          "file_id_range": [0, 26]
        },
        ### reddit
        {
          #"file_name": "LOCATION_OF_DOWNLOADED_DOLMAv1.6_DATA/reddit/reddit-v5-dedupe-pii-nsfw-toxic-{file_id:04d}.json.gz",
          "text_key": "text",
          "file_id_range": [0, 78]
        },
        # stack
        {
          #"file_name": "LOCATION_OF_DOWNLOADED_DOLMAv1.6_DATA/stack/stack-v4-train-{file_id:04d}.json.gz",
          "text_key": "text",
          "file_id_range": [0, 149]
        },
        # wiki
        {
          #"file_name": "LOCATION_OF_DOWNLOADED_DOLMAv1.6_DATA/wiki/en_simple_wiki_v0-{file_id:04d}.json.gz",
          "text_key": "text",
          "file_id_range": [0, 2]
        },
      ]
```

## Training OpenELM models

OpenELM models are trained on multiple nodes, each node with multiple GPUs. Please see comments in [configuration files](../../projects/openelm/pretraining_configs/) for exact number of GPUs and nodes used in our experiments.


An example command for training on `i-th` node is
```bash
export CFG_FILE="PATH_TO_OPENELM_MODEL_CONFIGURATION_FILE"
export RANK=<NODE_ID> * <NUM_GPUS_PER_NODE>
export WORLD_SIZE=<NUM_NODES> * <NUM_GPUS_PER_NODE>
corenet-train --common.config-file $CFG_FILE --ddp.rank $RANK --ddp.world-size $WORLD_SIZE --ddp.dist-url 'tcp://IP_OF_NODE0:FREEPORT'
```

## Pretraining checkpoints, model weights, and logs

### Model weight checkpoints

The following checkpoints include only the model weights. These maybe useful for evaluation or fine-tuning purporses.

| Iteration | OpenELM-270M | OpenELM-450M | OpenELM-1.1B | OpenELM-3B |
| ---- | ---- | ---- | ---- | ---- | 
| 50k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/checkpoint_epoch_0_iter_49999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/checkpoint_epoch_0_iter_49999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/checkpoint_epoch_0_iter_49999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/checkpoint_epoch_0_iter_49999.pt) |
| 100k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/checkpoint_epoch_0_iter_99999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/checkpoint_epoch_0_iter_99999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/checkpoint_epoch_0_iter_99999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/checkpoint_epoch_0_iter_99999.pt) |
| 150k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/checkpoint_epoch_0_iter_149999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/checkpoint_epoch_0_iter_149999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/checkpoint_epoch_0_iter_149999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/checkpoint_epoch_0_iter_149999.pt) |
| 200k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/checkpoint_epoch_0_iter_199999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/checkpoint_epoch_0_iter_199999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/checkpoint_epoch_0_iter_199999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/checkpoint_epoch_0_iter_199999.pt) |
| 250k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/checkpoint_epoch_0_iter_249999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/checkpoint_epoch_0_iter_249999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/checkpoint_epoch_0_iter_249999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/checkpoint_epoch_0_iter_249999.pt) |
| 300k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/checkpoint_epoch_0_iter_299999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/checkpoint_epoch_0_iter_299999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/checkpoint_epoch_0_iter_299999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/checkpoint_epoch_0_iter_299999.pt) |
| 330k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/checkpoint_epoch_0_iter_329999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/checkpoint_epoch_0_iter_329999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/checkpoint_epoch_0_iter_329999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/checkpoint_epoch_0_iter_329999.pt) |
| 335k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/checkpoint_epoch_0_iter_334999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/checkpoint_epoch_0_iter_334999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/checkpoint_epoch_0_iter_334999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/checkpoint_epoch_0_iter_334999.pt) |
| 340k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/checkpoint_epoch_0_iter_339999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/checkpoint_epoch_0_iter_339999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/checkpoint_epoch_0_iter_339999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/checkpoint_epoch_0_iter_339999.pt) |
| 345k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/checkpoint_epoch_0_iter_344999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/checkpoint_epoch_0_iter_344999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/checkpoint_epoch_0_iter_344999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/checkpoint_epoch_0_iter_344999.pt) |
| 350k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/checkpoint_epoch_0_iter_349999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/checkpoint_epoch_0_iter_349999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/checkpoint_epoch_0_iter_349999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/checkpoint_epoch_0_iter_349999.pt) |
| Avg. last 5 ckpts | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/checkpoint_average.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/checkpoint_average.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/checkpoint_average.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/checkpoint_average.pt) |
| Training logs | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/training_logs.txt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/training_logs.txt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/training_logs.txt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/training_logs.txt) |

### Full checkpoints

The following checkpoints contain training state information, such as model and optimizer states, which maybe useful in resuming the training.

| Iteration | OpenELM-270M | OpenELM-450M | OpenELM-1.1B | OpenELM-3B |
| ---- | ---- | ---- | ---- | ---- | 
| 50k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/training_checkpoint_epoch_0_iter_49999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/training_checkpoint_epoch_0_iter_49999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/training_checkpoint_epoch_0_iter_49999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/training_checkpoint_epoch_0_iter_49999.pt) |
| 100k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/training_checkpoint_epoch_0_iter_99999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/training_checkpoint_epoch_0_iter_99999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/training_checkpoint_epoch_0_iter_99999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/training_checkpoint_epoch_0_iter_99999.pt) |
| 150k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/training_checkpoint_epoch_0_iter_149999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/training_checkpoint_epoch_0_iter_149999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/training_checkpoint_epoch_0_iter_149999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/training_checkpoint_epoch_0_iter_149999.pt) |
| 200k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/training_checkpoint_epoch_0_iter_199999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/training_checkpoint_epoch_0_iter_199999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/training_checkpoint_epoch_0_iter_199999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/training_checkpoint_epoch_0_iter_199999.pt) |
| 250k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/training_checkpoint_epoch_0_iter_249999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/training_checkpoint_epoch_0_iter_249999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/training_checkpoint_epoch_0_iter_249999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/training_checkpoint_epoch_0_iter_249999.pt) |
| 300k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/training_checkpoint_epoch_0_iter_299999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/training_checkpoint_epoch_0_iter_299999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/training_checkpoint_epoch_0_iter_299999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/training_checkpoint_epoch_0_iter_299999.pt) |
| 330k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/training_checkpoint_epoch_0_iter_329999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/training_checkpoint_epoch_0_iter_329999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/training_checkpoint_epoch_0_iter_329999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/training_checkpoint_epoch_0_iter_329999.pt) |
| 335k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/training_checkpoint_epoch_0_iter_334999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/training_checkpoint_epoch_0_iter_334999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/training_checkpoint_epoch_0_iter_334999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/training_checkpoint_epoch_0_iter_334999.pt) |
| 340k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/training_checkpoint_epoch_0_iter_339999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/training_checkpoint_epoch_0_iter_339999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/training_checkpoint_epoch_0_iter_339999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/training_checkpoint_epoch_0_iter_339999.pt) |
| 345k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/training_checkpoint_epoch_0_iter_344999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/training_checkpoint_epoch_0_iter_344999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/training_checkpoint_epoch_0_iter_344999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/training_checkpoint_epoch_0_iter_344999.pt) |
| 350k | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/training_checkpoint_epoch_0_iter_349999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/training_checkpoint_epoch_0_iter_349999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/training_checkpoint_epoch_0_iter_349999.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/training_checkpoint_epoch_0_iter_349999.pt) |
| End of training| [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/training_checkpoint_last.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/450M/training_checkpoint_last.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/1.1B/training_checkpoint_last.pt) | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/3B/training_checkpoint_last.pt) |

