# OpenELM Parameter-Efficient Finetuning (PEFT)

We fine-tune models using the evaluation setup described in [LLM Adapters](https://arxiv.org/abs/2304.01933). This involves jointly fine-tuning on 8 commonsense reasoning datasets with a training set of size 170k. We follow the evaluation setup of [the official code release of LLM Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters), with the exception that we use log-likelihood rather than regex parsing to determine the model's output.

## Setup

To ensure consistency of evaluations with LLM Adapters, we use helper functions defined in their code. To set up for evaluations, run the following command:

```bash
# Change this to the path to CoreNet
cd /path/to/corenet

# Install LM Harness.
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout 3196e907fa195b684470a913c7235ed7f08a4383
python3 -m pip install -e . -c ../requirements.txt
cd ..

# Install LLM Adapters.
git clone https://github.com/AGI-Edgerunners/LLM-Adapters.git
cd LLM-Adapters
git checkout 816657208af4db747803f87ba40a4c71383fed7a
touch __init__.py
python3 -m pip install -r requirements.txt -c ../requirements.txt
cd ..

# Install Huggingface and its dependencies.
python3 -m pip install --upgrade \
    transformers==4.36.2 \
    datasets==2.19.0 \
    accelerate==0.29.3 \
    sentencepiece==0.2.0 \
    -c requirements.txt
```

In our experiments, we used LLamav1/v2 tokenizer. Please download the tokenizer from the [official repository](https://github.com/meta-llama/llama).

## Training

To fine-tune a 270M-parameter model with [LoRA](https://arxiv.org/abs/2106.09685), use the following command:

```bash
CFG_FILE="projects/openelm/peft_configs/openelm_lora_270M.yaml"
WTS_FILE="https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/pretrained/270M/checkpoint_average.pt"
TOKENIZER_FILE="<PATH_TO_TOKENIZER_FILE>"
# NOTE: The dataset can currently be obtained from https://github.com/AGI-Edgerunners/LLM-Adapters/blob/main/ft-training_set/commonsense_170k.json.
DATASET_FILE="<PATH_TO_COMMONSENSE_170K>"
corenet-train --common.config-file $CFG_FILE \
    --model.language-modeling.pretrained $WTS_FILE \
    --text-tokenizer.sentence-piece.model-path $TOKENIZER_FILE \
    --dataset.language-modeling.commonsense-170k.path $DATASET_FILE
```

To train with [DoRA](https://arxiv.org/abs/2402.09353) instead, edit the config file to set `use_dora` to True.

## Evaluation
To evaluate a pre-trained LoRA 270M model, use the following command:

```bash
CFG_FILE="projects/openelm/peft_configs/openelm_lora_270M_eval.yaml"
WTS_FILE="https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/peft/openelm_lora_270M.pt"
TOKENIZER_FILE="<PATH_TO_TOKENIZER_FILE>"
corenet-eval-llmadapters --common.config-file $CFG_FILE \
    --model.language-modeling.pretrained $WTS_FILE \
    --text-tokenizer.sentence-piece.model-path $TOKENIZER_FILE
```

The expected results are:

| boolq | piqa | siqa | hellaswag | winogrande | arc-easy | arc-challenge | obqa |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 62.14 | 50.05 | 42.02 | 24.84 | 49.88 | 26.60 | 24.57 | 28.00 |

To evaluate other pretrained models, edit the config file to use different backbones. To evaluate [DoRA](https://arxiv.org/abs/2402.09353) models, edit the config file to set `use_dora` to True.

### Pretraining checkpoints
| Model | LoRA/DoRA | Weights |
| ---- | ---- | ---- |
| OpenELM-270M | LoRA | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/peft/openelm_lora_270M.pt)
| OpenELM-450M | LoRA | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/peft/openelm_lora_450M.pt)
| OpenELM-1.1B | LoRA | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/peft/openelm_lora_1.1B.pt)
| OpenELM-3B | LoRA | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/peft/openelm_lora_3B.pt)
| OpenELM-270M | DoRA | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/peft/openelm_dora_270M.pt)
| OpenELM-450M | DoRA | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/peft/openelm_dora_450M.pt)
| OpenELM-1.1B | DoRA | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/peft/openelm_dora_1.1B.pt)
| OpenELM-3B | DoRA | [Link](https://docs-assets.developer.apple.com/ml-research/models/corenet/v0.1.0/openelm/peft/openelm_dora_3B.pt)
