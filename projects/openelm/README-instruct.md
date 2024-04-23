# OpenELM Instruction Tuning

We use [The Alignment Handbook](https://github.com/huggingface/alignment-handbook.git) library to finetune OpenELM models on the [UltraFeedback dataset](https://huggingface.co/datasets/csarron/argilla-ultrafeedback-binarized-preferences-cleaned). Below is the instrution to produce the instruction-tuned OpenELM models.

```bash
# Change this to the path that you've cloned the repository
cd /path/to/corenet 

# Install dependencies.

git clone https://github.com/huggingface/alignment-handbook.git hf-align
# 70769f9 is the main branch on 2024-04-11.
cd hf-align && git checkout 70769f9 && cd .. 

pip install -e ./hf-align

# Copy openelm-instruct recipe to hf-align/recipes.
cp projects/openelm/instruction_tuning/openelm-instruct.yaml hf-align/recipes/

# Prepare the converted OpenELM Huggingface model to ckpt_dir.
ckpt_dir=<your_converted_OpenELM_hf_model>
# Prepare tokenizer.
local_tokenizer_dir=<your_downloaded_Llama-2-7b-hf-tokenizer>
# Set output checkpoint dir.
dpo_ckpt_dir=<your_output_checkpoint_dir>

# Set lr, epochs, and loss_type based on the paper, also see the table below.
# e.g. for OpenELM-270M model:
ep=5
lr=2e-5
loss_type=hinge

accelerate launch --config_file hf-align/recipes/accelerate_configs/deepspeed_zero3.yaml \
hf-align/scripts/run_dpo.py hf-align/recipes/openelm-instruct.yaml \
--trust_remote_code=true \
--model_name_or_path=${ckpt_dir} \
--tokenizer_name_or_path=${local_tokenizer_dir} \
--output_dir=${dpo_ckpt_dir} \
--num_train_epochs=$ep \
--learning_rate=$lr \
--loss_type=$loss_type

# Results will be in ${dpo_ckpt_dir}/all_results.json.

```

OpenELM instruction tuning hyperparameters:

| Hyperparameters                    | **270M**      | **450M**      | **1.1B**      | **3B**      |
|------------------------------------|:-------------:|:-------------:|:-------------:|:-----------:|
| Training epochs                    |       5       |       8       |       5       |      10     |
| Learning rate                      |      2e-5     |      3e-5     |      5e-5     |     1e-4    |
| Loss function                      |     hinge     |     hinge     |    sigmoid    |    hinge    |

