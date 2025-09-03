#!/bin/bash

set -ex

###################
# Environment Setup
###################

export ACCELERATE_USE_FSDP=1
export FSDP_CPU_RAM_EFFICIENT_LOADING=1

######################
# Task Parameters
######################

# Student model (from HuggingFace)
model_path="meta-llama/Llama-3.2-1B"

# Use HuggingFace dataset name and path
# For cnn_dailymail, the HF dataset loader uses just the name, default to be cnn_dailymail version 3.0.0
dataset="cnn_dailymail"
data_path=""  # Leave empty to use HF datasets library

# Teacher model (from HuggingFace)
teacher_model_path="meta-llama/Llama-3.1-8B"
optimizer="adamw"
lr=4e-6
keep_sparse=False # only used for sparse training of all mlp layers in model ("proj" or "weight" in layer name key)
enable_distill=True
distill_forward_ratio=-1.0  # not used for Jensen-Shannon divergence (js) distillation loss
distill_loss="js"
sample_method="supervised"
max_new_tokens=2  # not used
return_prompt_input_ids=False
use_lora=False
verify_lora_saving_correctness=False  # Set to True to verify that LoRA saving works correctly
use_liger=True  # Enable Liger-Kernel optimizations

# Common parameters
max_length=4096
batch_size=2
num_epoch=1
weight_decay=0.05
warmup_ratio=0.1
n_train=100
n_val=10
val_check_interval=5
distillation_loss_ratio=0.5

# Set the output directory for the training results
OUTPUT_DIR="out/distillation-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

######################
# Run training script
######################

python training.py \
    --model_path "$model_path" \
    --teacher_model_path "$teacher_model_path" \
    --dataset "$dataset" \
    --data_path "$data_path" \
    --max_length "$max_length" \
    --batch_size "$batch_size" \
    --output_dir "$OUTPUT_DIR" \
    --optimizer "$optimizer" \
    --lr "$lr" \
    --num_epoch "$num_epoch" \
    --weight_decay "$weight_decay" \
    --warmup_ratio "$warmup_ratio" \
    --n_train "$n_train" \
    --n_val "$n_val" \
    --val_check_interval "$val_check_interval" \
    --enable_distill "$enable_distill" \
    --forward_ratio "$distill_forward_ratio" \
    --distillation_loss_ratio "$distillation_loss_ratio" \
    --distill_loss "$distill_loss" \
    --sample_method "$sample_method" \
    --max_new_tokens "$max_new_tokens" \
    --return_prompt_input_ids "$return_prompt_input_ids" \
    --keep_sparse "$keep_sparse" \
    --use_lora "$use_lora" \
    --use_liger "$use_liger" \
    --verify_lora_saving_correctness "$verify_lora_saving_correctness"
