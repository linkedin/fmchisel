#!/usr/bin/env bash
set -ex

# -------------------------------------------------
# Structured pruning with explicit CLI arguments
# Results saved to: out/structured_pruning-<timestamp>/
# -------------------------------------------------

######################
# Task Parameters
######################

# Base model (HF hub id or local path)
MODEL="Qwen/Qwen3-0.6B"

# Calibration dataset
DATASET="allenai/c4"
DATA_FIELD="text"
DATA_SPLIT="train"
DATA_DIR="en"
NUM_CAL_SAMPLES=1024

# Structured-pruning settings (OSSCAR)
NUM_DROP_MLP_NEURON=128   # neurons removed per transformer block
NUM_DROP_ATTN_GROUP=1     # KV-head groups removed per block
SAVE_COMPRESSED="True"    # store compressed model (True/False)

# Output directory (timestamped like distillation/pruning examples)
OUTPUT_DIR="out/structured_pruning-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

######################
# Run pruning script
######################
python main_structured_pruning.py \
   --model "$MODEL" \
   --output_dir "$OUTPUT_DIR" \
   --dataset "$DATASET" \
   --data_field "$DATA_FIELD" \
   --data_split "$DATA_SPLIT" \
   --data_dir "$DATA_DIR" \
   --num_calibration_samples "$NUM_CAL_SAMPLES" \
   --num_drop_mlp_neuron "$NUM_DROP_MLP_NEURON" \
   --num_drop_attn_group "$NUM_DROP_ATTN_GROUP" \
   --save_compressed "$SAVE_COMPRESSED"
