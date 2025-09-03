#!/usr/bin/env bash
set -ex

#--------------------------------------------------
# Prune a model using a YAML recipe
# Results will be written to: out/pruning-YYYYMMDD-HHMMSS
#--------------------------------------------------

######################
# Task Parameters
######################

# Base model (HF hub id or local path)
MODEL="Qwen/Qwen3-0.6B"

# Calibration dataset
DATASET="Salesforce/wikitext"
DATA_FIELD="text"
DATA_SPLIT="train"
DATA_DIR="wikitext-103-raw-v1"
NUM_CAL_SAMPLES=1024

# YAML pruning recipe
RECIPE="./alps_24_ignore_attn.yaml"

# Whether to store compressed tensors (True/False)
SAVE_COMPRESSED="False"

# Output directory (timestamped)
OUTPUT_DIR="out/pruning-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

######################
# Run pruning script
######################
python main_pruning.py \
   --model "$MODEL" \
   --output_dir "$OUTPUT_DIR" \
   --dataset "$DATASET" \
   --data_field "$DATA_FIELD" \
   --data_split "$DATA_SPLIT" \
   --data_dir "$DATA_DIR" \
   --num_calibration_samples "$NUM_CAL_SAMPLES" \
   --save_compressed "$SAVE_COMPRESSED" \
   --pruning_yaml_recipe "$RECIPE"
