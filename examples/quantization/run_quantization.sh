#!/usr/bin/env bash
set -ex


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

# Output directory (timestamped)
OUTPUT_DIR="out/quantization-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Quantization recipe
RECIPE="./recipes/w4a16_int.yaml"

python main_quantization.py \
   --model $MODEL \
   --output_dir $OUTPUT_DIR \
   --dataset $DATASET \
   --data_field $DATA_FIELD \
   --data_split $DATA_SPLIT \
   --data_dir $DATA_DIR \
   --num_calibration_samples $NUM_CAL_SAMPLES \
   --quantization_recipe $RECIPE
