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

# Pruning settings
PRUNING_METHOD="ALPS"   # Choose from: ALPS, wanda, SparseGPT
SPARSITY=0.5            # Unstructured sparsity ratio
PRUNE_N=2               # N for N:M pattern
PRUNE_M=4               # M for N:M pattern
SAVE_COMPRESSED="False" # Store compressed tensors (True/False)

# Output directory (timestamped)
OUTPUT_DIR="out/pruning-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

######################
# Run pruning script
######################
python main_pruning.py \
   --model "$MODEL" \
   --output_dir "$OUTPUT_DIR" \
   --pruning_strategy "$PRUNING_METHOD" \
   --dataset "$DATASET" \
   --data_field "$DATA_FIELD" \
   --data_split "$DATA_SPLIT" \
   --data_dir "$DATA_DIR" \
   --num_calibration_samples "$NUM_CAL_SAMPLES" \
   --sparsity "$SPARSITY" \
   --prunen "$PRUNE_N" \
   --prunem "$PRUNE_M" \
   --save_compressed "$SAVE_COMPRESSED"
