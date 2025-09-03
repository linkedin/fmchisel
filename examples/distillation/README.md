# Distilling LLMs with FMCHISEL

The example scripts demonstrate how to **distill a large teacher model into a smaller, faster, and cheaper student model** via knowledge-distillation using FMCHISEL. It is intended to cover data loading, teacher–student setup, training with mixed-precision + FSDP, and exporting the distilled model back to the Hugging Face Hub format.


## Getting Started

Make sure you have installed `fmchisel` and its dependencies before running the scripts.

```bash
# 1. (Optional) Login to HF if model(s) are gated
huggingface-cli login

# 2. Launch the example with default hyper-parameters
bash run.sh
```

The script will:

1. Download the **teacher** model (default: `meta-llama/Llama-3.1-8B`) and the **student** model (default: `meta-llama/Llama-3.2-1B`).
2. Load the *CNN-DailyMail* dataset from Hugging Face Datasets.
3. Train the student with **Jensen–Shannon Divergence** loss (`--distill_loss js`).
4. Convert the best checkpoint to the HF format at `out/distillation-<ID>/best_model/`.
5. Save the training logs in `out/distillation-<ID>/lightning_logs/`.
6. Save the intermediate PyTorch Lightning checkpoints in `out/distillation-<ID>/ckpts/` and the fully trained model checkpoint as `out/distillation-<ID>/ckpts/model.ckpt`.

### Output Artifacts

```text
out/distillation-<ID>/
├── lightning_logs/         # Log/metric files
├── best_model/             # HF compatible weights + config.json + tokenizer
├── ckpts/                  # Pytorch lightning checkpoint(s)
```

Convert checkpoints to HF is handled automatically by `training.py` through `fmchisel.utils.consolidate_ckpt_to_hf`.

## Customizations

You can override any parameter in `run.sh` as defined in [`TrainingArgs`](../../src/fmchisel/config.py) or [`DistillTrainingConfig`](../../src/fmchisel/distillation/config.py).


| Flag | Description | Example |
|------|-------------|---------|
| `--model_path` | Path to the student model (default: `meta-llama/Llama-3.2-1B`). | `meta-llama/Llama-3.2-1B` |
| `--teacher_model_path` | Path to the teacher model (default: `meta-llama/Llama-3.1-8B`). | `meta-llama/Llama-3.1-8B` |
| `--dataset` | Name of the dataset to use for distillation (default: `cnn_dailymail`). | `cnn_dailymail` |
| `--batch_size` | Batch size for training (default: `2`). | `2` |
| `--num_epoch` | Number of training epochs (default: `3`). | `3` |
| `--enable_distill` | Enable distillation (default: `True`). | `True` |
| `--distillation_loss_ratio` | Ratio of the distillation loss to the total loss (default: `0.5`). | `0.7` |
| `--teacher_forward_dtype` | Data type for the teacher model forward pass (default: `fp32`). Options include: `fp16`, `bfloat16`, `fp32`. | `fp16` |
| `--distill_loss_ratio` | Ratio of the distillation loss to the total loss (default: `0.5`). | `0.7` |
| `--distill_loss` | Select the distillation loss function. Options include: `forward_kl`, `combined_kl`, `reverse_kl`, `js`, `tvd`, `skl`, `srkl`, `fljsd`. | `js` |
| `--sample_method` | Switch between `"supervised"` (next-token) distillation and `"on-policy"` or `"sequence-level"` sampling-based methods. | `"supervised"` |
| `--forward_ratio` | For *KL* or *R-D* style losses, controls what fraction of the student forward pass uses teacher logits. Set to `-1` for JS divergence. | `0.5` |
| `--use_lora` / `--lora_rank` | Activate **LoRA** adapters to speed-up fine-tuning. | `--use_lora True --lora_rank 16` |
| `--keep_sparse` | Apply **dynamic sparsity** training to the MLP-projections (great when combined with pruning recipes). | `True` |
| `--use_liger` | Enable **Liger** optimizations for faster attention. | `True` |


## Tips & Tricks

1. **GPU Memory** – if you hit OOM, lower `--batch_size` or enable `--cpu_offload`.
2. **Teacher in FP16** – you may cast the teacher model to half precision to save memory:
   `--teacher_forward_dtype fp16`.
3. **Dataset Sub-Sampling** – Use `--n_train` / `--n_val` for quick debugging.
