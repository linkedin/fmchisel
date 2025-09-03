# Pruning LLMs with FMCHISEL

This example shows how to **prune a Large Language Model (LLM) into a sparse, memory- and compute-efficient variant** using FMCHISEL. It demonstrates data loading, model preparation, unstructured & N:M pruning, and exporting the pruned model back to the Hugging Face Hub format.

## 1. Background

* **Unstructured sparsity** – Any weight can be zero. Offers maximum flexibility but requires special kernels for runtime speed-ups.
* **Semi-structured (N:M) sparsity** – Exactly **N** non-zeros in every **M** consecutive weights (e.g. 2:4 = 50 % sparsity). Readily accelerated by NVIDIA A100/H100 tensor cores.

FMCHISEL implements **ALPS** (ADMM-based Layerwise Pruning with Saliency) and wraps two post-training methods from *llmcompressor*: **SparseGPT** and **Wanda**. See the [ALPS paper](https://arxiv.org/abs/2406.07831) for technical details.

---

## 2. Getting Started

```bash
# 1. (Optional) login to HF if models / datasets are gated
huggingface-cli login

# 2a. Prune via direct CLI arguments
bash run.sh

# 2b. Prune via a YAML recipe
bash run_recipe.sh
```

The run.sh script will:

1. Download the **base model** (default: `Qwen/Qwen3-0.6B`).
2. Load **C4-en** calibration samples (default: `1024`).
3. Apply **ALPS 2:4** pruning on all linear MLP layers while keeping attention layers dense.
4. Save the pruned model to `out/pruning-<ID>/` (HF-compatible).
5. Optionally store compressed tensor formats if `--save_compressed True` is used.

### Output Artifacts

```
out/pruning-<ID>/
└── [HF model files]         # weights + config.json + tokenizer
```

If `--save_compressed True` is enabled, an additional **compressed** directory containing CSR tensors & metadata will be created alongside the standard HF files.

## 3. Customizations

All flags from `PruningConfig` and `CalibrationDataConfig` can be overridden on the CLI or via a **YAML recipe**.

| Flag | Description | Example |
|------|-------------|---------|
| `--model` | Path / HF-hub id of the model to prune. | `meta-llama/Llama-3.1-8B` |
| `--output_dir` | Where to write the pruned model. | `output_model/pruning/my_llama` |
| `--dataset` | HF dataset used for calibration. | `allenai/c4` |
| `--data_field` | Text column inside the dataset. | `text` |
| `--num_calibration_samples` | #Samples used for calibration. | `2048` |
| `--pruning_strategy` | `ALPS`, `SparseGPT`, or `wanda`. | `ALPS` |
| `--sparsity` | Global sparsity ratio for unstructured pruning. | `0.5` |
| `--prunen / --prunem` | N and M for N:M sparsity. Use `0 0` to disable. | `2 4` |
| `--pruning_yaml_recipe` | Path to a YAML pruning recipe (overrides other pruning flags). | `examples/pruning/alps_24_ignore_attn.yaml` |
| `--save_compressed` | Store compressed tensors (CSR & metadata). | `True` |
| `--model_max_length` | Max sequence length during calibration. | `4096` |

### YAML Recipes

Complex sparsity patterns are easiest to express via a YAML file.

```yaml
# alps_24_ignore_attn.yaml
sparsity_stage:
  sparsity_modifiers:
    ALPSModifier:
      sparsity: 0.5          # 50 % overall
      mask_structure: "2:4"  # N:M pattern
      targets: ["Linear"]    # prune all Linear layers
      ignore: [              # keep attention dense
        "re:.*q_proj", "re:.*k_proj", "re:.*v_proj", "re:.*o_proj", "re:.*lm_head"
      ]
```

Pass it via `--pruning_yaml_recipe path/to/file.yaml` (see `run_recipe.sh`).


## 4. Tips & Tricks

1. **Speed** – Use N:M sparsity (e.g. 2:4) on Ampere/Hopper GPUs for actual inference acceleration.
2. **Layer dropping** – Excluding attention heads (via `ignore` regexes) often preserves accuracy.
3. **Model size** – Enable `--save_compressed True` to store the model in a compact CSR format.
