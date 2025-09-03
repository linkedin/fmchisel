# Structured Pruning LLMs with FMCHISEL

This example shows how to **compress a Large Language Model (LLM) by removing entire neural structures**—hidden MLP neurons and attention-head groups—using FMCHISEL’s structured-pruning algorithm **OSSCAR**. The workflow covers data loading, model preparation, pruning, and exporting the compressed model back to the Hugging Face format.

## 1. Background

Structured pruning differs from unstructured sparsity in that whole blocks of computation are removed:

* **MLP-neuron removal** – shrinks the feed-forward hidden dimension (`intermediate_size`).
* **Attention-head removal** – drops groups of key/value heads and their matching query heads, reducing `num_key_value_heads` and `num_attention_heads`.

Because full structures are deleted, the pruned model runs with **standard Transformer kernels**—no custom sparse ops required. See the [OSSCAR paper](https://arxiv.org/pdf/2403.12983) for algorithmic details.


## 2. Getting Started

```bash
# 1. (Optional) login to HF if models / datasets are gated
huggingface-cli login

# 2. Run structured pruning with default hyper-parameters
bash run.sh
```

The run.sh script will:

1. Download the **base model** (default: `Qwen/Qwen3-0.6B`).
2. Load **C4-en** calibration samples (default: `1024`).
3. Remove **128 MLP neurons** and **1 KV-head group** from *each* Transformer block (uniform pruning).
4. Save the compressed model to `out/structured_pruning-<ID>/` (HF-compatible).
5. Optionally save an even smaller on-disk representation if `--save_compressed True` is used.

### Output Artifacts

```
out/structured_pruning-<ID>/
└── [HF model files]         # weights + config.json + tokenizer
```

If `--save_compressed True` is enabled, an additional **compressed** directory containing dense weights with pruned dimensions removed is stored alongside the standard HF files.


## 3. Customizations

All flags from `StructuredPruningConfig` and `CalibrationDataConfig` can be overridden on the CLI.

| Flag | Description | Example |
|------|-------------|---------|
| `--model` | Path / HF-hub id of the model to prune. | `meta-llama/Llama-3.1-8B` |
| `--output_dir` | Directory for the pruned model. | `out/structured_pruning-myllama` |
| `--dataset` | HF dataset for calibration. | `allenai/c4` |
| `--data_field` | Text column in the dataset. | `text` |
| `--num_calibration_samples` | #Samples for calibration. | `2048` |
| `--num_drop_mlp_neuron` | Hidden neurons removed **per block**. | `256` |
| `--num_drop_attn_group` | KV-head groups removed **per block**. | `2` |
| `--save_compressed` | Store stripped-down tensors. | `True` |
| `--model_max_length` | Max sequence length during calibration. | `4096` |


**Accuracy vs. size** – Balance `--num_drop_mlp_neuron` and `--num_drop_attn_group` to reach your target footprint.
