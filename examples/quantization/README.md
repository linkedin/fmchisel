# Quantizing LLMs with FMCHISEL

This example shows how to **quantize a Large Language Model (LLM) into a low precision, memory- and compute-efficient variant** using FMCHISEL. It demonstrates data loading, model preparation and quantization, and exporting the quantizing the model into the `compressed-tensors` format.

## 1. Background

* **Weight-only quantization** – Only model weights are quantized to low precision (e.g., 4 bits), while activations are kept in 16 bits..
* **Weight and activation quantization** – Both weights and activations are quantized to lower precisions.

FMCHISEL implements **QuantEase**. See the [QuantEase paper](https://arxiv.org/abs/2309.01885) for technical details.

---

## 2. Getting Started

```bash
# 1. (Optional) login to HF if models / datasets are gated
huggingface-cli login

# 2. Quantize via passing YAML recipes
bash run_quantization.sh
```

The run.sh script will:

1. Download the **base model** (default: `Qwen/Qwen3-0.6B`).
2. Load **C4-en** calibration samples (default: `1024`).
3. Apply **QuantEase** quantization (4 bits weight-only quantization).
4. Save the quantized model to `out/quantization-<ID>/` (compressed-tensors-compatible).

### Output Artifacts

```
out/quantization-<ID>/
└── [model files]         # weights + config.json + tokenizer
```


## 3. Customizations


| Flag | Description | Example |
|------|-------------|---------|
| `--model` | Path / HF-hub id of the model to quantize. | `meta-llama/Llama-3.1-8B` |
| `--output_dir` | Where to write the quantized model. | `output_model/quantization/my_llama` |
| `--dataset` | HF dataset used for calibration. | `allenai/c4` |
| `--data_field` | Text column inside the dataset. | `text` |
| `--num_calibration_samples` | #Samples used for calibration. | `2048` |
| `--quantization_recipe` | The path to the recipe for quantization | '/my_recipe.yaml' |
| `--model_max_length` | Max sequence length during calibration. | `4096` |

### YAML Recipes

Complex sparsity patterns are easiest to express via a YAML file. We follow the same recipe patterns as `llmcompressor`. These recipes allow for customization of the quantization scheme (number of bits, grouping, activation ordering, etc).

```yaml
# w4a16_int.yaml
quantization_stage:
  run_type: oneshot
  quantization_modifiers:
    QuantEaseModifier:  
      dampening_frac: 0.01
      ignore: ["re:.*lm_head"]
      num_iter: 5                    # Number of QuantEase iterations
      config_groups:                 # Quantization config
        group_0:
          targets:
            - "Linear"
          input_activations: null    # No activation quantization
          output_activations: null
          weights:
            num_bits: 4              # 4-bit weight quantization
            type: "int"              # int or float
            symmetric: true
            strategy: "group"        # group-level quantization
            group_size: 128          # group size
```



## 4. Tips & Tricks

1. **Speed** – Use serving engines such as vLLM for end-to-end speed ups of quantized models.
2. **Layer dropping** – Excluding attention heads (via `ignore` regexes) often preserves accuracy.