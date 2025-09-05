<!-- PROJECT TITLE & BADGES -->
<h1 align="center">FMCHISEL – Efficient Foundation Model Algorithms</h1>

<p align="center">
  <b>State-of-the-art compression & distillation recipes for Large Language Models</b><br/>
</p>

---

## ✨ Overview

FMCHISEL (_Foundation&nbsp;Model&nbsp;Chisel_) is an **open-source research library** that makes it simple to:

* **Compress** LLMs with cutting-edge pruning and quantization techniques.
* **Distill** knowledge from larger models to smaller ones.
* **Accelerate** inference on consumer hardware by combining sparse + low-bit weight formats.
* **Train** efficiently with advanced optimizers such as schedule-free **AdamW**.
* **Prototype** new compression ideas rapidly.

FMCHISEL is built on **PyTorch** and integrates seamlessly with 📚 **🤗 Transformers**.

---

## 📦 Installation

Install from source. Linux is required (enforced by setup). Installing on macOS or Windows will fail at setup time:

```bash
# Clone the repo
git clone https://github.com/linkedin/FMCHISEL.git
cd fmchisel

# Base install
pip install -e .

# Optional extras
# - inference: pruning/quantization via llmcompressor
# - train: distillation (Lightning, liger-kernel)
# - all: both of the above
pip install -e ".[inference]"
pip install -e ".[train]"
# or
pip install -e ".[all]"
```

---

## 🚀 Quick Start

Ready-to-run recipes in `examples/`:

- Distillation: `bash examples/distillation/run.sh`
- Unstructured or N:M pruning (ALPS, SparseGPT, Wanda): `bash examples/pruning/run.sh`
- Structured pruning (OSSCAR): `bash examples/structured_pruning/run.sh`
- Quantization (QuantEase via YAML recipes): `bash examples/quantization/run_quantization.sh`

Tweak the scripts or pass flags to adjust models, datasets, and hyper-parameters.

---

## 🗂️ Project Structure

```
fmchisel/
│
├─ data/               # Calibration & data utilities
├─ distillation/       # Knowledge-distillation components
├─ pruning/            # ALPS + OSSCAR implementations; SparseGPT/Wanda via llmcompressor
├─ quantization/       # QuantEase & helpers
├─ optimizers/         # AdamW schedule-free implementation
├─ utils/              # Callbacks, training helpers
└─ config.py           # Global configuration
examples/              # End-to-end reproducible recipes
tests/                 # PyTest suite
```

---

## 🧪 Research Components

| Area             | Algorithm(s)                  | Implementation Module |
|------------------|-------------------------------|-----------------------|
| **Pruning**      | ALPS (unstructured, N:M)      | `fmchisel.pruning.alps` |
| **Structured**   | OSSCAR (MLP/attn-group drop)  | `fmchisel.pruning.osscar` |
| **Quantization** | QuantEase (weight-only/group) | `fmchisel.quantization.quantease` |
| **Distillation** | Per-token KD (e.g., JSD)      | `fmchisel.distillation.losses` |
| **Optimization** | AdamW Schedule-Free           | `fmchisel.optimizers.adamw_schedulefree` |

Notes:
- SparseGPT and Wanda pruning are available through `llmcompressor` and wired up in `examples/pruning/pruning_utils.py`.
- Quantization uses `llmcompressor` pipelines with a QuantEase modifier and YAML recipes.
 - To combine pruning and quantization, compose both modifiers in a single YAML recipe and pass it to `llmcompressor.oneshot`. See `llmcompressor` documentation for composing modifiers. Example composite recipes are not included in this repo.

### Minimal Python usage (grounded in the repo)

Pruning (ALPS or SparseGPT/Wanda) via `oneshot` and `HFCalibrationDataLoader`:

```python
from llmcompressor import oneshot
from transformers import AutoTokenizer
from fmchisel.data.calibration_datautil import HFCalibrationDataLoader
from fmchisel.pruning.alps.base import ALPSModifier

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = HFCalibrationDataLoader(
    nsamples=1024,
    tokenizer=tokenizer,
    max_seq_length=tokenizer.model_max_length,
    dataset="allenai/c4",
    data_field="text",
    data_dir="en",
    data_split="train",
).get_tokenized_calibration()

recipe = ALPSModifier(sparsity=0.5, mask_structure="2:4", targets="__ALL_PRUNABLE__")
oneshot(model=model_id, dataset=dataset, recipe=recipe, output_dir="out/pruned")
```

Structured pruning (OSSCAR):

```python
from llmcompressor import oneshot
from transformers import AutoTokenizer
from fmchisel.data.calibration_datautil import HFCalibrationDataLoader
from fmchisel.pruning.osscar.base import OSSCARModifier

model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
dataset = HFCalibrationDataLoader(
    nsamples=1024,
    tokenizer=tokenizer,
    max_seq_length=tokenizer.model_max_length,
    dataset="allenai/c4",
    data_field="text",
    data_dir="en",
    data_split="train",
).get_tokenized_calibration()

recipe = OSSCARModifier(num_drop_mlp_neuron=128, num_drop_attn_group=1)
oneshot(model=model_id, dataset=dataset, recipe=recipe, output_dir="out/structured")
```

Quantization (QuantEase) is driven by YAML recipes (see `examples/quantization/recipes/*`):

```bash
bash examples/quantization/run_quantization.sh
```

Distillation with JSD loss (Lightning + FSDP):

```bash
bash examples/distillation/run.sh
```

---

## 🛠️ Contributing

1. Fork & clone the repository.  
2. Install dev deps: `pip install -e ".[dev]"`  (note: A Linux system is required.)
3. Run linters/formatters: `make checkstyle`.  
4. Execute tests: `make test`.  
5. Open a pull request!

> [!NOTE]
> Please open an issue first to discuss major changes.

---

## 🔒 License

See [LICENSE](LICENSE) for details.

## 📝 Citation
```
@software{behdin2025,
  author       = {Behdin, Kayhan and Fatahibaarzi, Ata and Yun, Dai and 
                  Song, Qingquan and Kothapalli, Vignesh and Tang, Shao and 
                  Sang, Hejian and Gupta, Aman and Wang, Zhipeng and 
                  Dexter, Gregory and Zhu, Sirou and Zhu, Siyu},
  title        = {FMCHISEL},
  year         = {2025},
}

```
### Additional references
This library implements compression methods from the following papers:
```
@article{meng2024alps,
  title={Alps: Improved optimization for highly sparse one-shot pruning for large language models},
  author={Meng, Xiang and Behdin, Kayhan and Wang, Haoyue and Mazumder, Rahul},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={37594--37625},
  year={2024}
}
```
```
@inproceedings{mengosscar,
  title={OSSCAR: One-Shot Structured Pruning in Vision and Language Models with Combinatorial Optimization},
  author={Meng, Xiang and Ibrahim, Shibal and Behdin, Kayhan and Hazimeh, Hussein and Ponomareva, Natalia and Mazumder, Rahul},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
```
@article{behdin2023quantease,
  title={QuantEase: Optimization-based quantization for language models},
  author={Behdin, Kayhan and Acharya, Ayan and Gupta, Aman and Song, Qingquan and Zhu, Siyu and Keerthi, Sathiya and Mazumder, Rahul},
  journal={arXiv preprint arXiv:2309.01885},
  year={2023}
}
```
