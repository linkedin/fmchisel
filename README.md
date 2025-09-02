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

The package will soon be available on PyPI. Until then, install from source (note that a Linux system will be required):

```bash
# Clone the repo
git clone https://github.com/linkedin/FMCHISEL.git
cd fmchisel

# Install with all optional dependencies
pip install -e ".[all]"
```

---

## 🚀 Quick Start

Start with these ready-to-run recipes from the `examples/` folder:

* **Distillation:** `examples/distillation/run.sh` – distill a 8B teacher into a 1B student model
* **Unstructured Pruning:** `examples/pruning/run.sh` – N:M semi-structured pruning in one shot  
* **Structured Pruning:** `examples/structured_pruning/run.sh` – remove entire heads / neurons  

Simply run a script (or tweak hyper-parameters) and you’re good to go!

---

## 🗂️ Project Structure

```
fmchisel/
│
├─ data/               # Calibration & data utilities
├─ distillation/       # Knowledge-distillation components
├─ pruning/            # ALPS, OSSCAR, SparseGPT, structured pruning
├─ quantization/       # QuantEase & helpers
├─ optimizers/         # AdamW schedule-free implementation
├─ utils/              # Callbacks, training helpers
└─ config.py           # Global configuration
examples/              # End-to-end reproducible recipes
tests/                 # PyTest suite
```

---

## 🧪 Research Components

| Area            | Algorithm(s) | Implementation |
|-----------------|-------------------|----------------|
| **Pruning**     | ALPS | `fmchisel.pruning.alps` |
|                 | OSSCAR | `fmchisel.pruning.osscar` |
| **Quantization**| QuantEase | `fmchisel.quantization.quantease` |
| **Distillation**| Lightweight KD recipes | `fmchisel.distillation` |
| **Optimization**| AdamW Schedule-Free | `fmchisel.optimizers.adamw_schedulefree` |

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
