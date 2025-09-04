
# Efficient Model Compression & Evaluation

A complete pipeline for compressing, selecting, and evaluating Large Language Models (LLMs) using binary diffs (BinaryDiff), with automatic per-layer row/col variant selection. The project is structured as a professional Python package with CLI, configs, and tests.

## Features
- Load HuggingFace models and tokenizers
- Cache reference logits from a finetuned model
- Compress with BinaryDiffRow/Col
- Automatically choose row/col per layer based on validation loss
- Train the compressed model to match finetuned logits
- Save/load/merge delta files
- Flexible YAML configuration
- Easy to extend and integrate

## Project structure

```
├── src/bitdelta_pipeline/   # Library (modules: args, models, data, binary_variants, io_capture, choose_compress, ...)
├── scripts/                 # CLI scripts for pipeline and choose-compress
├── configs/                 # YAML configurations
├── tests/                   # Tests
├── requirements.txt         # Dependencies
├── pyproject.toml           # Packaging
├── README.md                # This file
└── .gitignore
```

## Installation & setup

### 1) Clone and prepare environment
```powershell
git clone https://anonymous.4open.science/r/Per-Axis-Weight-Deltas-for-Frequent-Model-Updates-0F1C.git
cd Per-Axis-Weight-Deltas-for-Frequent-Model-Updates-0F1C
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### 2) HuggingFace login
```powershell
huggingface-cli login
```

### 3) Configure
Edit `configs/default.yaml` to match your models, devices, and parameters.

## Run
### Automatic row/col selection per layer
```powershell
python -m scripts.run_choose_compress --config configs/default.yaml
```

## Example config (configs/default.yaml)
```yaml
base_model: "meta-llama/Llama-3.1-8B"
finetuned_model: "meta-llama/Llama-3.1-8B-Instruct"
base_model_device: "auto"
finetuned_model_device: "cuda:0"
finetuned_compressed_model_device: "cuda:0"
split_memory_map:
	1: "24GiB"
	0: "20GiB"
max_length: 128
batch_size: 1
num_steps: 800
lr: 0.0005
save_dir: "output"
debug: true
```

## Requirements
- Python 3.10+
- CUDA GPU (for acceleration)
- Access to HuggingFace models (e.g., Llama)
- See requirements.txt for all dependencies

## Notes & tips
- Don’t commit large model weights to the repo—use HF Hub or release artifacts
- If BitDelta isn’t installed, some features will be skipped with a warning
- Consider adding a GitHub Actions workflow for tests/CI
