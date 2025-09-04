
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

# Efficient Model Compression & Evaluation (1-bit Delta with Vector Scales)

Serving many task-specialized LLM variants is often limited by the large size of fine-tuned checkpoints and the resulting cold-start latency. Since fine-tuned weights differ from their base model by relatively small structured residuals, a natural approach is to represent them as compressed deltas. We propose a simple 1-bit delta scheme that stores only the sign of the weight difference together with lightweight per-axis (row/column) FP16 scaling factors, learned from a small calibration set. This design preserves the compactness of 1-bit deltas while more accurately capturing variation across weight dimensions, leading to improved reconstruction quality over scalar alternatives. From a systems perspective, a streamlined loader that transfers packed deltas in a single operation per module reduces cold-start latency and storage overhead, with artifacts several times smaller than a full FP16 checkpoint.

## Results (Zero-shot Accuracy, %)

After calibrating on 150 samples from C4. Vector scales are trained for five epochs with learning rate 1e-5; BitDelta uses the same setup with a single scalar per matrix.

| Model              | ARC-C  | ARC-E  | HellaSwag | PIQA   | Winogrande | Avg   |
|--------------------|--------|--------|-----------|--------|------------|-------|
| Baseline           | 51.70  | 81.81  | 59.06     | 79.86  | 73.87      | 69.26 |
| BitDelta (scalar)  | 52.55  | 82.32  | 59.73     | 81.22  | 73.95      | 69.95 |
| Vector (row/col)   | 53.58  | 82.99  | 59.78     | 80.63  | 74.19      | 70.23 |

## Project Structure

- `src/bitdelta_pipeline/` — core library modules
	- `args.py` — configuration dataclass
	- `models.py` — tokenizer/model loading helpers
	- `data.py` — dataset and dataloader builders (C4 subsets)
	- `binary_variants.py` — BinaryDiffRow/Col modules and autograd helpers
	- `io_capture.py` — forward hooks to collect per-layer inputs/outputs
	- `choose_compress.py` — per-layer row/col selection and fitting
	- `compress.py` — baseline BitDelta compression by module
	- `eval_utils.py` — logits caching and evaluation utilities
	- `save_load_delta.py` — save/load/merge delta artifacts
	- `weights.py` — load specific weights from local HF cache
- `scripts/` — command-line entry points
	- `run_pipeline.py` — end-to-end compression + training
	- `run_choose_compress.py` — row/col selection per layer using calibration I/O
- `configs/` — YAML configuration files
	- `default.yaml` — default configuration
- `tests/` — smoke tests
- `requirements.txt`, `pyproject.toml`, `.gitignore`

## Installation & Usage (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install -e .

# (Optional) Hugging Face login for gated models
huggingface-cli login
```

Run the main pipeline:
```powershell
python -m scripts.run_pipeline --config configs/default.yaml
```

Run per-layer row/col selection:
```powershell
python -m scripts.run_choose_compress --config configs/default.yaml
```

## Attribution and Adaptation

This repository is primarily adapted from the BitDelta project (FasterDecoding\
BitDelta). We extended and reorganized components to:
- Add vector (row/col) scaling variants alongside the scalar BitDelta baseline
- Implement per-layer I/O capture and automatic row/col selection
- Provide utilities to save/load/merge vector-delta artifacts
- Offer a clean CLI, configuration, and modular pipeline

## Experiment-specific Files (this repo)

- Vector delta modules and helpers:
	- `src/bitdelta_pipeline/binary_variants.py`
	- `src/bitdelta_pipeline/save_load_delta.py`
- Per-layer I/O capture and chooser:
	- `src/bitdelta_pipeline/io_capture.py`
	- `src/bitdelta_pipeline/choose_compress.py`
- End-to-end scripts:
	- `scripts/run_pipeline.py`
	- `scripts/run_choose_compress.py`
- Notebook-derived utilities and weight loading:
	- `src/bitdelta_pipeline/weights.py`
