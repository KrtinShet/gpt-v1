# GPT-2 Minimal Language Model (Shakespeare Edition)

This project is a minimal implementation of a GPT-2 style language model, written in PyTorch, designed to train on Shakespearean text. It demonstrates the core mechanics of transformer-based language modeling, including a custom lightweight data loader and a simple training loop. The model and code are intended for educational purposes and experimentation with transformer architectures.

## Device & Hardware Requirements

- **Tested on:** MacBook Pro (M4 Pro, 24GB VRAM)
- **OS:** macOS (Darwin 24.5.0) 

## Getting Started

### 1. Prerequisites

- Python 3.10+
- [UV](https://github.com/astral-sh/uv) package manager (ensure it is installed and available in your PATH)

### 2. Installation

All dependencies are managed via `pyproject.toml`. To install them using UB:

```sh
ub pip install -r pyproject.toml
```

Or, if UB supports direct TOML installation:

```sh
uv sync
```

### 3. Prepare the Data

The project includes a sample Shakespeare text in `data/shakesphere/input.txt`. You can preprocess or replace this file as needed. (See `data/shakesphere/prepare.py` for any data preparation scripts.)

### 4. Train the Model

To start training, simply run:

```sh
ub python main.py
```

This will:

- Load and tokenize the Shakespeare text
- Initialize a GPT-2 style model
- Train the model on your Apple Silicon GPU (using Metal acceleration)

### 5. File Structure

- `main.py` — Main training script
- `gpt2.py` — GPT-2 model implementation
- `gpt_config.py` — Model configuration dataclass
- `data_loader.py` — Lightweight data loader for batching and tokenization
- `data/shakesphere/input.txt` — Sample training data (Shakespeare)
- `data/shakesphere/prepare.py` — (Optional) Data preparation script
- `pyproject.toml` — Project dependencies and metadata

## Notes

- The model is configured for demonstration and may require tuning for larger datasets or different tasks.
- Training time and memory usage will depend on your hardware. The provided MacBook Pro (M4 Pro, 24GB VRAM) is well-suited for this workload.

## Tribute

This project is inspired by [Andrej Karpathy's YouTube video on building GPT from scratch](https://www.youtube.com/watch?v=l8pRSuU81PU). His clear explanations and hands-on approach to deep learning and language models have been invaluable in shaping this implementation. If you want to understand the inner workings of GPT models, we highly recommend watching his video!
