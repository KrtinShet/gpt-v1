# Production-Grade GPT-2 Clone

This project is a refactored, production-grade implementation of a GPT-2 style language model in PyTorch. It is designed to be modular, configurable, testable, and ready for inference.

## Overview

This codebase is a result of transforming an experimental GPT-2 implementation into a robust and maintainable project. The key features include:

*   **Modular Structure:** Code is organized into logical components for data handling, modeling, training, evaluation, and inference.
*   **Configuration-driven:** All parameters are managed through `hydra` and YAML configuration files.
*   **Testable:** Unit tests are included for core components using `pytest`.
*   **Logging and Monitoring:** Integrated with `wandb` for experiment tracking.
*   **Checkpointing:** Save and resume training from checkpoints.
*   **Inference Ready:** A dedicated script for generating text from a trained model.

## Project Structure

```
├── src/
│   ├── config/              # Configs for training/inference
│   ├── models/              # GPT2 model code
│   ├── data/                # Data loading/preprocessing
│   ├── train/               # Training logic
│   ├── eval/                # Evaluation scripts
│   ├── inference/           # Inference/prompting interface
│   └── utils/               # Logging, checkpointing, etc.
├── scripts/                 # CLI wrappers (train.py, eval.py, infer.py)
├── tests/                   # Pytest-compatible unit tests
├── configs/                 # YAML config files
├── notebooks/               # Jupyter experiments (optional)
├── README.md
├── pyproject.toml
└── .env / .env.example
```

## Getting Started

### 1. Prerequisites

*   Python 3.10+
*   [UV](https://github.com/astral-sh/uv) package manager

### 2. Installation

Clone the repository and install the dependencies using `uv`:

```sh
git clone <repository-url>
cd <repository-name>
uv pip install -e .[dev]
```

The `.[dev]` extra installs the development dependencies, including `black`, `ruff`, and `mypy`.

### 3. Configuration

All aspects of the project are controlled by YAML configuration files in the `configs/` directory. The main configuration file is `configs/default.yaml`. You can override any parameter from the command line.

For example, to change the batch size for training:

```sh
python scripts/train.py train.batch_size=64
```

### 4. Training

To start training the model, run the `scripts/train.py` script:

```sh
python scripts/train.py
```

This will:

*   Load the data and tokenizer.
*   Initialize the GPT-2 model.
*   Start the training loop, logging to `wandb`.
*   Save checkpoints to the `checkpoints/` directory.

To resume training from a checkpoint, set `train.resume=true` in the configuration or on the command line:

```sh
python scripts/train.py train.resume=true
```

### 5. Inference

To generate text from a trained model, use the `scripts/infer.py` script. You need to provide a path to a checkpoint and a prompt.

```sh
python scripts/infer.py inference.checkpoint_path=checkpoints/latest.pt inference.prompt="To be, or not to be"
```

### 6. Evaluation

To evaluate the model's perplexity on the test set, use the `scripts/eval.py` script.

```sh
python scripts/eval.py eval.checkpoint_path=checkpoints/latest.pt
```

### 7. Testing

To run the unit tests, use `pytest`:

```sh
pytest
```

### 8. Code Quality

This project uses `black` for code formatting, `ruff` for linting, and `mypy` for type checking. To run the checks:

```sh
black .
ruff check .
mypy .
```

## Tribute

This project is inspired by [Andrej Karpathy's YouTube video on building GPT from scratch](https://www.youtube.com/watch?v=l8pRSuU81PU). His clear explanations and hands-on approach to deep learning and language models have been invaluable in shaping this implementation.