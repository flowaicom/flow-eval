# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Flow-eval is an evaluation library for LLM applications that supports LLM-as-a-judge, function-based, and similarity-based evaluation. The library provides flexible evaluation specifications, multiple inference backends, and seamless integration with popular frameworks like LlamaIndex and Haystack.

## Core Architecture

### Main Components
- **Evaluators**: `LMEvaluator` and `AsyncLMEvaluator` in `flow_eval/lm_eval.py` - main evaluation engines
- **Base Classes**: `BaseEvaluator` and `AsyncBaseEvaluator` in `flow_eval/core/base.py` - abstract interfaces
- **Models**: Multiple inference backends in `flow_eval/lm/models/` (vLLM, HuggingFace, Llamafile, OpenAI, Baseten)
- **Core Types**: `EvalInput`, `EvalOutput` in `flow_eval/core/types.py`
- **Evaluation Specs**: `LMEval` and rubrics in `flow_eval/lm/` for defining evaluation criteria

### Inference Backend Architecture
The library supports multiple model backends through a unified interface:
- **vLLM**: High-performance backend for Ampere GPUs or newer
- **HuggingFace Transformers**: Broader hardware compatibility
- **Llamafile**: CPU-focused, good for Apple Silicon
- **OpenAI**: Remote API-based evaluation
- **Baseten**: Remote execution with async capabilities

Each backend implements `BaseEvaluatorModel` or `AsyncBaseEvaluatorModel` interfaces.

## Development Commands

### Installation
```bash
# Development installation
pip install -e ".[dev]"

# With specific backends
pip install -e ".[dev,vllm,hf,llamafile,baseten]"

# For vLLM with flash attention
pip install 'flash_attn>=2.6.3' --no-build-isolation
```

### Testing
```bash
# Run all unit tests
pytest tests/unit

# Run specific test categories
pytest tests/unit/             # Unit tests only
pytest tests/e2e-local/        # Local end-to-end tests
pytest tests/e2e-cloud-gpu/    # Cloud GPU tests (requires GPU)

# Run with coverage
pytest tests/unit --cov=./

# Run single test file
pytest tests/unit/test_lm_eval.py
```

### Linting and Formatting
```bash
# Check all linting (run these before committing)
ruff check . --config pyproject.toml
black --check . --config pyproject.toml
isort --check-only . --settings-path pyproject.toml

# Auto-fix formatting
black . --config pyproject.toml
isort . --settings-path pyproject.toml
ruff check . --config pyproject.toml --fix

# Type checking
mypy flow_eval/
```

### Pre-commit Setup
```bash
pre-commit install
pre-commit run --all-files
```

## Key Patterns

### Creating Evaluators
```python
from flow_eval import LMEvaluator
from flow_eval.lm.models import Vllm
from flow_eval.lm.metrics import RESPONSE_FAITHFULNESS_5POINT

model = Vllm()  # or Hf(), Llamafile(), etc.
evaluator = LMEvaluator(eval=RESPONSE_FAITHFULNESS_5POINT, model=model)
```

### Evaluation Input Structure
All evaluations use `EvalInput` with:
- `inputs`: List of input dictionaries (query, context, etc.)
- `output`: Single output dictionary (response, etc.)

### Model Adapter Pattern
New model backends should inherit from `BaseEvaluatorModel` and implement:
- `_generate(prompt: str) -> str`
- `_batch_generate(prompts: list[str]) -> list[str]`

For async models, inherit from `AsyncBaseEvaluatorModel` and implement async variants.

## Configuration Notes

- All model backends are configured in `flow_eval/lm/models/`
- Evaluation metrics are defined in `flow_eval/lm/metrics.py`
- The library uses Pydantic for type validation and parsing
- Results are saved to `output/` directory by default
- GPU memory utilization can be controlled via model parameters (e.g., `Vllm(gpu_memory_utilization=0.70)`)

## Testing Strategy

- **Unit tests**: Mock model responses, focus on evaluation logic
- **E2E local**: Test with actual models on local hardware
- **E2E cloud GPU**: Test GPU-specific backends in CI
- All tests use pytest with coverage reporting to Codecov
