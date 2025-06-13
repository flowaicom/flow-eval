# `flow-eval`

<p align="center" style="font-family: 'Courier New', Courier, monospace;">
  <strong>
    <a href="https://www.flow-ai.com/judge">Technical Report</a> |
    <a href="https://huggingface.co/collections/flowaicom/flow-eval-v01-66e6af5fc3b3a128bde07dec">Model Weights</a> |
    <a href="https://huggingface.co/spaces/flowaicom/Flow-eval-v0.1">HuggingFace Space</a> |
    <a href="https://github.com/flowaicom/lm-evaluation-harness/tree/Flow-eval-v0.1_evals/lm_eval/tasks/flow_eval_evals">Evaluation Code</a> |
    <a href="https://github.com/flowaicom/flow-eval/tree/main/examples">Tutorials</a>
  </strong>
</p>

<p align="center">
<a href="https://github.com/flowaicom/flow-eval/stargazers/" target="_blank">
    <img src="https://img.shields.io/github/stars/flowaicom/flow-eval?style=social&label=Star&maxAge=3600" alt="GitHub stars">
</a>
<a href="https://github.com/flowaicom/flow-eval/releases" target="_blank">
    <img src="https://img.shields.io/github/v/release/flowaicom/flow-eval?color=white" alt="Release">
</a>
<a href="https://www.youtube.com/@flowaicom" target="_blank">
    <img alt="YouTube Channel Views" src="https://img.shields.io/youtube/channel/views/UCo2qL1nIQRHiPc0TF9xbqwg?style=social">
</a>
<a href="https://github.com/flowaicom/flow-eval/actions/workflows/test-and-lint.yml" target="_blank">
    <img src="https://github.com/flowaicom/flow-eval/actions/workflows/test-and-lint.yml/badge.svg" alt="Build">
</a>
<a href="https://codecov.io/gh/flowaicom/flow-eval" target="_blank">
    <img src="https://codecov.io/gh/flowaicom/flow-eval/branch/main/graph/badge.svg?token=AEGC7W3DGE" alt="Code coverage">
</a>
<a href="https://github.com/flowaicom/flow-eval/blob/main/LICENSE" target="_blank">
    <img src="https://img.shields.io/static/v1?label=license&message=Apache%202.0&color=white" alt="License">
</a>
<a href="https://app.fossa.com/projects/git%2Bgithub.com%2Fflowaicom%2Fflow-eval?ref=badge_shield" alt="FOSSA Status"><img src="https://app.fossa.com/api/projects/git%2Bgithub.com%2Fflowaicom%2Fflow-eval.svg?type=shield"/></a>
</p>

`flow-eval` is an evaluation library for LLM applications that supports LLM-as-a-judge, function-based, and similarity-based evaluation. It provides flexible evals, multiple inference backends, and seamless integration with popular frameworks.

## Installation

Install `flow-eval` using pip:

```bash
pip install -e ".[vllm,hf]"
pip install 'flash_attn>=2.6.3' --no-build-isolation
```

Extras available:
- `dev` to install development dependencies
- `hf` to install Hugging Face Transformers dependencies
- `vllm` to install vLLM dependencies
- `llamafile` to install Llamafile dependencies
- `baseten` to install Baseten dependencies
- `similarity` to install similarity evaluation dependencies (torch, sentence-transformers)
- `openai` to install OpenAI API dependencies

## Quick Start

Here's a simple example to get you started:

```python
from flow_eval import LMEvaluator
from flow_eval.lm.models import Vllm, Llamafile, Hf
from flow_eval.core import EvalInput
from flow_eval.lm.metrics import RESPONSE_FAITHFULNESS_5POINT
from IPython.display import Markdown, display

# If you are running on an Ampere GPU or newer, create a model using VLLM
model = Vllm()

# If you have other applications open taking up VRAM, you can use less VRAM by setting gpu_memory_utilization to a lower value.
# model = Vllm(gpu_memory_utilization=0.70)

# Or if not running on Ampere GPU or newer, create a model using no flash attn and Hugging Face Transformers
# model = Hf(flash_attn=False)

# Or create a model using Llamafile if not running an Nvidia GPU & running a Silicon MacOS for example
# model = Llamafile()

# Initialize the evaluator
faithfulness_evaluator = LMEvaluator(
    eval=RESPONSE_FAITHFULNESS_5POINT,
    model=model
)

# Sample to evaluate
query = """..."""
context = """...""""
response = """..."""

# Create an EvalInput
# We want to evaluate the response to the customer issue based on the context and the user instructions
eval_input = EvalInput(
    inputs=[
        {"query": query},
        {"context": context},
    ],
    output={"response": response},
)

# Run the evaluation
result = faithfulness_evaluator.evaluate(eval_input, save_results=False)

# Display the result
display(Markdown(f"__Feedback:__\n{result.feedback}\n\n__Score:__\n{result.score}"))
```

## Usage

### Inference Options

The library supports multiple inference backends to accommodate different hardware configurations and performance needs:

1. **vLLM**:
   - Best for NVIDIA GPUs with Ampere architecture or newer (e.g., RTX 3000 series, A100, H100)
   - Offers the highest performance and throughput
   - Requires CUDA-compatible GPU

   ```python
   from flow_eval.lm.models import Vllm

   model = Vllm()
   ```

2. **Hugging Face Transformers**:
   - Compatible with a wide range of hardware, including older NVIDIA GPUs
   - Supports CPU inference (slower but universally compatible)
   - It is slower than vLLM but generally compatible with more hardware.

    If you are running on an Ampere GPU or newer:
   ```python
   from flow_eval.lm.models import Hf

   model = Hf()
   ```

   If you are not running on an Ampere GPU or newer, disable flash attention:
   ```python
   from flow_eval.lm.models import Hf

   model = Hf(flash_attn=False)
   ```

3. **Llamafile**:
   - Ideal for non-NVIDIA hardware, including Apple Silicon
   - Provides good performance on CPUs
   - Self-contained, easy to deploy option

   ```python
   from flow_eval.lm.models import Llamafile

   model = Llamafile()
   ```

4. **Baseten**:
    - Remote execution.
    - Machine independent.
    - Improved concurrency patterns for larger workloads.

  ```python
  from flow_eval.lm.models import Baseten

  model = Baseten()
  ```
  For detailed information on using Baseten, visit the [Baseten readme](https://github.com/flowaicom/flow-eval/blob/feat/baseten-integration/flow_eval/models/adapters/baseten/README.md).

5. **OpenAI**:
    - Uses OpenAI's API for remote evaluation
    - Supports various OpenAI models including GPT-4
    - Requires OPENAI_API_KEY environment variable

  ```python
  from flow_eval.lm.models import OpenAIModel
  import os

  # Set your API key
  os.environ["OPENAI_API_KEY"] = "your-api-key-here"

  model = OpenAIModel(model="gpt-4")
  ```
  Install with: `pip install flow-eval[openai]`

Choose the inference backend that best matches your hardware and performance requirements. The library provides a unified interface for all these options, making it easy to switch between them as needed.


### Evals

`Flow-eval-v0.1` was trained to handle any custom metric that can be expressed as a combination of evaluation criteria and rubric, and required inputs and outputs.

#### Pre-defined Evals

For convenience, `flow-eval` library comes with pre-defined metrics such as `RESPONSE_CORRECTNESS` or `RESPONSE_FAITHFULNESS`. You can check the full list by running:

```python
from flow_eval import list_all_lm_evals

list_all_lm_evals()
```

### Batched Evaluations

For efficient processing of multiple inputs, you can use the `batch_evaluate` method:

```python
# Read the sample data
import json
from flow_eval import LMEvaluator
from flow_eval.core import EvalInput
from flow_eval.lm.models import Vllm
from flow_eval.lm.metrics import RESPONSE_FAITHFULNESS_5POINT
from IPython.display import Markdown, display

# Initialize the model
model = Vllm()

# Initialize the evaluator
faithfulness_evaluator = LMEvaluator(
    eval=RESPONSE_FAITHFULNESS_5POINT,
    model=model
)

# Load some sampledata
with open("sample_data/csr_assistant.json", "r") as f:
    data = json.load(f)

# Create a list of inputs and outputs
inputs_batch = [
    [
        {"query": sample["query"]},
        {"context": sample["context"]},
    ]
    for sample in data
]
outputs_batch = [{"response": sample["response"]} for sample in data]

# Create a list of EvalInput
eval_inputs_batch = [EvalInput(inputs=inputs, output=output) for inputs, output in zip(inputs_batch, outputs_batch)]

# Run the batch evaluation
results = faithfulness_evaluator.batch_evaluate(eval_inputs_batch, save_results=False)

# Visualizing the results
for i, result in enumerate(results):
    display(Markdown(f"__Sample {i+1}:__"))
    display(Markdown(f"__Feedback:__\n{result.feedback}\n\n__Score:__\n{result.score}"))
    display(Markdown("---"))
```

### Similarity-Based Evaluation

For evaluating semantic similarity between responses and expected outputs, you can use the `AnswerSimilarityEvaluator`:

```python
from flow_eval import AnswerSimilarityEvaluator
from flow_eval.core import EvalInput

# Initialize the similarity evaluator
similarity_evaluator = AnswerSimilarityEvaluator(
    model_name="all-mpnet-base-v2",  # Sentence transformer model
    similarity_fn_name="cosine",      # cosine, dot, euclidean, or manhattan
    output_dir=None                   # Set to save results
)

# Create evaluation input with expected output
eval_input = EvalInput(
    inputs=[],
    output={"response": "The quick brown fox jumps over the lazy dog"},
    expected_output={"reference": "A fast brown fox leaps above a sleeping canine"}
)

# Run similarity evaluation
result = similarity_evaluator.evaluate(eval_input, save_results=False)
print(f"Similarity score: {result.score:.3f}")  # Range: 0.0 to 1.0

# Batch evaluation
eval_inputs = [
    EvalInput(
        inputs=[],
        output={"response": "Hello world"},
        expected_output={"reference": "Hi world"}
    ),
    EvalInput(
        inputs=[],
        output={"response": "Python programming"},
        expected_output={"reference": "Python coding"}
    )
]

results = similarity_evaluator.batch_evaluate(eval_inputs, save_results=False)
for i, result in enumerate(results):
    print(f"Sample {i+1} similarity: {result.score:.3f}")
```

The similarity evaluator requires the `similarity` extra:
```bash
pip install flow-eval[similarity]
```

## Advanced Usage

> [!WARNING]
> There exists currently a reported issue with Phi-3 models that produces gibberish outputs with contexts longer than 4096 tokens, including input and output. This issue has been recently fixed in the transformers library so we recommend using the `Hf()` model configuration for longer contexts at the moment. For more details, refer to: [#33129](https://github.com/huggingface/transformers/pull/33129) and [#6135](https://github.com/vllm-project/vllm/issues/6135)


### Custom Evals

Create your own evaluation metrics:

```python
from flow_eval.lm import LMEval, RubricItem

custom_metric = LMEval(
    name="My Custom Metric",
    criteria="Evaluate based on X, Y, and Z.",
    rubric=[
        RubricItem(score=0, description="Poor performance"),
        RubricItem(score=1, description="Good performance"),
    ],
    input_columns=["query"],
    output_column="response"
)

evaluator = LMEvaluator(eval=custom_metric, config="Flow-eval-v0.1-AWQ")
```

### Integrations

We support an integration with Llama Index evaluation module and Haystack:
- [Llama Index tutorial](https://github.com/flowaicom/flow-eval/blob/main/examples/4_llama_index_evaluators.ipynb)
- [Haystack tutorial](https://github.com/flowaicom/flow-eval/blob/main/examples/5_evaluate_haystack_rag_pipeline.ipynb)

> Note that we are currently working on adding more integrations with other frameworks in the near future.
## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/flowaicom/flow-eval.git
   cd flow-eval
   ```

2. Create a virtual environment:
    ```bash
    virtualenv ./.venv
    ```
    or

    ```bash
    python -m venv ./.venv
    ```

3. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
   or
   ```bash
   pip install -e ".[dev,vllm]"
   ```
   for vLLM support.

5. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

6. Make sure you have trufflehog installed:
   ```bash
   # make trufflehog available in your path
   # macos
   brew install trufflehog
   # linux
   curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh | sh -s -- -b /usr/local/bin
   # nix
   nix profile install nixpkgs#trufflehog
   ```

7. Run pre-commit on all files:
   ```bash
   pre-commit run --all-files
   ```

8. You're now ready to start developing! You can run the main script with:
   ```bash
   python -m flow_eval
   ```

Remember to always activate your virtual environment when working on the project. To deactivate the virtual environment when you're done, simply run:
```bash
deactivate
```

## Running Tests

To run the tests for Flow-eval, follow these steps:

1. Navigate to the root directory of the project in your terminal.

2. Run the tests using pytest:
   ```bash
   pytest tests/
   ```

   This will discover and run all the tests in the `tests/` directory.

3. Run different test categories:
   ```bash
   pytest tests/unit/              # Unit tests only
   pytest tests/e2e-local/         # Local end-to-end tests
   pytest tests/e2e-cloud-gpu/     # Cloud GPU tests (requires GPU)
   ```

4. Run tests with coverage:
   ```bash
   pytest tests/unit --cov=./
   ```

5. For more verbose output, you can use the `-v` flag:
   ```bash
   pytest -v tests/
   ```

## Contributing

Contributions to `flow-eval` are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure that your code adheres to the project's coding standards and passes all tests.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2Fflowaicom%2Fflow-eval.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2Fflowaicom%2Fflow-eval?ref=badge_large)

## Acknowledgments

`flow-eval` is developed and maintained by the Flow AI team. We appreciate the contributions and feedback from the AI community in making this tool more robust and versatile.
