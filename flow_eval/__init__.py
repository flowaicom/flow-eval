from importlib.metadata import PackageNotFoundError, version

from flow_eval.eval_data_types import EvalInput, EvalOutput
from flow_eval.flow_eval import AsyncEvaluator, Evaluator
from flow_eval.metrics import CustomMetric, Metric, RubricItem, list_all_metrics
from flow_eval.models.common import BaseEvaluatorModel
from flow_eval.utils.prompt_formatter import format_rubric, format_user_prompt, format_vars

try:
    __version__ = version("flow-eval")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

__all__ = [
    "Evaluator",
    "AsyncEvaluator",
    "EvalInput",
    "format_vars",
    "format_rubric",
    "format_user_prompt",
    "RubricItem",
    "Metric",
    "CustomMetric",
    "BaseEvaluatorModel",
    "EvalOutput",
]

# Conditional imports for optional dependencies
try:
    from flow_eval.models.huggingface import Hf

    __all__.append("Hf")
except ImportError:
    Hf = None

try:
    from flow_eval.models.vllm import Vllm

    __all__.append("Vllm")
except ImportError:
    Vllm = None

try:
    from flow_eval.models.llamafile import Llamafile

    __all__.append("Llamafile")
except ImportError:
    Llamafile = None

try:
    from flow_eval.models.baseten import Baseten

    __all__.append("Baseten")
except ImportError:
    Baseten = None

try:
    from flow_eval.models.openai import OpenAIModel

    __all__.append("OpenAIModel")
except ImportError:
    OpenAIModel = None


def get_available_models():
    """Return a list of available model classes based on installed extras."""
    models = [BaseEvaluatorModel]
    if Hf is not None:
        models.append(Hf)
    if Vllm is not None:
        models.append(Vllm)
    if Llamafile is not None:
        models.append(Llamafile)
    if Baseten is not None:
        models.append(Baseten)
    if OpenAIModel is not None:
        models.append(OpenAIModel)
    return models


__all__.append("get_available_models")

# Add all metric names to __all__
__all__ += list_all_metrics()
