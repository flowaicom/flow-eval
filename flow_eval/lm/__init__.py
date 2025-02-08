"""Language Model evaluation module."""

from flow_eval.lm.metrics import list_all_lm_evals
from flow_eval.lm.models.common import AsyncBaseEvaluatorModel, BaseEvaluatorModel
from flow_eval.lm.types import LMEval, RubricItem

__all__ = [
    "AsyncBaseEvaluatorModel",
    "BaseEvaluatorModel",
    "LMEval",
    "RubricItem",
]

# Conditional imports for optional model implementations
try:
    from flow_eval.lm.models.huggingface import Hf

    __all__.append("Hf")
except ImportError:
    Hf = None

try:
    from flow_eval.lm.models.vllm import Vllm

    __all__.append("Vllm")
except ImportError:
    Vllm = None

try:
    from flow_eval.lm.models.llamafile import Llamafile

    __all__.append("Llamafile")
except ImportError:
    Llamafile = None

try:
    from flow_eval.lm.models.baseten import Baseten

    __all__.append("Baseten")
except ImportError:
    Baseten = None

try:
    from flow_eval.lm.models.openai import OpenAIModel

    __all__.append("OpenAIModel")
except ImportError as e:
    print(f"Error importing OpenAIModel: {e}")
    OpenAIModel = None


def list_all_available_models():
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


__all__ += list_all_lm_evals()
__all__ += list_all_available_models()
