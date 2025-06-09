"""Language Model evaluation module."""

import importlib
import sys
from typing import TYPE_CHECKING, Any, List, Type

from flow_eval.lm.metrics import list_all_lm_evals
from flow_eval.lm.models.common import AsyncBaseEvaluatorModel, BaseEvaluatorModel
from flow_eval.lm.types import LMEval, RubricItem

# Base exports that are always available
__all__ = [
    "AsyncBaseEvaluatorModel",
    "BaseEvaluatorModel",
    "LMEval",
    "RubricItem",
    "list_all_available_models",
]

# Add metric names to __all__
__all__ += list_all_lm_evals()

# Model names that can be imported (but won't be imported until accessed)
_AVAILABLE_MODELS = ["Hf", "Vllm", "Llamafile", "Baseten", "OpenAIModel"]
__all__ += _AVAILABLE_MODELS


def list_all_available_models() -> list[type[BaseEvaluatorModel]]:
    """Return a list of available model classes based on installed extras.

    This function now dynamically checks which models can be imported
    rather than attempting to import them at module level.
    """
    models = [BaseEvaluatorModel]

    model_mapping = {
        "Hf": "flow_eval.lm.models.huggingface",
        "Vllm": "flow_eval.lm.models.vllm",
        "Llamafile": "flow_eval.lm.models.llamafile",
        "Baseten": "flow_eval.lm.models.baseten",
        "OpenAIModel": "flow_eval.lm.models.openai",
    }

    for model_name, module_path in model_mapping.items():
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, model_name)
            models.append(model_class)
        except ImportError:
            # Model not available due to missing dependencies
            pass

    return models


def __getattr__(name: str) -> Any:
    """Lazy import models from flow_eval.lm.models."""
    if name in _AVAILABLE_MODELS:
        # Delegate to the models subpackage
        from flow_eval.lm import models

        return getattr(models, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# For static type checking and IDE support

if TYPE_CHECKING:
    from flow_eval.lm.models import Baseten, Hf, Llamafile, OpenAIModel, Vllm
