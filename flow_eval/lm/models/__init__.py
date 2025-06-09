# Lazy import system to avoid loading heavy dependencies when not needed
# Models will only be imported when actually accessed

import importlib
import sys
from typing import TYPE_CHECKING, Any, Dict, Type

# Model registry mapping model names to their module paths
_MODEL_REGISTRY: dict[str, str] = {
    "Baseten": "flow_eval.lm.models.baseten",
    "Hf": "flow_eval.lm.models.huggingface",
    "Llamafile": "flow_eval.lm.models.llamafile",
    "OpenAIModel": "flow_eval.lm.models.openai",
    "Vllm": "flow_eval.lm.models.vllm",
}

# Cache for imported models
_IMPORTED_MODELS: dict[str, type[Any]] = {}

__all__ = list(_MODEL_REGISTRY.keys())


def __getattr__(name: str) -> Any:
    """Lazy import models only when accessed."""
    if name in _MODEL_REGISTRY:
        # Check cache first
        if name in _IMPORTED_MODELS:
            return _IMPORTED_MODELS[name]

        # Import the module and get the class
        try:
            module_path = _MODEL_REGISTRY[name]
            module = importlib.import_module(module_path)
            model_class = getattr(module, name)

            # Cache for future use
            _IMPORTED_MODELS[name] = model_class
            return model_class
        except ImportError as e:
            # Re-raise with more helpful error message
            raise ImportError(
                f"Failed to import {name}. Make sure you have installed the required "
                f"dependencies. For example: pip install flow-eval[{_get_extra_name(name)}]"
                f"\nOriginal error: {e}"
            ) from e

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_extra_name(model_name: str) -> str:
    """Get the pip extra name for a model."""
    extras_map = {
        "Hf": "hf",
        "Vllm": "vllm",
        "Llamafile": "llamafile",
        "OpenAIModel": "openai",
        "Baseten": "baseten",
    }
    return extras_map.get(model_name, "unknown")


# For static type checking and IDE support

if TYPE_CHECKING:
    from .baseten import Baseten
    from .huggingface import Hf
    from .llamafile import Llamafile
    from .openai import OpenAIModel
    from .vllm import Vllm
