"""Flow-eval: A comprehensive evaluation framework for LLMs."""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from flow_eval.fn import FunctionEvaluator
from flow_eval.lm import list_all_available_models
from flow_eval.lm.metrics import list_all_lm_evals
from flow_eval.lm_eval import AsyncLMEvaluator, LMEvaluator

if TYPE_CHECKING:
    from flow_eval.similarity import AnswerSimilarityEvaluator

try:
    __version__ = version("flow-eval")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

__all__ = [
    "LMEvaluator",
    "AsyncLMEvaluator",
    "AnswerSimilarityEvaluator",
    "FunctionEvaluator",
    "list_all_available_models",
]

__all__ += list_all_lm_evals()


def __getattr__(name: str) -> Any:
    """Lazy import for AnswerSimilarityEvaluator."""
    if name == "AnswerSimilarityEvaluator":
        try:
            from flow_eval.similarity import AnswerSimilarityEvaluator

            return AnswerSimilarityEvaluator
        except ImportError as e:
            raise ImportError(
                "AnswerSimilarityEvaluator requires additional dependencies. "
                "Install them with: pip install flow-eval[similarity]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
