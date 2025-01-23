"""Flow-eval: A comprehensive evaluation framework for LLMs."""

from importlib.metadata import PackageNotFoundError, version

from flow_eval.fn import FunctionEvaluator
from flow_eval.lm import list_all_available_models
from flow_eval.lm.metrics import list_all_lm_evals
from flow_eval.lm_eval import AsyncLMEvaluator, LMEvaluator
from flow_eval.similarity import AnswerSimilarityEvaluator

try:
    __version__ = version("flow-eval")
except PackageNotFoundError:
    # package is not installed
    __version__ = "unknown"

__all__ = ["LMEvaluator", "AsyncLMEvaluator", "AnswerSimilarityEvaluator", "FunctionEvaluator"]

__all__ += list_all_lm_evals()
__all__ += list_all_available_models()
