import logging
from abc import ABC, abstractmethod

from flow_eval.eval_data_types import EvalInput, EvalOutput
from flow_eval.utils.result_writer import write_results_to_disk

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Base class for all synchronous evaluators.

    All evaluators must implement both evaluate() and batch_evaluate() methods.
    """

    def __init__(self, output_dir: str | None = "output/"):
        """Initialize base evaluator.

        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir

    def _save_results(
        self,
        eval_inputs: list[EvalInput],
        eval_outputs: list[EvalOutput],
        metadata: dict,
        eval_name: str,
        append: bool = False,
    ):
        """Save results to disk."""
        logger.info(f"{'Appending' if append else 'Saving'} results to {self.output_dir}")
        write_results_to_disk(
            eval_inputs,
            eval_outputs,
            metadata,
            eval_name,
            self.output_dir,
            append=append,
        )

    @abstractmethod
    def evaluate(self, eval_input: EvalInput, save_results: bool = False) -> EvalOutput:
        """Evaluate a single input.

        Args:
            eval_input: Input to evaluate
            save_results: Whether to save results to disk

        Returns:
            Evaluation output containing score and optional feedback
        """
        pass

    @abstractmethod
    def batch_evaluate(
        self,
        eval_inputs: list[EvalInput],
        save_results: bool = True,
    ) -> list[EvalOutput]:
        """Batch evaluate multiple inputs.

        Args:
            eval_inputs: List of inputs to evaluate
            save_results: Whether to save results to disk

        Returns:
            List of evaluation outputs
        """
        pass


class AsyncBaseEvaluator(ABC):
    """Base class for all asynchronous evaluators.

    All async evaluators must implement both async_evaluate() and async_batch_evaluate() methods.
    """

    def __init__(self, output_dir: str | None = "output/"):
        """Initialize async base evaluator.

        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir

    def _save_results(
        self,
        eval_inputs: list[EvalInput],
        eval_outputs: list[EvalOutput],
        metadata: dict,
        eval_name: str,
        append: bool = False,
    ):
        """Save results to disk."""
        logger.info(f"{'Appending' if append else 'Saving'} results to {self.output_dir}")
        write_results_to_disk(
            eval_inputs,
            eval_outputs,
            metadata,
            eval_name,
            self.output_dir,
            append=append,
        )

    @abstractmethod
    async def async_evaluate(
        self,
        eval_input: EvalInput,
        save_results: bool = False,
        append: bool = False,
    ) -> EvalOutput:
        """Evaluate a single input asynchronously.

        Args:
            eval_input: Input to evaluate
            save_results: Whether to save results to disk
            append: Whether to append results to existing file

        Returns:
            Evaluation output containing score and optional feedback
        """
        pass

    @abstractmethod
    async def async_batch_evaluate(
        self,
        eval_inputs: list[EvalInput],
        save_results: bool = True,
        append: bool = False,
    ) -> list[EvalOutput]:
        """Batch evaluate multiple inputs asynchronously.

        Args:
            eval_inputs: List of inputs to evaluate
            save_results: Whether to save results to disk
            append: Whether to append results to existing file

        Returns:
            List of evaluation outputs
        """
        pass
