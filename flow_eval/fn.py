import logging
from collections.abc import Callable
from inspect import signature
from typing import Any, get_type_hints

from flow_eval.core import BaseEvaluator, EvalInput, EvalOutput

logger = logging.getLogger(__name__)


class FunctionEvaluator(BaseEvaluator):
    """Evaluator that wraps simple evaluation functions.

    Example:
        def exact_match(response: str, ground_truth: str) -> bool:
            return response == ground_truth

        evaluator = FunctionEvaluator(exact_match)

        # The function args will be extracted from EvalInput.output and EvalInput.expected_output
        result = evaluator.evaluate(eval_input)
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        output_dir: str | None = "output/",
    ):
        """Initialize FunctionEvaluator with an evaluation function.

        Args:
            fn: Function that takes simple args and returns a score
            output_dir: Directory to save evaluation results
        """
        super().__init__(output_dir)
        self.fn = fn
        self.param_names = self._validate_function(fn)

    def _validate_function(self, fn: Callable) -> list[str]:
        """Validate function and return its parameter names."""
        sig = signature(fn)
        param_names = list(sig.parameters.keys())

        if not param_names:
            raise ValueError("Function must take at least one parameter")

        # Get type hints
        type_hints = get_type_hints(fn)

        # Validate param types are all strings
        for param in param_names:
            if param not in type_hints:
                raise ValueError(f"Parameter '{param}' must have a type annotation")
            if type_hints[param] != str:
                raise ValueError(
                    f"Parameter '{param}' must be annotated as str, got" f" {type_hints[param]}"
                )

        # Validate return type is either bool, int, float or str
        return_type = type_hints.get("return")
        if return_type not in (bool, int, float, str):
            raise ValueError(f"Return type must be bool, int, float or str, got" f" {return_type}")

        return param_names

    def _extract_args(self, eval_input: EvalInput) -> dict[str, str]:
        """Extract function arguments from EvalInput."""
        args = {}

        # Try to find args in output and expected_output
        for param in self.param_names:
            if param in eval_input.output:
                args[param] = eval_input.output[param]
            elif param in eval_input.expected_output:
                args[param] = eval_input.expected_output[param]
            else:
                # Look in inputs
                for input_dict in eval_input.inputs:
                    if param in input_dict:
                        args[param] = input_dict[param]
                        break
                else:
                    raise ValueError(f"Could not find value for parameter '{param}' in EvalInput")

        return args

    def evaluate(self, eval_input: EvalInput, save_results: bool = False) -> EvalOutput:
        """Evaluate a single EvalInput object."""
        try:
            # Extract args from eval_input
            args = self._extract_args(eval_input)

            # Call function with extracted args
            score = self.fn(**args)
            eval_output = EvalOutput(score=score)

            if save_results:
                self._save_results(
                    [eval_input],
                    [eval_output],
                    metadata={"model_id": f"function-{self.fn.__name__}", "model_type": "function"},
                    eval_name=self.fn.__name__,
                    append=False,
                )

            return eval_output
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def batch_evaluate(
        self,
        eval_inputs: list[EvalInput],
        save_results: bool = True,
    ) -> list[EvalOutput]:
        """Batch evaluate a list of EvalInput objects."""
        try:
            eval_outputs = []
            for eval_input in eval_inputs:
                eval_output = self.evaluate(eval_input, save_results=False)
                eval_outputs.append(eval_output)

            if save_results:
                self._save_results(
                    eval_inputs=eval_inputs,
                    eval_outputs=eval_outputs,
                    metadata={"model_id": f"function-{self.fn.__name__}", "model_type": "function"},
                    eval_name=self.fn.__name__,
                    append=False,
                )

            return eval_outputs
        except Exception as e:
            logger.error(f"Batch evaluation failed: {e}")
            raise
