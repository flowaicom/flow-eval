import asyncio
from collections.abc import Sequence
from typing import Any

from langchain.evaluation import StringEvaluator

from flow_eval import AsyncLMEvaluator, LMEvaluator
from flow_eval.core import EvalInput
from flow_eval.lm import LMEval
from flow_eval.lm.models.common import AsyncBaseEvaluatorModel, BaseEvaluatorModel


class LangChainLMEvaluator(StringEvaluator):
    """LangChainLMEvaluator is a custom evaluator for LangChain.

    It uses LMEvaluator to evaluate the LLM outputs.
    """

    def __init__(self, eval: LMEval, model: BaseEvaluatorModel | AsyncBaseEvaluatorModel):
        """Initialize the LlamaIndexEvaluator."""
        if isinstance(eval, LMEval):
            self.eval = eval
        else:
            raise ValueError("Invalid eval type. Use LMEval.")

        # Validate model and choose appropriate Evaluator class
        if isinstance(model, (BaseEvaluatorModel, AsyncBaseEvaluatorModel)):
            self.model = model
        else:
            raise ValueError(
                "The model must be an instance of BaseEvaluatorModel or AsyncBaseEvaluatorModel."
            )

        # Determine if the model is async-capable
        self.is_async = hasattr(self.model, "exec_async") and self.model.exec_async

        # Initialize the appropriate judge based on async capability
        if self.is_async:
            self.evaluator = AsyncLMEvaluator(eval=self.eval, model=self.model)
        else:
            self.evaluator = LMEvaluator(eval=self.eval, model=self.model)

    def _prepare_eval_input(
        self,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        **kwargs: Any,
    ) -> EvalInput:
        # Combine all inputs into a single dictionary
        all_inputs = {"prediction": prediction, "reference": reference, "input": input, **kwargs}

        # Prepare eval_inputs based on metric's input_columns
        eval_inputs = []
        for req_input in self.eval.input_columns:
            if req_input in all_inputs:
                value = all_inputs[req_input]
                if isinstance(value, (list, Sequence)) and not isinstance(value, str):
                    eval_inputs.extend([{req_input: v} for v in value])
                else:
                    eval_inputs.append({req_input: value})

        # Prepare the output
        output_key = self.eval.output_column
        output_value = all_inputs.get(
            output_key, prediction
        )  # Default to prediction if not specified
        expected_output_key = self.eval.expected_output_column
        expected_output_value = all_inputs.get(
            expected_output_key, reference
        )  # Default to reference if not specified

        return EvalInput(
            inputs=eval_inputs,
            output={output_key: output_value},
            expected_output={expected_output_key: expected_output_value},
        )

    def _evaluate_strings(
        self,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        eval_input = self._prepare_eval_input(prediction, reference, input, **kwargs)
        result = self.evaluator.evaluate(eval_input, save_results=False)

        return {
            "score": result.score,
            "reasoning": result.feedback,
        }

    async def _aevaluate_strings(
        self,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        sleep_time_in_seconds: int = 1,
        **kwargs: Any,
    ) -> dict[str, Any]:
        await asyncio.sleep(sleep_time_in_seconds)
        eval_input = self._prepare_eval_input(prediction, reference, input, **kwargs)
        result = await self.evaluator.async_evaluate(eval_input, save_results=False)

        return {
            "score": result.score,
            "reasoning": result.feedback,
        }

    @property
    def requires_input(self) -> bool:
        """Requires input."""
        return "input" in self.eval.input_columns

    @property
    def requires_reference(self) -> bool:
        """Requires reference."""
        return "reference" in self.eval.expected_output_column

    @property
    def evaluation_name(self) -> str:
        """Get metric name."""
        return f"flow_eval_{self.eval.name}"

    def get_required_inputs(self) -> list[str]:
        """Get required inputs."""
        return (
            self.eval.input_columns + [self.eval.output_column] + [self.eval.expected_output_column]
        )
