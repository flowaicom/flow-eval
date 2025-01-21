import asyncio
from collections.abc import Sequence
from typing import Any

from langchain.evaluation import StringEvaluator

from flow_eval import AsyncEvaluator, EvalInput, Evaluator
from flow_eval.metrics import CustomMetric, Metric
from flow_eval.models import AsyncBaseEvaluatorModel, BaseEvaluatorModel


class EvaluatorLangChainEvaluator(StringEvaluator):
    """EvaluatorLangchainEvaluator is a custom evaluator for LangChain.

    It uses Evaluator to evaluate the LLM outputs.
    """

    def __init__(
        self, metric: Metric | CustomMetric, model: BaseEvaluatorModel | AsyncBaseEvaluatorModel
    ):
        """Initialize the LlamaIndexEvaluator."""
        if isinstance(metric, (Metric, CustomMetric)):
            self.metric = metric
        else:
            raise ValueError("Invalid metric type. Use Metric or CustomMetric.")

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
            self.judge = AsyncEvaluator(metric=self.metric, model=self.model)
        else:
            self.judge = Evaluator(metric=self.metric, model=self.model)

    def _prepare_eval_input(
        self,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        **kwargs: Any,
    ) -> EvalInput:
        # Combine all inputs into a single dictionary
        all_inputs = {"prediction": prediction, "reference": reference, "input": input, **kwargs}

        # Prepare eval_inputs based on metric's required_inputs
        eval_inputs = []
        for req_input in self.metric.required_inputs:
            if req_input in all_inputs:
                value = all_inputs[req_input]
                if isinstance(value, (list, Sequence)) and not isinstance(value, str):
                    eval_inputs.extend([{req_input: v} for v in value])
                else:
                    eval_inputs.append({req_input: value})

        # Prepare the output
        output_key = self.metric.required_output
        output_value = all_inputs.get(
            output_key, prediction
        )  # Default to prediction if not specified

        return EvalInput(inputs=eval_inputs, output={output_key: output_value})

    def _evaluate_strings(
        self,
        prediction: str,
        reference: str | None = None,
        input: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        eval_input = self._prepare_eval_input(prediction, reference, input, **kwargs)
        result = self.judge.evaluate(eval_input, save_results=False)

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
        result = await self.judge.async_evaluate(eval_input, save_results=False)

        return {
            "score": result.score,
            "reasoning": result.feedback,
        }

    @property
    def requires_input(self) -> bool:
        """Requires input."""
        return "input" in self.metric.required_inputs

    @property
    def requires_reference(self) -> bool:
        """Requires reference."""
        return "reference" in self.metric.required_inputs

    @property
    def evaluation_name(self) -> str:
        """Get metric name."""
        return f"flow_eval_{self.metric.name}"

    def get_required_inputs(self) -> list[str]:
        """Get required inputs."""
        return self.metric.required_inputs + [self.metric.required_output]
