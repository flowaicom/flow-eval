import asyncio
import logging

from flow_eval.eval_data_types import EvalInput, EvalOutput
from flow_eval.evals.lm_eval import LMEval
from flow_eval.evaluators.base import AsyncBaseEvaluator, BaseEvaluator
from flow_eval.models.adapters.baseten.data_io import BatchResult
from flow_eval.models.adapters.baseten.errors import EvaluatorError
from flow_eval.models.common import AsyncBaseEvaluatorModel, BaseEvaluatorModel
from flow_eval.utils.prompt_formatter import format_rubric, format_user_prompt, format_vars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LMEvaluator(BaseEvaluator):
    """Evaluator using LLM-based evaluation specifications."""

    def __init__(
        self,
        eval: LMEval,
        model: BaseEvaluatorModel,
        output_dir: str | None = "output/",
    ):
        """Initialize LMEvaluator with an eval specification and model."""
        super().__init__(output_dir)
        self.eval = eval
        self.model = model

    # TODO - How to handle expected output?
    def _format_prompt(self, eval_input: EvalInput) -> str:
        """Format the prompt for a single evaluation input."""
        prompt_variables = {
            "INPUTS": format_vars(eval_input.inputs),
            "OUTPUT": format_vars([eval_input.output]),
            "EVALUATION_CRITERIA": self.eval.criteria,
            "RUBRIC": format_rubric(self.eval.rubric),
        }
        return format_user_prompt(prompt_variables)

    def evaluate(self, eval_input: EvalInput, save_results: bool = False) -> EvalOutput:
        """Evaluate a single EvalInput object."""
        try:
            self.eval.validate_input(eval_input)
            prompt = self._format_prompt(eval_input)
            response = self.model._generate(prompt)
            eval_output = EvalOutput.parse(response)
            if save_results:
                self._save_results([eval_input], [eval_output])
            return eval_output
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def batch_evaluate(
        self,
        eval_inputs: list[EvalInput],
        use_tqdm: bool = True,
        save_results: bool = True,
        fail_on_parse_error: bool = False,
    ) -> list[EvalOutput]:
        """Batch evaluate a list of EvalInput objects."""
        for eval_input in eval_inputs:
            self.eval.validate_input(eval_input)

        prompts = [self._format_prompt(eval_input) for eval_input in eval_inputs]
        responses = self.model._batch_generate(prompts, use_tqdm=use_tqdm)
        eval_outputs = [
            EvalOutput.parse(response, fail_on_parse_error=fail_on_parse_error)
            for response in responses
        ]
        parse_failures = sum(1 for output in eval_outputs if output.score == -1)
        if save_results:
            self._save_results(eval_inputs, eval_outputs)
        if parse_failures > 0:
            logger.warning(f"Number of parsing failures: {parse_failures} out of {len(responses)}")

        return eval_outputs


class AsyncLMEvaluator(AsyncBaseEvaluator):
    """Asynchronous evaluator using LLM-based evaluation specifications."""

    def __init__(
        self,
        eval: LMEval,
        model: AsyncBaseEvaluatorModel,
        output_dir: str | None = "output/",
    ):
        """Initialize AsyncLMEvaluator with an eval specification and model."""
        super().__init__(output_dir)
        self.eval = eval
        self.model = model

    # TODO - How to handle expected output?
    def _format_prompt(self, eval_input: EvalInput) -> str:
        """Format the prompt for a single evaluation input."""
        prompt_variables = {
            "INPUTS": format_vars(eval_input.inputs),
            "OUTPUT": format_vars([eval_input.output]),
            "EVALUATION_CRITERIA": self.eval.criteria,
            "RUBRIC": format_rubric(self.eval.rubric),
        }
        return format_user_prompt(prompt_variables)

    def _handle_batch_result(
        self, batch_result: BatchResult, batch_len: int, fail_on_parse_error: bool
    ) -> list[EvalOutput]:
        """Handle output parsing for batched results.

        Args:
            batch_result: The result of the batch from Baseten.
            batch_len: The initial batch size derived from the length of Eval Inputs.
            fail_on_parse_error: Flag to raise a parse error for the EvalOutput.

        Returns:
            list[EvalOutput]: A list of eval outputs with score and feedback.

        Note:
            There might be instances when downstream errors result in missing entries
            for the eval outputs. We implement retry strategies where we can, but in
            certain instances (such as network failures) errors are inevitable.
            To ascertain predictability, we 'fill-in' the errors with empty EvalOutputs.
        """
        eval_outputs = [EvalOutput(feedback="BasetenError", score=None)] * batch_len
        for output in batch_result.successful_outputs:
            index = output.get("index")
            eval_outputs[index - 1] = EvalOutput.parse(
                response=output["response"], fail_on_parse_error=fail_on_parse_error
            )

        # Log all downstream errors
        if len(batch_result.errors) > 0:
            logger.warning(
                f"Number of Baseten API errors: {len(batch_result.errors)}"
                f" of {batch_result.total_requests}."
                f" Success rate is {batch_result.success_rate}"
                " List of errors: "
            )
            for error in batch_result.errors:
                logger.warning(f"{error.error_type}: {error.error_message}")

        return eval_outputs

    async def async_evaluate(
        self, eval_input: EvalInput, save_results: bool = False, append: bool = False
    ) -> EvalOutput | None:
        """Evaluate a single EvalInput object asynchronously."""
        try:
            self.eval.validate_input(eval_input)
            prompt = self._format_prompt(eval_input)
            result = await self.model._async_generate(prompt)

            if isinstance(result, EvaluatorError):
                logger.error(f" {result.error_type}: {result.error_message}")
                return

            eval_output = EvalOutput.parse(result)
            if save_results:
                logger.info(f"Saving result {'(append)' if append else '(overwrite)'}")
                await asyncio.to_thread(
                    self._save_results, [eval_input], [eval_output], append=append
                )
            return eval_output
        except Exception as e:
            logger.error(f"Asynchronous evaluation failed: {e}")
            raise

    # TODO: figure if we want to have the parser be passed the fail_on_parse_error flag
    async def async_batch_evaluate(
        self,
        eval_inputs: list[EvalInput],
        use_tqdm: bool = True,
        save_results: bool = True,
        append: bool = False,
        fail_on_parse_error: bool = False,
    ) -> list[EvalOutput]:
        """Batch evaluate a list of EvalInput objects asynchronously."""
        for eval_input in eval_inputs:
            self.eval.validate_input(eval_input)

        prompts = [self._format_prompt(eval_input) for eval_input in eval_inputs]
        batch_result = await self.model._async_batch_generate(prompts, use_tqdm=use_tqdm)

        if isinstance(batch_result, BatchResult):
            eval_outputs = self._handle_batch_result(
                batch_result=batch_result,
                batch_len=len(eval_inputs),
                fail_on_parse_error=fail_on_parse_error,
            )
        else:
            eval_outputs = [
                EvalOutput.parse(response, fail_on_parse_error=fail_on_parse_error)
                for response in batch_result
            ]
        parse_failures = sum(1 for output in eval_outputs if output.score and output.score == -1)

        if save_results:
            logger.info(f"Saving {len(eval_outputs)} results")
            for i, (eval_input, eval_output) in enumerate(
                zip(eval_inputs, eval_outputs, strict=True)
            ):
                await asyncio.to_thread(
                    self._save_results,
                    [eval_input],
                    [eval_output],
                    append=(append or i > 0),  # Append for all but the first, unless append is True
                )

        if parse_failures > 0:
            logger.warning(
                f"Number of parsing failures: {parse_failures} out of {len(eval_outputs)}"
            )

        return eval_outputs
