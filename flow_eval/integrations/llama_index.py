import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from llama_index.core.evaluation import BaseEvaluator, EvaluationResult

from flow_eval import AsyncLMEvaluator
from flow_eval.core import EvalInput
from flow_eval.lm import LMEval
from flow_eval.lm.models.common import AsyncBaseEvaluatorModel

logger = logging.getLogger(__name__)


class LlamaIndexLMEvaluator(BaseEvaluator):
    """A custom evaluator for LlamaIndex that uses Evaluator to evaluate RAG system performance.

    This class integrates Evaluator with LlamaIndex's evaluation framework, allowing for
    seamless evaluation of retrieval-augmented generation (RAG) systems using custom metrics
    and models.

    Attributes:
        eval (LMEval): The evaluation metric to be used.
        model (AsyncBaseEvaluatorModel): The model used for evaluation.
        output_dir (str): Directory to save evaluation results.
        save_results (bool): Whether to save evaluation results to disk.

    Raises:
        ValueError: If invalid eval or model types are provided.
    """

    def __init__(
        self,
        eval: LMEval,
        model: AsyncBaseEvaluatorModel,
        output_dir: str = "output/",
        save_results: bool = False,
    ) -> None:
        """Initialize the LlamaIndexEvaluator.

        Args:
            eval: The evaluation metric to be used.
            model: The model used for evaluation.
            output_dir: Directory to save evaluation results. Defaults to "output/".
            save_results: Whether to save evaluation results to disk. Defaults to False.

        Raises:
            ValueError: If invalid eval or model types are provided.
        """
        if not isinstance(eval, LMEval):
            raise ValueError("Invalid eval type. Use LMEval.")
        self.eval = eval

        if not isinstance(model, AsyncBaseEvaluatorModel):
            raise ValueError("Invalid model type. Use AsyncBaseEvaluatorModel or its subclasses.")
        self.model = model

        self.output_dir = output_dir
        self.save_results = save_results

        self.evaluator = AsyncLMEvaluator(
            eval=self.eval, model=self.model, output_dir=self.output_dir
        )

    def _get_prompts(self):
        """Get the prompts for the flow judge."""
        pass

    def _update_prompts(self):
        """Update the prompts for the flow judge."""
        pass

    # aevaluate naming required
    async def aevaluate(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: Sequence[str] | None = None,
        reference: str | None = None,
        sleep_time_in_seconds: int = 0,
        save_results: bool | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate the performance of a model asynchronously.

        This method evaluates a single query-response pair, optionally considering contexts
        and a reference answer.

        Args:
            query: The input query to evaluate.
            response: The model's response to evaluate.
            contexts: Relevant context information for the query.
            reference: The reference answer for comparison.
            sleep_time_in_seconds: Time to sleep before evaluation (for rate limiting).
            save_results: Whether to save this evaluation result.
            Overrides instance setting if provided.
            **kwargs: Additional keyword arguments.

        Returns:
            An EvaluationResult containing the evaluation feedback and score.

        Raises:
            ValueError: If required inputs for the eval are not provided.

        Note:
            At least one of query, response, contexts, or reference must be provided.
        """
        await asyncio.sleep(sleep_time_in_seconds)

        save_results = save_results if save_results is not None else self.save_results

        try:
            available_data = self._prepare_available_data(query, response, contexts, reference)
            eval_input = self._create_eval_input(available_data)
            eval_output = await self.evaluator.async_evaluate(eval_input, save_results=save_results)
            return EvaluationResult(
                query=query,
                response=response,
                contexts=contexts,
                feedback=eval_output.feedback,
                score=eval_output.score,
                invalid_result=False,
                invalid_reason=None,
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return EvaluationResult(
                query=query,
                response=response,
                contexts=contexts,
                feedback=None,
                score=None,
                invalid_result=True,
                invalid_reason=str(e),
            )

    def _prepare_available_data(
        self,
        query: str | None = None,
        response: str | None = None,
        contexts: Sequence[str] | None = None,
        reference: str | None = None,
    ) -> dict[str, Any]:
        """Prepare available data for evaluation.

        This method collects and formats the provided data into a dictionary for further processing.

        Args:
            query: The input query.
            response: The model's response.
            contexts: Relevant context information.
            reference: The reference answer.

        Returns:
            A dictionary containing the available data.

        Raises:
            ValueError: If no data is provided (all arguments are None).

        Note:
            Contexts, if provided, are joined into a single string with double newline separators.
        """
        available_data: dict[str, Any] = {}
        if query is not None:
            available_data["query"] = query
        if response is not None:
            available_data["response"] = response
        if contexts is not None:
            available_data["contexts"] = "\n\n".join(contexts)
        if reference is not None:
            available_data["reference"] = reference

        if not available_data:
            raise ValueError(
                "At least one of query, response, contexts, or reference must be provided"
            )
        return available_data

    def _create_eval_input(self, available_data: dict[str, Any]) -> EvalInput:
        """Create an EvalInput object from available data.

        This method constructs an EvalInput object based on the eval's required inputs and output.

        Args:
            available_data: A dictionary containing the available data for evaluation.

        Returns:
            An EvalInput object ready for evaluation.

        Raises:
            ValueError: If any required input or output for the eval is not available in the data.

        Note:
            The method strictly enforces that all required inputs
            and outputs for the eval are present.
        """
        inputs = []
        for required_input in self.eval.input_columns:
            if required_input not in available_data:
                raise ValueError(
                    f"Required input '{required_input}' is not available in this integration"
                )
            inputs.append({required_input: available_data[required_input]})

        if self.eval.output_column not in available_data:
            raise ValueError(
                f"Required output '{self.eval.output_column}' "
                "is not available in this integration"
            )

        output = {self.eval.output_column: available_data[self.eval.output_column]}

        if "reference" in available_data:
            return EvalInput(
                inputs=inputs,
                output=output,
                expected_output={"reference": available_data["reference"]},
            )
        else:
            return EvalInput(inputs=inputs, output=output)

    async def aclose(self) -> None:
        """Clean up resources asynchronously.

        This method should be called when the evaluator is no longer needed to ensure
        proper cleanup of resources, especially for the judge and model components.

        Note:
            It checks for the existence of 'aclose' and 'shutdown' methods before calling them,
            making it safe to call even if these methods are not implemented in all cases.
        """
        if hasattr(self.evaluator, "aclose") and callable(self.evaluator.aclose):
            await self.evaluator.aclose()
        if hasattr(self.model, "shutdown") and callable(self.model.shutdown):
            await asyncio.to_thread(self.model.shutdown)

    async def aevaluate_batch(
        self,
        queries: list[str],
        responses: list[str],
        contexts: list[Sequence[str]] | None = None,
        references: list[str] | None = None,
        sleep_time_in_seconds: int = 0,
        save_results: bool | None = None,
        **kwargs: Any,
    ) -> list[EvaluationResult]:
        """Evaluate a batch of query-response pairs asynchronously.

        This method processes multiple query-response pairs in a single batch, which can be more
        efficient than evaluating them individually.

        Args:
            queries: List of input queries to evaluate.
            responses: List of model responses to evaluate.
            contexts: List of relevant context information for each query.
            references: List of reference answers for comparison.
            sleep_time_in_seconds: Time to sleep before evaluation (for rate limiting).
            save_results: Whether to save these evaluation results.
            Overrides instance setting if provided.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of EvaluationResult objects, one for each query-response pair.

        Raises:
            ValueError: If the lengths of queries and responses don't match,
            or if required inputs are missing.

        Note:
            - The method logs progress and any errors encountered during batch processing.
            - Results are saved only after all evaluations are complete, if save_results is True.
            - Failed evaluations are included in the results list with appropriate
            error information.
        """
        await asyncio.sleep(sleep_time_in_seconds)

        if len(queries) != len(responses):
            raise ValueError("The number of queries and responses must be the same.")

        save_results = save_results if save_results is not None else self.save_results

        results = []
        eval_inputs = []
        eval_outputs = []
        logger.info(f"Processing {len(queries)} queries in aevaluate_batch")

        for i, (query, response) in enumerate(zip(queries, responses, strict=True)):
            context = contexts[i] if contexts else None
            reference = references[i] if references else None

            try:
                available_data = self._prepare_available_data(query, response, context, reference)
                eval_input = self._create_eval_input(available_data)
                eval_inputs.append(eval_input)

                logger.info(f"Evaluating query {i+1}/{len(queries)}")
                eval_output = await self.evaluator.async_evaluate(eval_input, save_results=False)
                eval_outputs.append(eval_output)

                logger.info(f"Finished evaluating query {i+1}/{len(queries)}")
                results.append(
                    EvaluationResult(
                        query=query,
                        response=response,
                        contexts=context,
                        feedback=eval_output.feedback,
                        score=eval_output.score,
                        invalid_result=False,
                        invalid_reason=None,
                    )
                )
            except Exception as e:
                logger.error(f"Evaluation failed for query {i+1}: {e}")
                results.append(
                    EvaluationResult(
                        query=query,
                        response=response,
                        contexts=context,
                        feedback=None,
                        score=None,
                        invalid_result=True,
                        invalid_reason=str(e),
                    )
                )

        if save_results:
            await asyncio.to_thread(
                self.evaluator._save_results,
                eval_inputs,
                eval_outputs,
                metadata=self.model.metadata,
                eval_name=self.eval.name,
                append=False,
            )

        logger.info(f"Collected {len(results)} results")
        return results
