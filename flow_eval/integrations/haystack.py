import logging
from typing import Any

import numpy as np
from haystack import component, default_from_dict, default_to_dict
from haystack.utils import deserialize_type

from flow_eval import LMEvaluator
from flow_eval.core import EvalInput, EvalOutput
from flow_eval.lm import LMEval
from flow_eval.lm.models.common import BaseEvaluatorModel

logger = logging.getLogger(__name__)

# Based on https://github.com/deepset-ai/haystack/blob/d234c75168dcb49866a6714aa232f37d56f72cab/haystack/components/evaluators/llm_evaluator.py#L354


@component
class HaystackLMEvaluator:
    """A component that uses LMEvaluator to evaluate inputs."""

    def __init__(
        self,
        eval: LMEval,
        model: BaseEvaluatorModel,
        output_dir: str = "output/",
        progress_bar: bool = True,
        raise_on_failure: bool = True,
        save_results: bool = True,
        fail_on_parse_error: bool = False,
    ):
        """Construct a new Evaluator evaluator."""
        if not isinstance(eval, LMEval):
            raise ValueError("Invalid eval type. Use LMEval.")

        if not isinstance(model, BaseEvaluatorModel):
            raise ValueError("Invalid model type. Use BaseEvaluatorModel or its subclasses.")

        self.model = model
        self.eval = eval
        self.output_dir = output_dir

        self.evaluator = LMEvaluator(eval=self.eval, model=self.model, output_dir=self.output_dir)

        # extract inputs and output from the metric
        self.inputs, self.outputs = self._extract_vars_from_eval(self.eval)
        self.validate_init_parameters(self.inputs, self.outputs)
        self.raise_on_failure = raise_on_failure
        self.progress_bar = progress_bar
        self.save_results = save_results
        self.fail_on_parse_error = fail_on_parse_error

        component.set_input_types(self, **dict(self.inputs))

    @staticmethod
    def _extract_vars_from_eval(
        eval: LMEval,
    ) -> tuple[list[tuple[str, type[list]]], list[str]]:
        """Extract the inputs to the component and its type from the eval.

        It also sets the output of the component.
        """
        eval_inputs_keys: list[str] = eval.input_columns
        eval_output_key: str = eval.output_column
        expected_output_key: list[str] = eval.expected_output_column

        inputs = [
            (key, list[str]) for key in eval_inputs_keys + [eval_output_key] + [expected_output_key]
        ]

        outputs = ["feedback", "score"]

        return inputs, outputs

    @staticmethod
    def validate_init_parameters(inputs: list[tuple[str, type[list]]], outputs: list[str]):
        """Validate the init parameters."""
        # Validate inputs
        if (
            not isinstance(inputs, list)
            or not all(isinstance(_input, tuple) for _input in inputs)
            or not all(
                isinstance(_input[0], str) and _input[1] is not list and len(_input) == 2
                for _input in inputs
            )
        ):
            msg = (
                f"Evaluator evaluator expects inputs to \
                be a list of tuples. Each tuple must contain an input name and "
                f"type of list but received {inputs}."
            )
            raise ValueError(msg)

        # Validate outputs
        if not isinstance(outputs, list) or not all(isinstance(output, str) for output in outputs):
            msg = f"Evaluator evaluator expects outputs \
                to be a list of str but received {outputs}."
            raise ValueError(msg)

    @component.output_types(
        results=list[dict[str, Any]],
        metadata=dict[str, Any],
        score=float,
        individual_scores=list[float],
    )
    def run(self, **inputs) -> dict[str, Any]:
        """Run the Evaluator evaluator on the provided inputs."""
        self._validate_input_parameters(dict(self.inputs), inputs)
        eval_inputs: list[EvalInput] = self._prepare_inputs(inputs=inputs, eval=self.eval)
        eval_outputs: list[EvalOutput] = self.evaluator.batch_evaluate(
            eval_inputs,
            save_results=self.save_results,
            fail_on_parse_error=self.fail_on_parse_error,
        )

        results: list[dict[str, Any] | None] = []
        parsing_errors = 0
        for eval_output in eval_outputs:
            if eval_output.score != -1:
                result = {
                    "feedback": eval_output.feedback,
                    "score": eval_output.score,
                }

                results.append(result)
            else:
                results.append({"feedback": eval_output.feedback, "score": eval_output.score})
                parsing_errors += 1

        if parsing_errors > 0:
            msg = (
                f"Evaluator failed to parse {parsing_errors} results out "
                f"of {len(eval_outputs)}. Score and Individual Scores are "
                "based on the successfully parsed results."
            )
            logger.warning(msg)

        metadata = self.model.metadata

        score = np.mean([result["score"] for result in results if result["score"] != -1])
        individual_scores = [float(result["score"]) for result in results if result["score"] != -1]

        return {
            "results": results,
            "metadata": metadata,
            "score": score,
            "individual_scores": individual_scores,
        }

    @staticmethod
    def _validate_input_parameters(expected: dict[str, Any], received: dict[str, Any]) -> None:
        """Validate the input parameters."""
        # Validate that all expected inputs are present in the received inputs
        for param in expected.keys():
            if param not in received:
                msg = f"Evaluator evaluator expected input \
                    parameter '{param}' but received only {received.keys()}."
                raise ValueError(msg)

        # Validate that all received inputs are lists
        if not all(isinstance(_input, list) for _input in received.values()):
            msg = (
                "Evaluator evaluator expects all input values to be lists but received "
                f"{[type(_input) for _input in received.values()]}."
            )
            raise ValueError(msg)

        # Validate that all received inputs are of the same length
        inputs = received.values()
        length = len(next(iter(inputs)))
        if not all(len(_input) == length for _input in inputs):
            msg = (
                f"Evaluator evaluator expects all input lists\
                    to have the same length but received {inputs} with lengths "
                f"{[len(_input) for _input in inputs]}."
            )
            raise ValueError(msg)

    @staticmethod
    def _prepare_inputs(inputs: dict[str, Any], eval: LMEval) -> list[EvalInput]:
        """Prepare the inputs for the LMEvaluator."""
        eval_inputs = []
        num_samples = len(next(iter(inputs.values())))

        for i in range(num_samples):
            input_list = []
            output_dict = {}
            expected_output_dict = {}
            for key, value_list in inputs.items():
                temp_dict = {}
                if key in eval.input_columns:
                    temp_dict[key] = value_list[i]
                    input_list.append(temp_dict)
                elif key == eval.output_column:
                    output_dict[key] = value_list[i]
                elif key == eval.expected_output_column:
                    expected_output_dict[key] = value_list[i]

            if not output_dict:
                raise ValueError(f"Required output '{eval.output_column}' not found in inputs.")

            eval_input = EvalInput(
                inputs=input_list, output=output_dict, expected_output=expected_output_dict
            )
            eval_inputs.append(eval_input)

        return eval_inputs

    def to_dict(self) -> dict[str, Any]:
        """Serialize this component to a dictionary."""
        return default_to_dict(
            self,
            eval=self.eval,
            model=self.model,
            output_dir=self.output_dir,
            progress_bar=self.progress_bar,
            raise_on_failure=self.raise_on_failure,
            save_results=self.save_results,
            fail_on_parse_error=self.fail_on_parse_error,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HaystackLMEvaluator":
        """Deserialize this component from a dictionary."""
        data["init_parameters"]["inputs"] = [
            (name, deserialize_type(type_)) for name, type_ in data["init_parameters"]["inputs"]
        ]

        return default_from_dict(cls, data)
