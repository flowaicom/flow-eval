import shutil
from unittest.mock import patch

import pytest

from flow_eval import LMEvaluator
from flow_eval.core import EvalInput, EvalOutput
from flow_eval.lm.metrics import RESPONSE_CORRECTNESS_BINARY
from flow_eval.lm.models.common import BaseEvaluatorModel
from flow_eval.lm.prompts import USER_PROMPT_TEMPLATE, format_rubric, format_vars
from flow_eval.lm.types import LMEval, RubricItem


class MockLMEvaluatorModel(BaseEvaluatorModel):
    """Mock model for testing."""

    def __init__(self, model_id, model_type, generation_params):
        """Initialize the mock model."""
        super().__init__(model_id, model_type, generation_params)

    def _generate(self, prompt):
        """Generate a mock response."""
        return "<feedback>Test feedback</feedback>\n<score>1</score>"

    def _batch_generate(self, prompts, use_tqdm=True):
        """Generate mock responses for a list of prompts."""
        return ["<feedback>Test feedback</feedback>\n<score>1</score>" for _ in prompts]

    def generate(self, prompt):
        """Generate a mock response."""
        return self._generate(prompt)

    def batch_generate(self, prompts, use_tqdm=True):
        """Generate mock responses for a list of prompts."""
        return self._batch_generate(prompts, use_tqdm=use_tqdm)


@pytest.fixture
def mock_model():
    """Fixture to create a mock model for testing."""
    return MockLMEvaluatorModel("test-model", "mock", {"temperature": 0.7})


def test_flow_eval_initialization(mock_model):
    """Test the initialization of LMEvaluator."""
    evaluator = LMEvaluator(eval=RESPONSE_CORRECTNESS_BINARY, model=mock_model)
    assert isinstance(evaluator, LMEvaluator)
    assert evaluator.eval == RESPONSE_CORRECTNESS_BINARY
    assert evaluator.model == mock_model


def test_flow_eval_initialization_invalid_metric():
    """Test LMEvaluator initialization with invalid metric."""
    with pytest.raises(ValueError):
        LMEvaluator(eval="invalid_metric", model=mock_model)


def test_flow_eval_evaluate(mock_model):
    """Test the evaluate method of LMEvaluator."""
    evaluator = LMEvaluator(eval=RESPONSE_CORRECTNESS_BINARY, model=mock_model)
    eval_input = EvalInput(
        inputs=[{"query": "Test query"}, {"reference_answer": "Test reference"}],
        output={"response": "Test response"},
    )
    result = evaluator.evaluate(eval_input)
    assert isinstance(result, EvalOutput)
    assert result.feedback == "Test feedback"
    assert result.score == 1


def test_flow_eval_batch_evaluate(mock_model):
    """Test the batch_evaluate method of LMEvaluator."""
    evaluator = LMEvaluator(eval=RESPONSE_CORRECTNESS_BINARY, model=mock_model)
    eval_inputs = [
        EvalInput(
            inputs=[{"query": "Test query 1"}, {"reference_answer": "Test reference 1"}],
            output={"response": "Test response 1"},
        ),
        EvalInput(
            inputs=[{"query": "Test query 2"}, {"reference_answer": "Test reference 2"}],
            output={"response": "Test response 2"},
        ),
    ]
    results = evaluator.batch_evaluate(eval_inputs, save_results=False)
    assert len(results) == 2
    for result in results:
        assert isinstance(result, EvalOutput)
        assert result.feedback == "Test feedback"
        assert result.score == 1


@pytest.mark.parametrize("save_results", [True, False])
def test_flow_eval_evaluate_save_results(mock_model, tmp_path, save_results):
    """Test saving results in the batch_evaluate method."""
    evaluator = LMEvaluator(
        eval=RESPONSE_CORRECTNESS_BINARY, model=mock_model, output_dir=str(tmp_path)
    )
    eval_input = EvalInput(
        inputs=[{"query": "Test query"}, {"reference_answer": "Test reference"}],
        output={"response": "Test response"},
    )
    with patch.object(LMEvaluator, "_save_results") as mock_save:
        evaluator.batch_evaluate([eval_input], save_results=save_results)
        if save_results:
            mock_save.assert_called_once()
        else:
            mock_save.assert_not_called()


def test_custom_metric():
    """Test creating and using a custom metric."""
    custom_metric = LMEval(
        name="custom_metric",
        criteria="Custom criteria",
        rubric=[RubricItem(score=0, description="Bad"), RubricItem(score=1, description="Good")],
        input_columns=["custom_input"],
        output_column="custom_output",
    )
    assert custom_metric.name == "custom_metric"
    assert custom_metric.criteria == "Custom criteria"
    assert len(custom_metric.rubric) == 2
    assert custom_metric.input_columns == ["custom_input"]
    assert custom_metric.output_column == "custom_output"


def test_eval_input_validation(mock_model):
    """Test EvalInput validation."""
    evaluator = LMEvaluator(eval=RESPONSE_CORRECTNESS_BINARY, model=mock_model)

    # Valid input
    valid_input = EvalInput(
        inputs=[{"query": "Test query"}, {"reference_answer": "Test reference"}],
        output={"response": "Test response"},
    )
    assert evaluator.evaluate(valid_input)

    # Invalid input - missing required input
    invalid_input = EvalInput(
        inputs=[{"query": "Test query"}], output={"response": "Test response"}
    )
    with pytest.raises(ValueError):
        evaluator.evaluate(invalid_input)

    # Invalid input - wrong output key
    invalid_output = EvalInput(
        inputs=[{"query": "Test query"}, {"reference_answer": "Test reference"}],
        output={"wrong_key": "Test response"},
    )
    with pytest.raises(ValueError):
        evaluator.evaluate(invalid_output)


def test_format_vars():
    """Test format_vars function."""
    variables = [{"question": "What is 2+2?"}, {"context": "Math basics"}]
    formatted = format_vars(variables)
    expected = """<question>
What is 2+2?
</question>
<context>
Math basics
</context>"""
    assert expected == formatted


def test_format_rubric():
    """Test format_rubric function."""
    rubric = [RubricItem(score=1, description="Good"), RubricItem(score=0, description="Poor")]
    formatted = format_rubric(rubric)
    expected = """- Score 0: Poor
- Score 1: Good"""
    assert expected == formatted


def test_format_prompt(mock_model):
    """Test LMEvaluator._format_prompt."""
    eval_input = EvalInput(
        inputs=[{"query": "Test query"}, {"reference_answer": "Test reference"}],
        output={"response": "Test response"},
    )

    evaluator = LMEvaluator(eval=RESPONSE_CORRECTNESS_BINARY, model=mock_model)
    prompt = evaluator._format_prompt(eval_input)

    expected_prompt = USER_PROMPT_TEMPLATE.format(
        INPUTS=format_vars(eval_input.inputs),
        OUTPUT=format_vars([eval_input.output]),
        EVALUATION_CRITERIA=RESPONSE_CORRECTNESS_BINARY.criteria,
        RUBRIC=format_rubric(RESPONSE_CORRECTNESS_BINARY.rubric),
    )
    assert prompt == expected_prompt


def test_eval_output_parse_fail_on_parse_error():
    """Test EvalOutput.parse with fail_on_parse_error."""
    # Invalid response without proper tags
    invalid_response = "This is an invalid response without proper tags"

    # Test with fail_on_parse_error=False (default behavior)
    result = EvalOutput.parse(invalid_response)
    assert isinstance(result, EvalOutput)
    assert result.feedback == "Error"
    assert result.score == -1

    # Test with fail_on_parse_error=True
    with pytest.raises(ValueError):
        EvalOutput.parse(invalid_response, fail_on_parse_error=True)

    # Test with valid response
    valid_response = "<feedback>Good job!</feedback><score>5</score>"
    result = EvalOutput.parse(valid_response)
    assert isinstance(result, EvalOutput)
    assert result.feedback == "Good job!"
    assert result.score == 5


@pytest.fixture(autouse=True)
def cleanup(request, tmp_path):
    """Cleanup files and directories created during the test."""
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_validate_eval_input():
    """Test the validate_eval_input function."""
    eval = LMEval(
        name="Test Metric",
        criteria="Test criteria",
        rubric=[RubricItem(score=0, description="Bad"), RubricItem(score=1, description="Good")],
        input_columns=["test_input"],
        output_column="test_output",
    )
    valid_input = EvalInput(
        inputs=[{"test_input": "Test value"}], output={"test_output": "Test output"}
    )
    eval.validate_input(valid_input)  # Should not raise an exception

    invalid_input = EvalInput(
        inputs=[{"wrong_input": "Test value"}], output={"test_output": "Test output"}
    )
    with pytest.raises(ValueError):
        eval.validate_input(invalid_input)
