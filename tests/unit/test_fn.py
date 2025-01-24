import pytest

from flow_eval.core import EvalInput, EvalOutput
from flow_eval.fn import FunctionEvaluator


def exact_match(response: str, reference: str) -> bool:
    """Test function that checks if response matches reference exactly."""
    return response == reference


def score_length(text: str) -> int:
    """Test function that returns the length of text."""
    return len(text)


def test_function_evaluator_initialization():
    """Test FunctionEvaluator initialization with valid functions."""
    # Test with boolean return type
    evaluator = FunctionEvaluator(exact_match)
    assert isinstance(evaluator, FunctionEvaluator)
    assert evaluator.fn == exact_match
    assert evaluator.param_names == ["response", "reference"]

    # Test with integer return type
    evaluator = FunctionEvaluator(score_length)
    assert isinstance(evaluator, FunctionEvaluator)
    assert evaluator.fn == score_length
    assert evaluator.param_names == ["text"]


def test_function_evaluator_invalid_function():
    """Test FunctionEvaluator initialization with invalid functions."""

    # Test with function having no type hints
    def no_type_hints(x):
        return True

    with pytest.raises(ValueError):
        FunctionEvaluator(no_type_hints)

    # Test with function having non-string parameters
    def invalid_param_type(x: int) -> bool:
        return x > 0

    with pytest.raises(ValueError):
        FunctionEvaluator(invalid_param_type)

    # Test with function having invalid return type
    def invalid_return_type(x: str) -> list:
        return [x]

    with pytest.raises(ValueError):
        FunctionEvaluator(invalid_return_type)


def test_function_evaluator_evaluate():
    """Test the evaluate method of FunctionEvaluator."""
    evaluator = FunctionEvaluator(exact_match)

    # Test with matching strings
    eval_input = EvalInput(
        inputs=[], output={"response": "test"}, expected_output={"reference": "test"}
    )
    result = evaluator.evaluate(eval_input)
    assert isinstance(result, EvalOutput)
    assert result.score is True

    # Test with non-matching strings
    eval_input = EvalInput(
        inputs=[], output={"response": "test"}, expected_output={"reference": "different"}
    )
    result = evaluator.evaluate(eval_input)
    assert isinstance(result, EvalOutput)
    assert result.score is False


def test_function_evaluator_batch_evaluate():
    """Test the batch_evaluate method of FunctionEvaluator."""
    evaluator = FunctionEvaluator(exact_match)

    eval_inputs = [
        EvalInput(inputs=[], output={"response": "test1"}, expected_output={"reference": "test1"}),
        EvalInput(
            inputs=[], output={"response": "test2"}, expected_output={"reference": "different"}
        ),
    ]

    results = evaluator.batch_evaluate(eval_inputs, save_results=False)
    assert len(results) == 2
    assert results[0].score is True
    assert results[1].score is False


def test_function_evaluator_arg_extraction():
    """Test argument extraction from different parts of EvalInput."""
    evaluator = FunctionEvaluator(exact_match)

    # Test extraction from output and expected_output
    eval_input = EvalInput(
        inputs=[], output={"response": "test"}, expected_output={"reference": "test"}
    )
    result = evaluator.evaluate(eval_input)
    assert result.score is True

    # Test extraction from inputs
    eval_input = EvalInput(
        inputs=[{"response": "test"}, {"reference": "test"}], output={}, expected_output={}
    )
    result = evaluator.evaluate(eval_input)
    assert result.score is True

    # Test with missing argument
    eval_input = EvalInput(inputs=[], output={"response": "test"}, output_type="text")
    with pytest.raises(ValueError):
        evaluator.evaluate(eval_input)


@pytest.mark.parametrize("save_results", [True, False])
def test_function_evaluator_save_results(tmp_path, save_results):
    """Test saving results in the evaluate and batch_evaluate methods."""
    evaluator = FunctionEvaluator(exact_match, output_dir=str(tmp_path))
    eval_input = EvalInput(
        inputs=[], output={"response": "test"}, expected_output={"reference": "test"}
    )

    # Test single evaluation
    result = evaluator.evaluate(eval_input, save_results=save_results)
    assert isinstance(result, EvalOutput)

    # Test batch evaluation
    results = evaluator.batch_evaluate([eval_input], save_results=save_results)
    assert len(results) == 1
    assert isinstance(results[0], EvalOutput)
