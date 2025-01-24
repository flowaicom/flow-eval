from flow_eval.lm.metrics import RESPONSE_CORRECTNESS_BINARY
from flow_eval.lm.types import LMEval, RubricItem


def test_response_correctness_binary():
    """Test the RESPONSE_CORRECTNESS_BINARY metric."""
    eval = RESPONSE_CORRECTNESS_BINARY
    assert eval.name == "Response Correctness (Binary)"
    assert len(eval.rubric) == 2
    assert eval.input_columns == ["query", "reference_answer"]
    assert eval.output_column == "response"


def test_custom_metric():
    """Test the CustomMetric class."""
    eval = LMEval(
        name="Test Metric",
        criteria="Test criteria",
        rubric=[RubricItem(score=0, description="Bad"), RubricItem(score=1, description="Good")],
        input_columns=["test_input"],
        output_column="test_output",
    )
    assert eval.name == "Test Metric"
    assert eval.criteria == "Test criteria"
    assert len(eval.rubric) == 2
    assert eval.input_columns == ["test_input"]
    assert eval.output_column == "test_output"
