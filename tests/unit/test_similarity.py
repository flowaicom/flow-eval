import pytest
import torch
from sentence_transformers import SentenceTransformer

from flow_eval.core import EvalInput, EvalOutput
from flow_eval.similarity import AnswerSimilarityEvaluator


@pytest.fixture
def evaluator():
    """Create a test evaluator instance."""
    return AnswerSimilarityEvaluator(
        model_name="all-mpnet-base-v2",
        model_type="sentence-transformer",
        similarity_fn_name="cosine",
        output_dir=None,
    )


def test_initialization():
    """Test evaluator initialization with different parameters."""
    # Test default initialization
    evaluator = AnswerSimilarityEvaluator()
    assert evaluator.model_name == "all-mpnet-base-v2"
    assert evaluator.model_type == "sentence-transformer"
    assert evaluator.eval_name == "embedding_similarity"

    # Test custom initialization
    evaluator = AnswerSimilarityEvaluator(
        model_name="multi-qa-mpnet-base-dot-v1 ",
        model_type="sentence-transformer",
        eval_name="custom_similarity",
        similarity_fn_name="dot",
    )
    assert evaluator.model_name == "multi-qa-mpnet-base-dot-v1 "
    assert evaluator.eval_name == "custom_similarity"


def test_model_lazy_loading(evaluator):
    """Test that model is lazily loaded."""
    assert evaluator._model is None
    _ = evaluator.model  # Access model property to trigger loading
    assert isinstance(evaluator._model, SentenceTransformer)
    # Check if model is on the correct device type (cuda or cpu)
    expected_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    assert str(evaluator._model.device).startswith(expected_device_type)


def test_extract_texts(evaluator):
    """Test text extraction from EvalInput."""
    # Test valid input
    eval_input = EvalInput(
        inputs=[],
        output={"response": "This is a test response"},
        expected_output={"reference": "This is a reference response"},
    )
    text1, text2 = evaluator._extract_texts(eval_input)
    assert text1 == "This is a test response"
    assert text2 == "This is a reference response"

    # Test empty output
    with pytest.raises(ValueError, match="EvalInput.output is empty"):
        evaluator._extract_texts(EvalInput(inputs=[], output={}, expected_output={"ref": "test"}))

    # Test empty expected_output
    with pytest.raises(ValueError, match="EvalInput.expected_output is empty"):
        evaluator._extract_texts(EvalInput(inputs=[], output={"out": "test"}, expected_output={}))


def test_compute_similarity(evaluator):
    """Test similarity computation between texts."""
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A fast brown fox leaps above a sleepy canine"

    similarity = evaluator._compute_similarity(text1, text2)
    assert isinstance(similarity, float)
    assert 0 <= similarity <= 1  # Cosine similarity should be between 0 and 1

    # Test identical texts should have high similarity
    identical_similarity = evaluator._compute_similarity(text1, text1)
    assert identical_similarity > 0.9  # Should be very close to 1

    # Test completely different texts should have lower similarity
    different_text = "This is a completely unrelated sentence about mathematics"
    different_similarity = evaluator._compute_similarity(text1, different_text)
    assert different_similarity < identical_similarity


def test_check_text_lengths(evaluator, caplog):
    """Test text length checking functionality."""
    # Create a very long text that exceeds model's max sequence length
    long_text = "test " * 512  # Should exceed most models' max sequence length

    evaluator._check_text_lengths(long_text, "short text")
    # Check that the warning message contains the key phrases
    assert "exceeds model's maximum sequence length" in caplog.text
    assert "tokens" in caplog.text
    assert "https://www.sbert.net/docs/pretrained_models.html" in caplog.text


def test_batch_compute_similarity(evaluator):
    """Test batch similarity computation."""
    texts1 = [
        "The quick brown fox jumps over the lazy dog",
        "Hello world",
        "Python is a programming language",
    ]
    texts2 = [
        "A fast brown fox leaps above a sleepy canine",
        "Hi world",
        "Python is great for coding",
    ]

    similarities = evaluator._compute_batch_similarity(texts1, texts2)
    assert len(similarities) == 3
    assert all(isinstance(sim, float) for sim in similarities)
    assert all(0 <= sim <= 1 for sim in similarities)


def test_evaluate(evaluator):
    """Test single evaluation."""
    eval_input = EvalInput(
        inputs=[],
        output={"response": "The quick brown fox jumps over the lazy dog"},
        expected_output={"reference": "A fast brown fox leaps above a sleepy canine"},
    )

    result = evaluator.evaluate(eval_input)
    assert isinstance(result, EvalOutput)
    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0


def test_batch_evaluate(evaluator):
    """Test batch evaluation."""
    eval_inputs = [
        EvalInput(
            inputs=[],
            output={"response": "Hello world"},
            expected_output={"reference": "Hi world"},
        ),
        EvalInput(
            inputs=[],
            output={"response": "Python programming"},
            expected_output={"reference": "Python coding"},
        ),
    ]

    results = evaluator.batch_evaluate(eval_inputs, save_results=False)
    assert len(results) == 2
    assert all(isinstance(result, EvalOutput) for result in results)
    assert all(isinstance(result.score, float) for result in results)
    assert all(0.0 <= result.score <= 1.0 for result in results)


def test_error_handling(evaluator):
    """Test error handling in evaluation."""
    # Test with invalid input structure
    with pytest.raises(ValueError):
        evaluator.evaluate(EvalInput(inputs=[], output={}, expected_output={}))

    # Test with empty input list for batch evaluation
    with pytest.raises(ValueError, match="Input list cannot be empty"):
        evaluator.batch_evaluate([])
