import os
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pytest import LogCaptureFixture, MonkeyPatch

from flow_eval.models.common import ModelType, OpenAIGenerationParams
from flow_eval.models.openai import OpenAIModel, OpenAIModelConfig, OpenAIModelError


@pytest.fixture
def valid_generation_params() -> OpenAIGenerationParams:
    """Fixture for creating valid OpenAIGenerationParams instance.

    :return: An OpenAIGenerationParams instance with default values
    """
    params = OpenAIGenerationParams()
    assert isinstance(params, OpenAIGenerationParams)
    return params


@given(
    generation_params=st.builds(OpenAIGenerationParams),
    temperature=st.floats(min_value=0.0, max_value=2.0),
    max_tokens=st.integers(min_value=1, max_value=4096),
    top_p=st.floats(min_value=0.0, max_value=1.0),
)
def test_openai_model_config_init_valid(
    generation_params: OpenAIGenerationParams,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> None:
    """Test initialization of OpenAIModelConfig with valid parameters.

    :param generation_params: Generated OpenAIGenerationParams
    :param temperature: Temperature parameter for generation
    :param max_tokens: Maximum tokens for generation
    :param top_p: Top-p parameter for generation
    """
    try:
        config = OpenAIModelConfig(
            generation_params=generation_params,
            exec_async=False,
            _model_id="gpt-4o",
        )
    except Exception as e:
        pytest.fail(f"OpenAIModelConfig initialization failed unexpectedly: {str(e)}")

    assert isinstance(config, OpenAIModelConfig)
    assert config.model_type == ModelType.OPENAI
    assert config.generation_params == generation_params
    assert config.kwargs["exec_async"] is False
    assert config.model_id == "gpt-4o"


@pytest.mark.parametrize(
    "invalid_input, expected_error",
    [
        (
            {"generation_params": {}},
            "generation_params must be an instance of OpenAIGenerationParams",
        ),
        (
            {"generation_params": None},
            "generation_params must be an instance of OpenAIGenerationParams",
        ),
        ({"_model_id": None}, "A model_id is required"),
    ],
)
def test_openai_model_config_init_invalid(
    invalid_input: dict[str, Any],
    expected_error: str,
    valid_generation_params: OpenAIGenerationParams,
) -> None:
    """Test initialization of OpenAIModelConfig with invalid parameters.

    :param invalid_input: Dictionary of invalid input parameters
    :param expected_error: Expected error message
    :param valid_generation_params: Fixture providing valid OpenAIGenerationParams
    """
    valid_params = {
        "generation_params": valid_generation_params,
        "exec_async": False,
        "_model_id": "gpt-4o",
    }
    test_params = {**valid_params, **invalid_input}

    with pytest.raises(ValueError, match=expected_error):
        OpenAIModelConfig(**test_params)


@pytest.mark.asyncio
async def test_openai_init_valid(monkeypatch: MonkeyPatch) -> None:
    """Test valid initializations of the OpenAI class.

    :param monkeypatch: pytest's monkeypatch fixture
    """
    # Test synchronous initialization
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        try:
            openai_sync = OpenAIModel(exec_async=False)
        except Exception as e:
            pytest.fail(f"Synchronous OpenAI initialization failed unexpectedly: {str(e)}")

        assert isinstance(openai_sync, OpenAIModel)
        assert openai_sync.config.model_id == "gpt-4o"
        assert openai_sync.config.model_type == ModelType.OPENAI
        assert openai_sync.config.kwargs["exec_async"] is False

        # Test asynchronous initialization
        try:
            openai_async = OpenAIModel(exec_async=True)
        except Exception as e:
            pytest.fail(f"Asynchronous OpenAI initialization failed unexpectedly: {str(e)}")

        assert isinstance(openai_async, OpenAIModel)
        assert openai_async.config.model_id == "gpt-4o"
        assert openai_async.config.model_type == ModelType.OPENAI_ASYNC
        assert openai_async.config.kwargs["exec_async"] is True

        # Test with custom model_id
        openai_custom = OpenAIModel(_model_id="gpt-3.5-turbo")
        assert openai_custom.config.model_id == "gpt-3.5-turbo"


def test_openai_format_conversation() -> None:
    """Test the _format_conversation method of the OpenAI class."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        openai = OpenAIModel()

        prompt = "Hello, world!"
        formatted = openai._format_conversation(prompt)
        assert formatted == [{"role": "user", "content": "Hello, world!"}]

        prompt_with_whitespace = "  Hello, world!  "
        formatted = openai._format_conversation(prompt_with_whitespace)
        assert formatted == [{"role": "user", "content": "Hello, world!"}]


@pytest.mark.asyncio
async def test_openai_generate_methods(caplog: LogCaptureFixture) -> None:
    """Test the generate methods of the OpenAI class.

    :param caplog: pytest's log capture fixture
    """
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_key"}):
        openai = OpenAIModel()

        with patch.object(openai.api_adapter, "_fetch_response", return_value="Response"):
            result = openai._generate("Test prompt")
            assert result == "Response"

        with patch.object(
            openai.api_adapter, "_fetch_batched_response", return_value=["Response1", "Response2"]
        ):
            result = openai._batch_generate(["Prompt1", "Prompt2"])
            assert result == ["Response1", "Response2"]

        openai_async = OpenAIModel(exec_async=True)

        with patch.object(
            openai_async.api_adapter, "_async_fetch_response", return_value="Async Response"
        ):
            result = await openai_async._async_generate("Test prompt")
            assert result == "Async Response"

        with patch.object(
            openai_async.api_adapter,
            "_async_fetch_batched_response",
            return_value=["Async1", "Async2"],
        ):
            result = await openai_async._async_batch_generate(["Prompt1", "Prompt2"])
            assert result == ["Async1", "Async2"]


def test_openai_model_error() -> None:
    """Test the OpenAIModelError exception."""
    error = OpenAIModelError(status_code=404, message="Model not found")
    assert error.status_code == 404
    assert error.message == "Model not found"
    assert str(error) == "Model not found"


if __name__ == "__main__":
    pytest.main([__file__])
