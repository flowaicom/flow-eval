import os
from unittest.mock import AsyncMock, Mock, patch

import pytest
from openai import (
    APIConnectionError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAIError,
    RateLimitError,
    UnprocessableEntityError,
)

from flow_eval.models.adapters.openai.adapter import AsyncOpenAIAdapter, OpenAIAdapter


@pytest.fixture
def openai_api_adapter():
    """Fixture to create an OpenAIAdapter instance with test values.

    Yields:
        OpenAIAdapter: The adapter instance.
    """
    model_id = "gpt-4o"
    base_url = "https://api.openai.com/v1"
    key_var = "OPENAI_API_KEY"
    os.environ[key_var] = "test_api_key"
    adapter = OpenAIAdapter(model_id, base_url, key_var)
    yield adapter
    del os.environ[key_var]


@pytest.fixture
def async_openai_api_adapter():
    """Fixture to create an AsyncOpenAIAdapter instance with test values.

    Yields:
        AsyncOpenAIAdapter: The adapter instance.
    """
    model_id = "gpt-4o"
    base_url = "https://api.openai.com/v1"
    key_var = "OPENAI_API_KEY"
    os.environ[key_var] = "test_api_key"
    adapter = AsyncOpenAIAdapter(model_id, base_url, key_var)
    yield adapter
    del os.environ[key_var]


def test_openai_api_adapter_init(openai_api_adapter: OpenAIAdapter) -> None:
    """Test case to ensure the OpenAIAdapter instance is initialized correctly.

    Args:
        openai_api_adapter: The OpenAIAdapter instance to test.
    """
    assert openai_api_adapter.model_id == "gpt-4o"
    assert openai_api_adapter.base_url == "https://api.openai.com/v1"
    assert openai_api_adapter.api_key == "test_api_key"


def test_openai_api_adapter_init_missing_api_key() -> None:
    """Test case to ensure a ValueError.

    It is raised when creating an OpenAIAdapter instance without API key.
    """
    with pytest.raises(ValueError):
        OpenAIAdapter("gpt-4o", "https://api.openai.com/v1", "MISSING_KEY")


@patch("openai.OpenAI")
def test_make_request(mock_openai: Mock, openai_api_adapter: OpenAIAdapter) -> None:
    """Test case to ensure the _make_request method works as expected.

    Args:
        mock_openai: Mock for the OpenAI API.
        openai_api_adapter: The OpenAIAdapter instance to test.
    """
    message = {"role": "user", "content": "test content"}
    request_messages = [message]

    # Test OpenAI error
    mock_openai.return_value.chat.completions.create.side_effect = OpenAIError("test error")
    result = openai_api_adapter._make_request(request_messages)
    assert result is None

    # Test other exceptions
    mock_openai.return_value.chat.completions.create.side_effect = Exception("test exception")
    result = openai_api_adapter._make_request(request_messages)
    assert result is None


@patch("flow_eval.models.adapters.openai.adapter.OpenAIAdapter._make_request")
def test_fetch_response(mock_make_request: Mock, openai_api_adapter: OpenAIAdapter) -> None:
    """Test case to ensure the _fetch_response method works as expected.

    Args:
        mock_make_request: Mock for the _make_request method.
        openai_api_adapter: The OpenAIAdapter instance to test.
    """
    message = {"role": "user", "content": "test content"}
    request_messages = [message]

    # Test successful response
    mock_completion = Mock()
    mock_completion.choices = [
        Mock(message=Mock(content="This is a test response to your test content"))
    ]
    mock_make_request.return_value = mock_completion
    result = openai_api_adapter._fetch_response(request_messages)
    assert result == "This is a test response to your test content"

    # Test exception handling
    mock_make_request.return_value = None
    result = openai_api_adapter._fetch_response(request_messages)
    assert result == ""


def test_async_openai_api_adapter_init(async_openai_api_adapter: AsyncOpenAIAdapter) -> None:
    """Test case to ensure the AsyncOpenAIAdapter instance is initialized correctly.

    Args:
        async_openai_api_adapter: The AsyncOpenAIAdapter instance to test.
    """
    assert async_openai_api_adapter.model_id == "gpt-4o"
    assert async_openai_api_adapter.base_url == "https://api.openai.com/v1"
    assert async_openai_api_adapter.api_key == "test_api_key"


def test_async_openai_api_adapter_init_validation():
    """Test validation of AsyncOpenAIAdapter initialization parameters."""
    with pytest.raises(ValueError, match="model_id must be a non-empty string"):
        AsyncOpenAIAdapter("", "https://api.openai.com/v1", "OPENAI_API_KEY")

    with pytest.raises(ValueError, match="base_url must be a non-empty string"):
        AsyncOpenAIAdapter("gpt-4o", "", "OPENAI_API_KEY")

    with pytest.raises(ValueError, match="max_retries must be a non-negative integer"):
        AsyncOpenAIAdapter("gpt-4o", "https://api.openai.com/v1", "OPENAI_API_KEY", max_retries=-1)

    with pytest.raises(ValueError, match="request_timeout must be a positive number"):
        AsyncOpenAIAdapter(
            "gpt-4o", "https://api.openai.com/v1", "OPENAI_API_KEY", request_timeout=0
        )


@patch("openai.AsyncOpenAI")
async def test_async_make_request(
    mock_async_openai: Mock, async_openai_api_adapter: AsyncOpenAIAdapter
) -> None:
    """Test case to ensure the async _make_request method works as expected."""
    # Create a mock instance and attach it to the adapter
    mock_client = AsyncMock()
    mock_async_openai.return_value = mock_client
    async_openai_api_adapter.client = mock_client

    message = {"role": "user", "content": "test content"}
    request_messages = [message]

    # Mock successful completion
    mock_completion = Mock()
    mock_completion.choices = [Mock(message=Mock(content="Hello, world!"))]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

    result = await async_openai_api_adapter._make_request(request_messages)
    assert result == mock_completion

    # Test various OpenAI errors with proper initialization
    mock_request = Mock()  # Mock request object
    mock_response = Mock(status_code=500, headers={})
    mock_body = {"error": {"message": "Error message"}}

    error_cases = [
        (APIConnectionError, {"request": mock_request}),
        (
            RateLimitError,
            {"message": "Rate limit exceeded", "response": mock_response, "body": mock_body},
        ),
        (BadRequestError, {"message": "Bad request", "response": mock_response, "body": mock_body}),
        (NotFoundError, {"message": "Not found", "response": mock_response, "body": mock_body}),
        (
            UnprocessableEntityError,
            {"message": "Unprocessable entity", "response": mock_response, "body": mock_body},
        ),
        (
            InternalServerError,
            {"message": "Internal server error", "response": mock_response, "body": mock_body},
        ),
    ]

    for error_class, error_kwargs in error_cases:
        mock_client.chat.completions.create = AsyncMock(side_effect=error_class(**error_kwargs))
        with pytest.raises(error_class):
            await async_openai_api_adapter._make_request(request_messages)

    # Test generic exception
    mock_client.chat.completions.create = AsyncMock(side_effect=Exception("Unexpected error"))
    with pytest.raises((RuntimeError, ValueError, Exception)):
        await async_openai_api_adapter._make_request(request_messages)


@patch("flow_eval.models.adapters.openai.adapter.AsyncOpenAIAdapter._make_request")
async def test_async_fetch_response(
    mock_make_request: AsyncMock, async_openai_api_adapter: AsyncOpenAIAdapter
) -> None:
    """Test case to ensure the _async_fetch_response method works as expected.

    Args:
        mock_make_request: Mock for the _make_request method.
        async_openai_api_adapter: The AsyncOpenAIAdapter instance to test.
    """
    message = {"role": "user", "content": "test content"}
    request_messages = [message]

    # Test successful response
    mock_completion = Mock()
    mock_completion.choices = [Mock(message=Mock(content="Hello, world!"))]
    mock_make_request.return_value = mock_completion
    result = await async_openai_api_adapter._async_fetch_response(request_messages)
    assert result == "Hello, world!"


@patch("flow_eval.models.adapters.openai.adapter.AsyncOpenAIAdapter._make_request")
async def test_async_fetch_batched_response(
    mock_make_request: Mock, async_openai_api_adapter: AsyncOpenAIAdapter
) -> None:
    """Test case to ensure the _async_fetch_batched_response method works as expected.

    Args:
        mock_make_request: Mock for the _make_request method.
        async_openai_api_adapter: The AsyncOpenAIAdapter instance to test.
    """
    messages = [
        {"role": "user", "content": "test1"},
        {"role": "user", "content": "test2"},
        {"role": "user", "content": "test3"},
    ]

    # Test successful batch response
    mock_completion = Mock()
    mock_completion.choices = [Mock(message=Mock(content="response"))]
    mock_make_request.return_value = mock_completion
    result = await async_openai_api_adapter._async_fetch_batched_response(messages)
    assert result == ["response", "response", "response"]

    # Test mixed success/failure batch
    mock_make_request.side_effect = [
        mock_completion,
        Exception("test error"),
        mock_completion,
    ]
    result = await async_openai_api_adapter._async_fetch_batched_response(messages)
    assert result == ["response", "", "response"]
