import asyncio
import os
from typing import Any

import openai
import structlog
from openai import AsyncOpenAI, OpenAI, OpenAIError

from flow_eval.models.adapters.base import AsyncBaseAPIAdapter, BaseAPIAdapter

logger = structlog.get_logger(__name__)


class OpenAIAdapter(BaseAPIAdapter):
    """API utility class to execute sync requests from OpenAI-compatible APIs."""

    def __init__(self, model_id: str, base_url: str, key_var: str) -> None:
        """Initialize the OpenAIAdapter.

        :param model_id: The model ID to use for requests.
        :param base_url: The base URL for the API.
        :param key_var: The environment variable containing the API key.
        """
        self.model_id = model_id
        self.base_url = base_url

        try:
            self.api_key = os.environ[key_var]
        except KeyError as e:
            raise ValueError(f"{key_var} is not provided in the environment.") from e

        self.client = OpenAI(api_key=self.api_key, base_url=base_url)

        super().__init__(base_url)

    def _make_request(self, request_messages: dict[str, Any]) -> dict:
        try:
            completion = self.client.chat.completions.create(
                messages=request_messages,
                model=self.model_id,
            )
            return completion

        except OpenAIError as e:
            logger.warning(f"Model request failed: {e}")
        except Exception as e:
            logger.warning(f"An unexpected error occurred: {e}")

    def _fetch_response(self, request_messages: dict[str, Any]) -> str:
        completion = self._make_request(request_messages)

        try:
            message = completion.choices[0].message.content.strip()
            return message
        except Exception as e:
            logger.warning(f"Failed to parse model response: {e}")
            logger.warning("Returning default value")
            return ""

    def _fetch_batched_response(self, request_messages: list[dict[str, Any]]) -> list[str]:
        outputs = []
        for message in request_messages:
            completion = self._make_request(message)
        try:
            outputs.append(completion.choices[0].message.content.strip())
        except Exception as e:
            logger.warning(f"Failed to parse model response: {e}")
            logger.warning("Returning default value")
            outputs.append("")
        return outputs


class AsyncOpenAIAdapter(AsyncBaseAPIAdapter):
    """API utility class to execute async requests from OpenAI-compatible APIs.

    This adapter provides methods for making asynchronous requests to OpenAI models,
    handling retries with exponential backoff, and managing rate limits.
    """

    def __init__(
        self,
        model_id: str,
        base_url: str,
        key_var: str,
        max_retries: int = 1,
        request_timeout: float = 120.0,
    ):
        """Initialize the AsyncOpenAIAdapter."""
        if not model_id or not isinstance(model_id, str):
            raise ValueError("model_id must be a non-empty string")
        if not base_url or not isinstance(base_url, str):
            raise ValueError("base_url must be a non-empty string")
        if max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be a positive number")
        if not key_var or not isinstance(key_var, str):
            raise ValueError("key_var must be a non-empty string")

        super().__init__(base_url)

        try:
            self.api_key = os.environ[key_var]
        except KeyError as e:
            raise ValueError(f"{key_var} is not provided in the environment.") from e

        self.model_id = model_id

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=request_timeout,
            max_retries=max_retries,
        )

    async def _make_request(self, request_messages: dict[str, Any]) -> dict:
        try:
            completion = await self.client.chat.completions.create(
                messages=request_messages,
                model=self.model_id,
            )
            return completion

        except openai.APIConnectionError as e:
            logger.error("Failed to connect to OpenAI API", error=str(e.__cause__))
            raise
        except openai.RateLimitError as e:
            logger.warning("Rate limit exceeded, consider implementing backoff", error=str(e))
            raise
        except openai.BadRequestError as e:
            logger.error("Invalid request parameters", error=str(e))
            raise
        except openai.NotFoundError as e:
            logger.error("Requested resource not found", error=str(e))
            raise
        except openai.UnprocessableEntityError as e:
            logger.error("Request is unprocessable", error=str(e))
            raise
        except openai.InternalServerError as e:
            logger.error("OpenAI API internal server error", error=str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error occurred", error=str(e))
            raise

    async def _async_fetch_response(self, request_messages: dict[str, Any]) -> str:
        completion = await self._make_request(request_messages=request_messages)
        try:
            message = completion.choices[0].message.content.strip()
            return message
        except Exception as e:
            logger.warning(f"Failed to parse model response: {e}")
            logger.warning("Returning default value")
            return ""

    async def _async_fetch_batched_response(
        self, request_messages: list[dict[str, Any]]
    ) -> list[str]:
        """Process multiple messages asynchronously and return their responses.

        Args:
            request_messages: List of message dictionaries to process

        Returns:
            List of response strings from the model
        """
        tasks = [self._make_request(message) for message in request_messages]
        completions = await asyncio.gather(*tasks, return_exceptions=True)

        outputs = []
        for completion in completions:
            try:
                if isinstance(completion, Exception):
                    logger.warning(f"Request failed: {completion}")
                    outputs.append("")
                else:
                    outputs.append(completion.choices[0].message.content.strip())
            except Exception as e:
                logger.warning(f"Failed to parse model response: {e}")
                outputs.append("")

        return outputs
