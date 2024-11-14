import logging
from typing import Any

from tqdm import tqdm

from .adapters.openai.adapter import AsyncOpenAIAdapter, OpenAIAdapter
from .common import (
    AsyncBaseFlowJudgeModel,
    BaseFlowJudgeModel,
    ModelConfig,
    ModelType,
    OpenAIGenerationParams,
)

logger = logging.getLogger(__name__)


class OpenAIModelConfig(ModelConfig):
    """Model config for the OpenAI model class."""

    def __init__(
        self,
        generation_params: OpenAIGenerationParams,
        exec_async: bool = False,
        **kwargs: Any,
    ):
        """Initialize the OpenAIModelConfig."""
        model_id = kwargs.pop("_model_id", None)
        if model_id is None:
            raise ValueError("A model_id is required")

        model_type = ModelType.OPENAI_ASYNC if exec_async else ModelType.OPENAI

        if not isinstance(generation_params, OpenAIGenerationParams):
            raise ValueError("generation_params must be an instance of OpenAIGenerationParams")

        super().__init__(model_id, model_type, generation_params, exec_async=exec_async, **kwargs)


class OpenAIModel(BaseFlowJudgeModel, AsyncBaseFlowJudgeModel):
    """Combined FlowJudge Model class for OpenAI-compatible APIs (OpenAI, Together, etc.)."""

    _DEFAULT_MODEL_ID = "gpt-4"
    _DEFAULT_BASE_URL = "https://api.openai.com/v1"
    _DEFAULT_KEY_VAR = "OPENAI_API_KEY"

    def __init__(
        self,
        api_adapter: OpenAIAdapter | AsyncOpenAIAdapter | None = None,
        exec_async: bool = False,
        generation_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Initialize the OpenAI-compatible Model class."""
        model_id = kwargs.pop("_model_id", self._DEFAULT_MODEL_ID)
        base_url = kwargs.pop("base_url", self._DEFAULT_BASE_URL)
        key_var = kwargs.pop("key_var", self._DEFAULT_KEY_VAR)

        if api_adapter is not None and not isinstance(
            api_adapter, (OpenAIAdapter, AsyncOpenAIAdapter)
        ):
            raise OpenAIModelError(
                status_code=3,
                message="Incompatible API adapter. Use OpenAIAdapter or AsyncOpenAIAdapter",
            )

        self.api_adapter = api_adapter or (
            OpenAIAdapter(model_id=model_id, base_url=base_url, key_var=key_var)
            if not exec_async
            else AsyncOpenAIAdapter(model_id=model_id, base_url=base_url, key_var=key_var)
        )

        generation_params = OpenAIGenerationParams(**(generation_params or {}))
        config = OpenAIModelConfig(
            generation_params=generation_params,
            exec_async=exec_async,
            _model_id=model_id,
        )
        self.config = config

        super().__init__(model_id, config.model_type, config.generation_params, **kwargs)

        logger.info("Successfully initialized OpenAI!")

    def _format_conversation(self, prompt: str) -> list[dict[str, Any]]:
        return [{"role": "user", "content": prompt.strip()}]

    def _generate(self, prompt: str) -> str:
        logger.info("Initiating single OpenAI request")

        conversation = self._format_conversation(prompt)
        return self.api_adapter._fetch_response(conversation)

    def _batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
        logger.info("Initiating batched OpenAI requests")

        iterator = tqdm(prompts) if use_tqdm else prompts
        conversations = [self._format_conversation(prompt) for prompt in iterator]
        return self.api_adapter._fetch_batched_response(conversations)

    async def _async_generate(self, prompt: str) -> str:
        logger.info("Initiating single OpenAI async request")
        conversation = self._format_conversation(prompt)
        if self.config.kwargs.get("exec_async", False):
            return await self.api_adapter._async_fetch_response(conversation)
        else:
            logger.error("Attempting to run an async request with a synchronous API adapter")

    async def _async_batch_generate(
        self, prompts: list[str], use_tqdm: bool = True, **kwargs: Any
    ) -> list[str]:
        logger.info("Initiating batched OpenAI async requests")
        # Use tqdm if requested
        iterator = tqdm(prompts) if use_tqdm else prompts
        conversations = [self._format_conversation(prompt) for prompt in iterator]
        return await self.api_adapter._async_fetch_batched_response(conversations)


class OpenAIModelError(Exception):
    """Custom exception for OpenAI-related errors."""

    def __init__(self, status_code: int, message: str):
        """Initialize with a status code and message."""
        self.status_code = status_code
        self.message = message
        super().__init__(self.message)
