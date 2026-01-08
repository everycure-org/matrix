import os
from abc import ABC, abstractmethod
from typing import Any

import tenacity
from pydantic import BaseModel
from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.litellm import LiteLLMProvider
from pydantic_ai.settings import ModelSettings
from tenacity import stop_after_attempt, wait_exponential


class InvokableGraph(ABC):
    def __init__(self):
        self._compiled_graph = self.compile_graph()

    @abstractmethod
    def compile_graph(self):
        pass

    @abstractmethod
    async def safe_invoke(self, *args, **kwargs) -> Any:
        pass

    async def invoke(self, inputs: dict[str, str]) -> dict[str, Any]:
        try:
            final_state = await self._compiled_graph.ainvoke(inputs)
            return final_state
        except Exception as e:
            print(f"Error in invoke: {str(e)}")
            raise


class LLMConfig:
    """Configuration class for LLM model settings"""

    def __init__(self, config: dict):
        """Initialize from config dict using a prefix for key names"""
        self.system_prompt = config.get("system_prompt")
        self.prompt = config.get("prompt")
        self.model_config = config.get("model_config")


@tenacity.retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, max=60))
async def get_llm_response(
    prompt: str,
    model_config: dict,
    pydantic_model: BaseModel,
    system_prompt: str,
    messages: list[Any] | None = None,
    retries: int = 2,
):
    openai_model = OpenAIModel(
        model_name=f"openai/{model_config['model']}",
        provider=LiteLLMProvider(
            api_base=os.environ["LITELLM_BASE_URL"],
            api_key=os.environ["LITELLM_API_KEY"],
        ),
    )
    agent = Agent(
        openai_model,
        system_prompt=system_prompt,
        retries=retries,
        output_type=NativeOutput([pydantic_model]),
        model_settings=ModelSettings(**model_config),
    )

    return await agent.run(
        user_prompt=prompt,
        message_history=messages,
    )
