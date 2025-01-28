from typing import Any

from anthropic import Anthropic
from loguru import logger
from ollama import Client as OllamaClient

from tooluse.settings import ClientType, ModelConfig
from tooluse.tools import ToolCollection, ToolRegistry


class LLMClient:
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.client = self._initialize_client()
        # Get all available tools
        registry = ToolRegistry()
        all_tools = ToolCollection(registry.available_tools)

        # Filter tools based on config
        if config.allowed_tools is None:
            self.tools: ToolCollection = all_tools
            logger.debug(f"All tools allowed: {self.tools.tool_names}")
        else:
            # Validate requested tools exist
            unknown = set(config.allowed_tools) - registry.available_tools
            if unknown:
                raise ValueError(f"Unknown tools requested: {unknown}")

            # Get all tools except those not in allowed_tools
            excluded = registry.available_tools - set(config.allowed_tools)
            logger.debug(f"Excluded tools: {excluded}")
            self.tools: ToolCollection = all_tools - excluded
            logger.debug(f"Allowed tools: {self.tools}")

    def _initialize_client(self):
        if self.config.client_type == ClientType.ANTHROPIC:
            if not self.config.max_tokens:
                raise ValueError("max_tokens required for Anthropic")
            return Anthropic()
        elif self.config.client_type == ClientType.OLLAMA:
            if not self.config.host:
                raise ValueError("host required for Ollama")
            logger.debug(f"Connecting to Ollama at {self.config.host}")
            return OllamaClient(host=str(self.config.host))
        else:
            raise ValueError(f"Unsupported client type: {self.config.client_type}")

    def _anthropic_call(self, messages, **kwargs) -> Any:
        assert isinstance(self.client, Anthropic)
        return self.client.messages.create(
            model=self.config.model_type.value,
            messages=messages,
            max_tokens=self.config.max_tokens,
            **kwargs,
        )

    def _ollama_call(self, messages, **kwargs) -> Any:
        assert isinstance(self.client, OllamaClient)
        return self.client.chat(
            model=self.config.model_type, messages=messages, **kwargs
        )

    def __call__(self, messages, **kwargs):
        if self.config.client_type == ClientType.ANTHROPIC:
            return self._tool_loop(
                call_func=self._anthropic_call,
                messages=messages,
                tools=self.tools.get_schemas(),
                **kwargs,
            )
        elif self.config.client_type == ClientType.OLLAMA:
            return self._tool_loop(
                self._ollama_call,
                messages=messages,
                tools=self.tools.get_schemas(),
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported client type: {self.config.client_type}")

    def _tool_loop(self, call_func, messages, **kwargs):
        try:
            logger.debug(f"received kwargs: {kwargs}")
            response = call_func(messages=messages, **kwargs)
            logger.debug(f"Initial response: {response}")

            if hasattr(response.message, "tool_calls") and response.message.tool_calls:
                for tool in response.message.tool_calls:
                    try:
                        logger.debug(f"Executing tool: {tool.function.name}")
                        if tool.function.name not in self.tools:
                            logger.warning(f"Tool {tool.function.name} not registred")

                        output = self.tools(
                            tool.function.name, **tool.function.arguments
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "content": str(output),
                                "name": tool.function.name,
                            }
                        )
                    except (AttributeError, ValueError) as e:
                        logger.error(f"Tool execution failed: {e}")
                        continue
                logger.debug(f"Messages after tool calls: {messages}")

                response = call_func(messages=messages, **kwargs)

            return response
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
