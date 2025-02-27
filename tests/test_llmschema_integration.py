import pytest
from loguru import logger
from pydantic import HttpUrl

from tooluse.calculator import add
from tooluse.schemagenerators import (
    BasicSchemaGenerator,
    LLMSchemaGenerator,
    ToolSchema,
)
from tooluse.settings import ClientType, ModelConfig, ModelType


@pytest.fixture
def ollama_config() -> ModelConfig:
    """Create a ModelConfig for local Ollama"""
    config = ModelConfig(
        client_type=ClientType.OLLAMA,
        model_type=ModelType.LLAMA31,  # or whichever model you have installed
        host=HttpUrl("http://localhost:11434"),
        allowed_tools=None,
        max_tokens=1000,
    )
    logger.info("Created Ollama config")
    return config


@pytest.fixture
def llm_generator(ollama_config):
    """Create LLMSchemaGenerator with Ollama client"""
    try:
        generator = LLMSchemaGenerator(ollama_config)
        logger.success("Successfully created LLM client")
    except Exception as e:
        logger.error(f"Failed to create LLM client: {e}")
        raise
    return generator


@pytest.mark.integration
def test_basic_function_schema_generation(llm_generator):
    """Test generating schema for a simple function with Ollama"""
    basic_generator = BasicSchemaGenerator()
    basic_schema = basic_generator.generate_schema(add)
    llm_schema = llm_generator.generate_schema(add)

    assert isinstance(
        basic_schema, ToolSchema
    ), f"Expected ToolSchema but got {type(basic_schema)}"
    assert isinstance(
        llm_schema, ToolSchema
    ), f"Expected ToolSchema but got {type(llm_schema)}"
    assert basic_schema != llm_schema, "Basic and LLM schemas should differ"
