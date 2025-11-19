# llm_tooluse - Seamless Function Integration for LLMs

`llm_tooluse` is a Python package that simplifies the integration of custom functions (tools) with Large Language Models (LLMs). It provides a streamlined way to register functions, automatically generate schemas, and enable LLMs to use these tools in a conversational context.

## Installation

```bash
uv install llm_tooluse
```
(or use `pip` instead of `uv` if you like to install your dependencies 100x slower)

## Quick Start

### Creating and Using Tools

```python
from llm_tooluse.tools import ToolFactory
from llm_tooluse.settings import ClientType, ModelType, ModelConfig
from llm_tooluse.llm import LLMClient

# Define some simple functions as tools
def add(a: int, b: int) -> int:
    """Add two numbers together"""
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract b from a"""
    return a - b

# Create a tool collection from these functions
factory = ToolFactory()
math_tools = factory.create_collection([add, subtract]) # registers the tools

# Configure and initialize the LLM client
config = ModelConfig(
    client_type=ClientType.OLLAMA,
    model_type=ModelType.LLAMA31,
)
llm = LLMClient(config) # picks up the registered tools

# Use the LLM with access to your tools
messages = [{"role": "user", "content": "Add 234 to 802348. Only respond with the result."}]
response = llm(messages) # will use the add tool
print(response.message.content)
# >>> The answer is: 802582
```

Or, run the demo with `python demo/database_example.py` to see a more complex example in action.

### Using with Different LLM Providers

The package supports different LLM providers:

```python
# For Anthropic's Claude
anthropic_config = ModelConfig(
    client_type=ClientType.ANTHROPIC,
    model_type=ModelType.HAIKU,
    max_tokens=1000
)
```

You can add more models by extending the `ModelType` with your own model:
```python
class ModelType(str, Enum):
    HAIKU = "claude-3-haiku-20240307"
    LLAMA31 = "llama3.1"
    CODELLAMA34B = "codellama:34b"
    PHI3 = "phi3:latest"
    YOUR_MODEL = "your_model:latest"
```
for anthropic; you need to have an `ANTROPIC_API_KEY` in your `.env`

## Advanced Usage

### The `Tool` Class

A `Tool` combines a callable function with its schema:

```python
from pathlib import Path
from llm_tooluse.tools import Tool

# Create a tool directly with the basic schemagenerator by default
calculator_tool = Tool.from_function(add)
# you can get the schema
print(calculator_tool.schema)
# ToolSchema({'name': 'add', ...

# or call the function directly
print(add_tool(2,2))
# 4

# the schema can be saved as json
from pathlib import Path
schemafile = Path("add_tool_schema.json")
add_tool.schema.to_file(schemafile)

# and updated
add_tool.update_schema(schemafile)
# or recreated
toolschema = ToolSchema.from_file(schemafile)
new_tool = Tool(func=add, schema=toolschema)
```

You can also
- use `tool.schema.to_dict()` or `tool.schema.to_json()` to get the schema as a dictionary or JSON string
- update directly from a json string `tool.update_schema(json_string)`

### `ToolCollection` Class

`ToolCollection` provides a view into the global registry, enabling operations on groups of tools:

```python
from llm_tooluse.tools import ToolCollection

# Create collections from different domains, automatically registering the tools
math_tools = factory.create_collection([add, subtract, multiply, divide])
text_tools = factory.create_collection([uppercase, lowercase, capitalize])

# Combine collections
all_tools = math_tools * text_tools

# Remove specific tools (can be another ToolCollection, but also a list of tool names)
basic_math = all_tools - ["divide", "multiply"]

# or recreate collections from the registry with just their names
basic_tools = ToolCollection({"add", "subtract"})

# Check if a tool is in a collection
if "add" in math_tools:
    print("Add function is available")

# Execute a tool from a collection
result = math_tools("add", a=5, b=3)
```

### Schema Generators

The package provides different ways to generate schemas for your tools:

```python
from llm_tooluse.schemagenerators import (
    BasicSchemaGenerator,
    LLMSchemaGenerator
)
from llm_tooluse.llm import LLMClient

# Basic schema generation from function signatures and docstrings
basic_generator = BasicSchemaGenerator()
basic_schema = basic_generator.generate_schema(add)

# LLM-powered schema generation for richer descriptions
llm_client = LLMClient(config)
llm_generator = LLMSchemaGenerator(llm_client)
llm_schema = llm_generator.generate_schema(add)

# Create a factory with LLM-powered schema generation
factory = ToolFactory(llm_client=llm_client)
enhanced_tool_collection = factory.create_collection([function1, function2])

# View the enhanced schemas
schemas = enhanced_tool_collection.get_schemas()
```

### `ToolRegistry`

The `ToolRegistry` provides global access to registered tools, and is used by both the `ToolFactory` and `LLMClient` in the background:

```python
from llm_tooluse.tools import ToolRegistry

# Access the global registry
registry = ToolRegistry()

# Get all available tools
available_tools = registry.available_tools

# Get a specific tool by name
add_tool = registry.get("add")

# Reset the registry (clear all tools)
registry.reset()
```

## Adapters for Different LLM Providers

The package includes adapters to format tool schemas and parse responses for different LLM providers:

```python
from llm_tooluse.schemagenerators import AnthropicAdapter, LlamaAdapter

# Get schemas formatted for specific providers
anthropic_schema = AnthropicAdapter.format_schema(tool.schema)
llama_schema = LlamaAdapter.format_schema(tool.schema)
```
This is used in the backend by the LLMClient to ensure that the tool schemas are compatible with the specific LLM provider.

## Configuration via TOML

You can also configure models using TOML files:

```python
from pathlib import Path
from llm_tooluse.settings import ModelConfig

config = ModelConfig.from_toml(Path("config.toml"))
llm = LLMClient(config)
```

Example TOML configuration for haiku:
```toml
[llm]
client_type = "anthropic"
model_type = "claude-3-haiku-20240307"
max_tokens = 1000
allowed_tools = ["add", "subtract", "search"]
```

Or llama3.1
```toml
[llm]
client_type = "ollama"
model_type = "llama3.1"
max_tokens = 1000
host = "http://127.0.0.1:11434"
```


## Best Practices

1. Write clear docstrings for your functions to generate better schemas
2. Use type hints to help with parameter validation
3. For complex tools, consider using the LLMSchemaGenerator. You might need to inspect the generated schema and make adjustments, eg via the json export/import.
4. Group related tools into separate collections for better organization

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## resources
anthropic:
- tool use course https://github.com/anthropics/courses/tree/master/tool_use
- docs https://docs.anthropic.com/en/docs/build-with-claude/tool-use
