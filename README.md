# llm_tooluse - Seamless MCP Tool Integration for LLMs

`llm_tooluse` is a Python package that simplifies the integration of Model Context Protocol (MCP) tools with Large Language Models (LLMs). It provides a streamlined way to load tools from MCP servers, manage them in collections, and enable LLMs to use them in a conversational context.

## Quick Start

### Loading Tools from an MCP Server

The core of `llm_tooluse` is the `MCPToolLoader`, which connects to MCP servers and loads their tools into a `ToolCollection`.

```python
import asyncio
from pathlib import Path
from llm_tooluse import MCPToolLoader, ModelConfig, LLMClient

async def main():
    # 1. Initialize the loader
    loader = MCPToolLoader()

    # 2. Load tools from a local MCP server script
    # This example assumes you have a python script that runs an MCP server
    # See demo/servers/calc_server.py for an example server
    server_path = Path("demo/servers/calc_server.py").resolve()
    
    tools = await loader.load_server(
        name="calculator",
        target=str(server_path)
    )
    
    print(f"Loaded tools: {tools}")
    # Output: Loaded tools: ToolCollection(['add', 'subtract', 'multiply', 'divide'])

    # 3. Configure the LLM Client
    config = ModelConfig(
        client_type="anthropic", # or "ollama"
        model_type="claude-3-haiku-20240307",
        max_tokens=1000
    )
    llm = LLMClient(config)

    # 4. Use the LLM with the loaded tools
    # The LLMClient automatically has access to registered tools
    messages = [{"role": "user", "content": "What is 15 plus 27?"}]
    response = await llm(messages)
    
    # Use the appropriate adapter to parse the response if needed, 
    # or access response.content directly depending on the client type.
    print(response)

    # 5. Cleanup connections
    await loader.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

## Core Concepts

### MCPToolReference

`MCPToolReference` is the wrapper for any tool loaded from an MCP server. It holds the tool's schema and handles the execution logic via the underlying MCP client. It replaces the legacy `Tool` class.

### MCPToolLoader

The `MCPToolLoader` is responsible for connecting to MCP servers and discovering tools. It supports various targets via `fastmcp`:
- **Python Scripts**: `target="/path/to/server.py"` (uses stdio)
- **Node.js Scripts**: `target="/path/to/server.js"` (uses stdio)
- **HTTP Servers**: `target="http://localhost:8000"` (uses HTTP)

### ToolCollection

`ToolCollection` provides a view into the global registry, enabling operations on groups of tools. It supports set operations:

```python
# Create subsets
math_tools = all_tools - ["search", "weather"]
basic_math = math_tools - ["power", "sqrt"]

# Combine collections
my_toolkit = basic_math * text_tools

# Execute a tool directly from a collection
result = await my_toolkit("add", a=5, b=3)
```

### ToolRegistry

The `ToolRegistry` is a singleton that maintains all loaded tools. When you load a server, its tools are automatically registered here. The `LLMClient` uses this registry to provide tools to the model.

## Configuration

You can configure the LLM client using a TOML file or directly via `ModelConfig`.

**config.toml**:
```toml
[llm]
client_type = "ollama"
model_type = "llama3.1"
max_tokens = 1000
host = "http://localhost:11434"
# allowed_tools = ["add", "subtract"] # Optional: restrict accessible tools
```

**Loading Config**:
```python
from llm_tooluse import ModelConfig

config = ModelConfig.from_toml(Path("config.toml"))
```

## Demos

Check the `demo` directory for complete examples:

- `demo/test_calc.py`: Basic example using a calculator MCP server.
- `demo/test_integration.py`: Comprehensive test suite for tool loading and execution.
