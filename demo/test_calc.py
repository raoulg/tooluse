from llm_tooluse import ModelConfig, ClientType, LLMClient, MCPToolLoader
from llm_tooluse.schemagenerators import AnthropicAdapter, LlamaAdapter
from pathlib import Path
from loguru import logger
import sys
import asyncio


logger.remove()
logger.add(sys.stderr, level="INFO")


async def main():
    calculator_path = Path("demo/servers/calc_server.py").resolve()
    loader = MCPToolLoader()
    collection = await loader.load_server(
        name="calculator",
        target=calculator_path,
    )
    logger.info(f"Loaded tools: {collection}")

    configfile = Path("config.toml").resolve()
    if not configfile.exists():
        logger.error(f"Config file {configfile} does not exist!")
    logger.info(f"Loading config from {configfile}")

    config = ModelConfig.from_toml(configfile)

    llm = LLMClient(config)
    logger.info(f"Initialized LLM client: {llm}")

    queries = [
        "What is 15 plus 27?",
        "What is 100 minus 45?",
        "What is 12 times 8?",
        "What is 144 divided by 12?",
    ]

    adapter = AnthropicAdapter
    if config.client_type == ClientType.OLLAMA:
        adapter = LlamaAdapter

    for i, query in enumerate(queries):
        logger.info(f"{query}")
        messages = [{"role": "user", "content": query}]
        response = await llm(messages)
        content = adapter.get_content(response)
        logger.info(f"LLM response: \n{content}")

if __name__ == "__main__":
    asyncio.run(main())