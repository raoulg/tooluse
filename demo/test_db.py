from llm_tooluse import ModelConfig, ClientType, LLMClient, MCPToolLoader
from llm_tooluse.schemagenerators import AnthropicAdapter, LlamaAdapter
from pathlib import Path
from loguru import logger
import sys
import asyncio


logger.remove()
logger.add(sys.stderr, level="DEBUG")


async def main():
    server_path = Path("demo/servers/db_server.py").resolve()
    loader = MCPToolLoader()
    collection = await loader.load_server(
        name="db_server",
        target=server_path,
    )
    logger.info(f"Loaded tools: {collection}")

    result = await collection("get_min_max_per_category",
                              category="All",
                              min_price=0,
                              max_price=1000
                              )
    logger.info(f"Test call result: {result}")

    configfile = Path("config.toml").resolve()
    if not configfile.exists():
        logger.error(f"Config file {configfile} does not exist!")
    logger.info(f"Loading config from {configfile}")

    config = ModelConfig.from_toml(configfile)

    llm = LLMClient(config)
    logger.info(f"Initialized LLM client: {llm}")

    queries = [
        "How many products do we have in total?",
        "I have a budget of 700,-, which Phones are available?",
        "How many products are there below 400,-?",
        "I am thinking about getting something nice for myself. I want to spend about 500,-. What combinations of products are available so i get to a total of 500,-?",
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