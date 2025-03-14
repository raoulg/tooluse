from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from llm_tooluse.calculator import add, subtract
from llm_tooluse.llm import LLMClient
from llm_tooluse.schemagenerators import BasicSchemaGenerator, LLMSchemaGenerator
from llm_tooluse.settings import ModelConfig
from llm_tooluse.tools import Tool, ToolCollection, ToolFactory


def main():
    load_dotenv()
    basic_generator = BasicSchemaGenerator()
    basic_schema = basic_generator.generate_schema(add)
    logger.info(f"basic schema: {basic_schema.to_dict()}")

    config = ModelConfig.from_toml(Path("config.toml"))
    llm = LLMClient(config)
    llm_generator = LLMSchemaGenerator(llm)

    math_collection = [add, subtract]
    toollist = [Tool.from_function(f) for f in math_collection]
    logger.info(f"toollist: {toollist}")
    toolcollection = ToolCollection.from_tools(toollist)
    logger.info(f"collection: {toolcollection}")
    logger.info(f"registry in collection {toolcollection._registry.available_tools}")

    factory = ToolFactory(
        schema_generator=llm_generator,
        llm_client=llm,
    )
    tools = factory.create_collection(math_collection)
    logger.info(f"tools: {tools}")
    schemas = tools.get_schemas()
    logger.info(f"schemas: {schemas}")
    #
    config = ModelConfig.from_toml(Path("config.toml"))
    llm = LLMClient(config)
    content = "add 2 to 8. Only respond with the result"
    messages = [{"role": "user", "content": content}]
    response = llm(messages)
    logger.info(f"response : {response}")
    # model = "claude-3-haiku-20240307"
    # content = "Give me a haiku about a pet chicken"


if __name__ == "__main__":
    main()
