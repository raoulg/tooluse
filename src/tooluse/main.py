import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from tooluse.calculator import add, subtract
from tooluse.llm import LLMClient
from tooluse.schemagenerators import BasicSchemaGenerator, LLMSchemaGenerator
from tooluse.settings import ModelConfig
from tooluse.tools import Tool

logger.remove()
logger.add(sys.stderr, level="DEBUG")
logger.add("logs/logfile.log", level="DEBUG")


def main():
    load_dotenv()
    basic_generator = BasicSchemaGenerator()
    basic_schema = basic_generator.generate_schema(add)
    logger.info(f"basic schema: {basic_schema.to_dict()}")
    config = ModelConfig.from_toml(Path("config.toml"))
    llm = LLMClient(config)
    generator = LLMSchemaGenerator(llm)
    schema = generator.generate_schema(add)
    logger.info(f"llm schema: {schema.to_dict()}")
    math_collection = [add, subtract]
    toollist = [Tool.from_function(f) for f in math_collection]
    logger.info(f"toollist: {toollist}")
    # toolcollection = ToolCollection.from_tools(toollist)
    # logger.info(f"collection: {toolcollection}")
    # logger.info(f"registry in collection {toolcollection._registry.available_tools}")

    # factory = ToolFactory()
    # tools = factory.create_collection(math_collection)
    # # logger.info(f"tools: {tools}")
    # schemas = tools.get_schemas()
    # logger.info(f"schemas: {schemas}")
    #
    # config = ModelConfig.from_toml(Path("config.toml"))
    # llm = LLMClient(config)
    # content = "add 2 to 8. Only respond with the result"
    # messages = [{"role": "user", "content": content}]
    # response = llm(messages)
    # logger.info(f"response : {response}")
    # model = "claude-3-haiku-20240307"
    # content = "Give me a haiku about a pet chicken"


if __name__ == "__main__":
    main()

# def haiku(client) -> str:
#     msg = client.messages.create(
#         model="claude-3-haiku-20240307",
#         max_tokens=1000,
#         messages=[
#             {
#                 "role": "user",
#                 "content": "Hi there! Please write me a haiku about a pet chicken",
#             }
#         ],
#     )
#     return msg.content[0].text


# def query(client: Anthropic, content: str, model: str, tools: list):
#     response = client.messages.create(
#         model=model,
#         system="You have access to tools, but only use them when necessary. If a tool is not required, respond as normal",
#         messages=[
#             {
#                 "role": "user",
#                 "content": content,
#             }
#         ],
#         max_tokens=400,
#         tools=tools,
#     )
#     return response


# def reaction(response) -> None:
#     if response.stop_reason == "tool_use":
#         tool_use = response.content[-1]
#         assert isinstance(tool_use, ToolUseBlock)
#         if tool_use.name == "calculator":
#             print("Claude wants to use the calculator tool")
#             try:
#                 result = calculator_tool(tool_use)
#                 print(f"Result is: {result}")
#                 print(response.content[0])
#             except ValueError as e:
#                 print(f"Error: {e}")
#     elif response.stop_reason == "end_turn":
#         print("Claude did not want to use a tool")
#         assert isinstance(response.content[0], TextBlock)
#         print(response.content[0].text)
