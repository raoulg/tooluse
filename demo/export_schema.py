from pathlib import Path
from llm_tooluse.tools import Tool
from database_example import get_min_max_per_category
from loguru import logger
from llm_tooluse.calculator import add
from llm_tooluse.schemagenerators import AnthropicAdapter, LlamaAdapter
import json



if __name__ == "__main__":
    logger.info("=" * 20 + "Schema for add" + "=" * 20)
    tool = Tool.from_function(add)
    schema = tool.schema
    adapter = AnthropicAdapter
    fmt = adapter.format_schema(schema)
    logger.info(json.dumps(fmt, indent=2))

    logger.info("=" * 20 + "Schema for get_min_max_per_category" + "=" * 20)
    tool = Tool.from_function(get_min_max_per_category)
    schema = tool.schema
    fmt = adapter.format_schema(schema)
    logger.info(json.dumps(fmt, indent=2))
