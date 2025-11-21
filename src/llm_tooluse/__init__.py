import sys

from loguru import logger

from llm_tooluse.llm import LLMClient
from llm_tooluse.settings import ClientType, ModelConfig
from llm_tooluse.tools import (MCPConnectionManager, MCPToolLoader,
                               MCPToolReference, ToolCollection, ToolRegistry)

logger.remove()
logger.add(sys.stderr, level="WARNING")
logger.add("logs/logfile.log", level="DEBUG")


__all__ = [
    "LLMClient",
    "MCPToolReference",
    "MCPConnectionManager",
    "MCPToolLoader",
    "ToolCollection",
    "ToolRegistry",
    "ClientType",
    "ModelConfig",
]
