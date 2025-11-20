import sys
from loguru import logger

from llm_tooluse.mcp_adapter import MCPToolReference
from llm_tooluse.mcp_client import MCPConnectionManager
from llm_tooluse.tools import MCPToolLoader, ToolCollection, ToolRegistry

logger.remove()
logger.add(sys.stderr, level="WARNING")
logger.add("logs/logfile.log", level="DEBUG")


__all__ = [
    "MCPToolReference",
    "MCPConnectionManager",
    "MCPToolLoader",
    "ToolCollection",
    "ToolRegistry",
]