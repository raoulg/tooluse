"""
MCP Connection Manager
Manages connections to multiple MCP servers and tool discovery.
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from llm_tooluse.mcp_adapter import MCPToolReference


class MCPConnectionManager:
    """
    Manages connections to MCP servers and provides tool discovery.
    """

    def __init__(self):
        self._sessions: Dict[str, ClientSession] = {}
        self._contexts: Dict[str, Any] = {}  # Store context managers

    async def connect_server(
        self,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Connect to an MCP server via stdio.

        Args:
            name: Unique name for this server connection
            command: Command to execute (e.g., 'python', 'node')
            args: Command arguments (e.g., ['-m', 'my_server'])
            env: Environment variables for the server process
        """
        if name in self._sessions:
            logger.warning(f"Server '{name}' already connected")
            return

        args = args or []
        env_vars = env or {}

        # Merge with current environment
        full_env = os.environ.copy()
        full_env.update(env_vars)

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=full_env if env_vars else None
        )

        logger.debug(f"Connecting to MCP server '{name}': {command} {' '.join(args)}")

        # Create and store the context manager
        context = stdio_client(server_params)
        read_stream, write_stream = await context.__aenter__()

        # Create session
        session = ClientSession(read_stream, write_stream)
        await session.__aenter__()

        # Initialize the session
        await session.initialize()

        self._sessions[name] = session
        self._contexts[name] = (context, session)

        logger.info(f"Connected to MCP server '{name}'")

    async def discover_tools(self, server_name: str) -> List[MCPToolReference]:
        """
        Discover all tools available from a connected server.

        Args:
            server_name: Name of the connected server

        Returns:
            List of MCPToolReference objects for all tools

        Raises:
            ValueError: If server is not connected
        """
        if server_name not in self._sessions:
            raise ValueError(f"Server '{server_name}' is not connected")

        session = self._sessions[server_name]

        logger.debug(f"Discovering tools from server '{server_name}'")
        tools_list = await session.list_tools()

        tool_refs = []
        for tool in tools_list.tools:
            tool_ref = MCPToolReference(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema,
                _session=session
            )
            tool_refs.append(tool_ref)
            logger.debug(f"Discovered tool: {tool.name}")

        logger.info(f"Discovered {len(tool_refs)} tools from '{server_name}'")
        return tool_refs

    async def disconnect_server(self, name: str) -> None:
        """
        Disconnect from a specific server.

        Args:
            name: Name of the server to disconnect
        """
        if name not in self._sessions:
            logger.warning(f"Server '{name}' is not connected")
            return

        context, session = self._contexts[name]

        # Close session
        await session.__aexit__(None, None, None)

        # Close transport
        await context.__aexit__(None, None, None)

        del self._sessions[name]
        del self._contexts[name]

        logger.info(f"Disconnected from server '{name}'")

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        server_names = list(self._sessions.keys())
        for name in server_names:
            await self.disconnect_server(name)

        logger.info("Disconnected from all servers")

    def is_connected(self, name: str) -> bool:
        """Check if a server is connected."""
        return name in self._sessions

    @property
    def connected_servers(self) -> List[str]:
        """Get list of all connected server names."""
        return list(self._sessions.keys())