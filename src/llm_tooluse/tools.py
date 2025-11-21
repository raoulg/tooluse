"""
MCP Tool Loader
Factory for loading tools from MCP servers using the fastmcp connection manager.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from loguru import logger

from llm_tooluse.mcp_adapter import MCPToolReference

from llm_tooluse.schemagenerators import (
    BasicSchemaGenerator,
    LLMSchemaGenerator,
    SchemaGenerator,
    ToolSchema,
)


@dataclass
class Tool:
    """
    Represents a tool that can be used by an LLM.
    Combines the callable function with its schema.
    """

    func: Callable
    schema: ToolSchema

    @classmethod
    def from_function(
        cls, func: Callable, schema_generator: Optional[SchemaGenerator] = None
    ) -> "Tool":
        """Create a Tool from a function, optionally using a schema generator"""
        generator = schema_generator or BasicSchemaGenerator()
        schema = generator.generate_schema(func)
        return cls(func=func, schema=schema)

    @classmethod
    def from_schema_dict(cls, func: Callable, schema_dict: Dict[str, Any]) -> "Tool":
        """Create a Tool from a function and a schema dictionary"""
        try:
            schema = ToolSchema(
                name=schema_dict["name"],
                description=schema_dict["description"],
                parameters=schema_dict["parameters"],
                required=schema_dict["required"],
            )
        except KeyError as e:
            raise ValueError(f"Invalid schema dictionary: {e}")
        return cls(func=func, schema=schema)

    @classmethod
    def from_schema_file(cls, func: Callable, schema_file: Path) -> "Tool":
        schema = ToolSchema.from_file(schema_file)
        return cls(func=func, schema=schema)

    def get_schema_fmt(self, format: str) -> Union[Dict[str, Any], str]:
        """Get the schema in either dict or JSON format"""
        if format.lower() == "json":
            return self.schema.to_json()
        if format.lower() == "dict":
            return self.schema.to_dict()
        raise ValueError(f"Unsupported format: {format}")

    def update_schema(self, schema_source: str | Path) -> None:
        """Update the schema from a JSON string"""
        if isinstance(schema_source, Path):
            self.schema = ToolSchema.from_file(schema_source)
        else:
            # Treat as JSON string
            self.schema = ToolSchema.from_json(schema_source)

    def __call__(self, *args, **kwargs) -> Any:
        """Make the tool callable, delegating to the underlying function"""
        return self.func(*args, **kwargs)

    def __name__(self) -> str:
        """Name of the tool is the name of the function"""
        return self.func.__name__

    def __str__(self) -> str:
        return f"{self.func.__name__}"

    def __repr__(self) -> str:
        return f"{self.func.__name__}"

    def __eq__(self, other: object) -> bool:
        """Tools are equal if they have the same function and schema"""
        if not isinstance(other, Tool):
            return NotImplemented
        same_func = self.func == other.func
        same_schema = self.has_compatible_schema(other)
        return same_func and same_schema

    def has_compatible_schema(self, other: "Tool") -> bool:
        """Check if schemas are compatible for LLM use"""
        return (
            self.schema.parameters == other.schema.parameters
            and self.schema.required == other.schema.required
        )

    def __hash__(self) -> int:
        """Hash based on function and schema since those determine equality"""
        params_tuple = tuple(
            sorted((p.name, str(p.param_type)) for p in self.schema.parameters)
        )
        required_tuple = tuple(sorted(self.schema.required))
        return hash((self.func, params_tuple, required_tuple))


class ToolRegistry:
    """Global registry that maintains single instances of tools"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if not self.initialized:
            logger.debug("initializing registry")
            self._tools: Dict[str, MCPToolReference] = {}
            self.initialized = True

    def register(self, tool: "MCPToolReference") -> None:
        """Register a single tool"""
        key = str(tool)
        self._tools[key] = tool
        logger.debug(f"Registered tool: {key}")

    def get(self, name: str) -> "MCPToolReference":
        """Get a tool by name"""
        logger.debug(f"Retrieving tool: {name}")
        if name not in self._tools:
            available = list(self.available_tools)
            logger.debug(f"registry contains: {available}")
            raise ValueError(f"Tool {name} not registered")
        return self._tools[name]

    def reset(self) -> None:
        """Clear all tools from the registry"""
        self._tools = {}
        logger.debug("Tool registry has been reset")

    @property
    def available_tools(self) -> Set[str]:
        """Names of all registered tools"""
        return set(self._tools.keys())


class ToolCollection:
    """
    Represents a view into the tool registry, maintaining only the names
    of tools in this collection
    """

    def __init__(self, tool_names: Optional[Set[str]]):
        self.tool_names: Set[str] = tool_names or set()

        # Validate all tools exist in registry
        self._registry = ToolRegistry()
        unknown = self.tool_names - self._registry.available_tools
        if unknown:
            logger.warning(f"unkown tools in collection: {unknown}")
            logger.debug(f"available tools:{self._registry.available_tools}")
            logger.debug(f"tool_names: {self.tool_names}")
            raise ValueError(f"Unknown tools: {unknown}")

    @classmethod
    def from_tools(cls, tools: list["MCPToolReference"]) -> "ToolCollection":
        """Create a collection from a list of tool functions"""
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)
        logger.debug(f"Tools in registry: {registry.available_tools}")
        names = {str(tool) for tool in tools}
        logger.debug(f"creating toolcollection with: {names}")

        return cls(names)

    def get_functions(self) -> list["MCPToolReference"]:
        """Returns list of tool functions"""
        return [self._registry.get(name) for name in self.tool_names]

    def get_schemas(self) -> List[ToolSchema]:
        """Returns list of tool schemas"""
        schemas = []
        for name in self.tool_names:
            logger.debug(f"getting schema for tool: {name}")
            schema = self._registry.get(name).get_schema()
            logger.debug(f"got schema: {schema}")
            schemas.append(schema)
        logger.debug(f"returning schemas: {schemas}")
        return schemas

        # return [self._registry.get(name).get_schema() for name in list(self.tool_names)]

    def __getitem__(self, name: str) -> "MCPToolReference":
        return self._registry.get(name)

    async def __call__(self, tool: str, **kwargs) -> Any:
        """Execute a tool from this collection"""
        if tool not in self:
            raise ValueError(f"Tool {tool} not in this collection")
        ref = self._registry.get(tool)
        logger.debug(f"Executing tool {tool} with args: {kwargs}")
        return await ref(**kwargs)

    def __contains__(self, item: str) -> bool:
        """Check if a tool is in this collection"""
        return item in self.tool_names

    def __mul__(self, other: "ToolCollection") -> "ToolCollection":
        """Combines two tool collections"""
        return ToolCollection(self.tool_names | other.tool_names)

    def __sub__(
        self, other: Union["ToolCollection", Set[str], List[str]]
    ) -> "ToolCollection":
        """Remove tools by name"""
        if isinstance(other, ToolCollection):
            exclude = other.tool_names
        elif isinstance(other, (set, list)):
            exclude = set(other)
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")
        return ToolCollection(self.tool_names - exclude)

    def __str__(self) -> str:
        names = [str(n) for n in self.tool_names]
        return f"ToolCollection({names}))"

    def __repr__(self) -> str:
        names = [str(n) for n in self.tool_names]
        return f"ToolCollection({names})"

    def __len__(self) -> int:
        """Number of tools in the collection"""
        return len(self.tool_names)

class ToolFactory:
    """Factory for creating collections of tools"""

    def __init__(
        self,
        schema_generator: Optional[SchemaGenerator] = None,
        llm_client: Optional[Any] = None,
    ):
        self.schema_generator = schema_generator
        if llm_client and not schema_generator:
            self.schema_generator = LLMSchemaGenerator(llm_client)
        if not self.schema_generator:
            self.schema_generator = BasicSchemaGenerator()

    def create_tool(self, func: Union[Callable, Tool]) -> Tool:
        """Create a single tool or return it if already a Tool"""
        if isinstance(func, Tool):
            return func
        return Tool.from_function(func, self.schema_generator)

    def create_collection(
        self, functions: List[Union[Callable, Tool]]
    ) -> ToolCollection:
        """Create a ToolCollection from a list of functions or Tools"""
        tools = [self.create_tool(func) for func in functions]
        return ToolCollection.from_tools(tools)

class MCPToolLoader:
    """
    Factory for loading tools from MCP servers.
    """

    def __init__(self, connection_manager=None):
        # We no longer need to import or expose TransportType
        from llm_tooluse.mcp_client import MCPConnectionManager

        self.connection_manager = connection_manager or MCPConnectionManager()

    async def load_server(
        self,
        name: str,
        target: str,
    ) -> ToolCollection:
        """
        Connect to an MCP server and load its tools.

        The 'target' argument automatically determines the transport:
        - URLs (starting with http://) use SSE.
        - File paths or commands use Stdio.

        Args:
            name: Unique name for this server connection
            target: Connection string (e.g., 'http://localhost:8000', 'my_script.py', 'npx -y server-pkg')
            env: Environment variables (optional, mostly for stdio connections)

        Returns:
            ToolCollection containing all tools from the server

        Examples:
            # Stdio (Local Python script)
            await loader.load_server("calculator", target="calc_server.py")

            # Stdio (Command)
            await loader.load_server("filesystem", target="npx -y @modelcontextprotocol/server-filesystem")

            # SSE (HTTP)
            await loader.load_server("ml_tools", target="http://localhost:8000/sse")
        """

        # Connect to server (ConnectionManager handles transport detection)
        await self.connection_manager.connect_server(
            name=name,
            target=target,
        )

        # Discover tools
        tools = await self.connection_manager.discover_tools(name)

        # Create collection
        return ToolCollection.from_tools(tools)

    async def cleanup(self) -> None:
        """Disconnect from all servers"""
        await self.connection_manager.disconnect_all()