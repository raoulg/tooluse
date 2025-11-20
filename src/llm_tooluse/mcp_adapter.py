"""
MCP Tool Adapter
Provides a reference-based wrapper for MCP tools that enables set operations.
"""
from dataclasses import dataclass
from typing import Any, Dict
from llm_tooluse.schemagenerators import ParameterSchema, ToolSchema


@dataclass
class MCPToolReference:
    """
    Reference to an MCP tool from any server.
    Provides a uniform interface regardless of tool source.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    _session: Any  # MCP ClientSession handle

    async def __call__(self, **kwargs) -> Any:
        """
        Call the MCP tool with the given arguments.

        Args:
            **kwargs: Tool arguments matching the input_schema

        Returns:
            Tool execution result
        """
        result = await self._session.call_tool(self.name, kwargs)

        # Extract content from MCP result
        if hasattr(result, 'content') and len(result.content) > 0:
            # Return first content item's text
            return result.content[0].text if hasattr(result.content[0], 'text') else result.content[0]
        return result

    def __hash__(self) -> int:
        """Hash based on tool name for set operations."""
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        """Tools are equal if they have the same name."""
        if not isinstance(other, MCPToolReference):
            return NotImplemented
        return self.name == other.name

    def __str__(self) -> str:
        """String representation is the tool name."""
        return self.name

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"MCPToolReference(name='{self.name}')"

    def get_schema(self) -> ToolSchema:
        """Convert MCP schema to ToolSchema format"""
        properties = self.input_schema.get("properties", {})
        required = self.input_schema.get("required", [])

        parameters = []
        for prop_name, prop_schema in properties.items():
            param = ParameterSchema(
                name=prop_name,
                param_type=prop_schema.get("type", "string"),
                description=prop_schema.get("description"),
                enum=prop_schema.get("enum"),
                nullable=prop_schema.get("nullable")
            )
            parameters.append(param)

        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=parameters,
            required=required
        )