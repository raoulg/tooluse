import inspect
import json
from abc import ABC, abstractmethod
from inspect import Parameter, getsource, signature
from string import Template
from typing import TYPE_CHECKING, Any, Callable, Dict, List, get_type_hints
from dataclasses import asdict, dataclass
from pathlib import Path

from loguru import logger
from tooluse.settings import ClientType

if TYPE_CHECKING:
    from tooluse.llm import LLMClient


SCHEMA_PROMPT_TEMPLATE = Template("""Given this Python function information:
source: ${source}
Basic schema: ${basic_schema}

Please extend this with clear, detailed descriptions of what this function and each parameter does.
respond with the following JSON schema:
{
    "description": "A clear, detailed description of what this function does",
    "parameters": {
        "param1": {
            "description": "A clear, detailed description of what this parameter does",
        },
    },
}
reply with the JSON schema only, and nothing else
""")


@dataclass
class ToolSchema:
    """Represents the schema for a tool, following Anthropic's format"""
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __repr__(self) -> str:
        dict_schema = self.to_dict()
        return f"ToolSchema({dict_schema})"

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string for easy viewing/editing"""
        return json.dumps(self.to_dict(), indent=indent)

    def to_file(self, file_path: Path, indent: int = 2) -> None:
        """Write schema to a JSON file

        Args:
            file_path: Path where the JSON file will be saved
            indent: Number of spaces for indentation in the JSON file
        """
        with file_path.open('w') as f:
            f.write(self.to_json(indent=indent))

    @classmethod
    def from_json(cls, json_str: str) -> "ToolSchema":
        """Create schema from JSON string"""
        data = json.loads(json_str)
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data["parameters"],
            required=data["required"],
        )

    @classmethod
    def from_file(cls, file_path: Path) -> "ToolSchema":
        """Create schema from a JSON file
        Args:
            file_path: Path to the JSON file containing the schema

        Returns:
            A new ToolSchema instance with data from the file
        """
        with file_path.open('r') as f:
            json_str = f.read()
        return cls.from_json(json_str)

class LLMAdapter(ABC):
    """Abstract base class for LLM client adapters."""

    @classmethod
    @abstractmethod
    def format_schema(cls, toolschema: ToolSchema) -> Dict[str, Any]:
        """Format tool schema for the specific LLM."""
        pass

    @classmethod
    @abstractmethod
    def get_content(cls, response) -> str:
        """Get content from LLM response."""
        pass

    @classmethod
    @abstractmethod
    def append_message(cls, messages: List, response) -> List:
        pass

    @classmethod
    @abstractmethod
    def extract_tool_calls(cls, response) -> List:
        """Extract tool calls from LLM response."""
        pass

    @classmethod
    @abstractmethod
    def parse_tool_call(cls, tool) -> dict[str, Any]:
        """Parse tool call to extract name and arguments."""
        pass

    @classmethod
    @abstractmethod
    def format_tool_response(cls, toolcall: dict, output) -> Dict[str, Any]:
        """Format tool response for the LLM."""
        pass

class AnthropicAdapter(LLMAdapter):
    @classmethod
    def format_schema(cls, toolschema: ToolSchema) -> Dict[str, Any]:
        return {
            "name": toolschema.name,
            "description": toolschema.description,
            "input_schema": {
                "type": "object",
                "properties": toolschema.parameters,
                "required": toolschema.required,
            },
        }

    @classmethod
    def append_message(cls, messages: List, response) -> List:
        messages.append({"role": "assistant", "content": response.content})
        return messages

    @classmethod
    def get_content(cls, response) -> str:
        return response.content


    @classmethod
    def extract_tool_calls(cls, response) -> List:
        if hasattr(response, "content"):
            return [block for block in response.content
                   if getattr(block, "type", None) == "tool_use"]
        return []

    @classmethod
    def parse_tool_call(cls, tool) -> dict[str, Any]:
        return {
            "id" : tool.id,
            "name" : tool.name,
            "args": tool.input,
        }

    @classmethod
    def format_tool_response(cls, toolcall: dict, output) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": toolcall["id"],
                    "content": str(output),
                }

            ]
        }


class LlamaAdapter(LLMAdapter):
    @classmethod
    def format_schema(cls, toolschema: ToolSchema) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": toolschema.name,
                "description": toolschema.description,
                "parameters": {
                    "type": "object",
                    "properties": toolschema.parameters,
                    "required": toolschema.required,
                }
            }
        }

    @classmethod
    def get_content(cls, response) -> str:
        return response.message.content

    @classmethod
    def append_message(cls, messages: List, response) -> List:
        messages.append(response.message)
        return messages

    @classmethod
    def extract_tool_calls(cls, response) -> List:
        if hasattr(response.message, "tool_calls") and response.message.tool_calls:
            return response.message.tool_calls
        return []

    @classmethod
    def parse_tool_call(cls, tool) -> dict[str, Any]:
        return {
            "name" : tool.function.name,
            "args": tool.function.arguments,
        }

    @classmethod
    def format_tool_response(cls, toolcall: dict, output) -> Dict[str, Any]:
        return {
            "role": "tool",
            "content": str(output),
            "name": toolcall["name"],
        }

class SchemaGenerator(ABC):
    """Abstract base class for schema generators"""

    @abstractmethod
    def generate_schema(self, func: Callable) -> ToolSchema:
        pass


class BasicSchemaGenerator(SchemaGenerator):
    """Basic schema generator using inspect module"""

    _TYPE_MAP = {
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
        bool: {"type": "boolean"},
    }

    def generate_schema(self, func: Callable) -> ToolSchema:
        sig = signature(func)
        type_hints = get_type_hints(func)

        parameters = {}
        logger.debug(f"Function signature: {sig}")
        required = []

        for name, param in sig.parameters.items():
            param_type = type_hints.get(name, Any)
            logger.debug(f"Parameter {name}: {param_type}")

            # Skip self parameter for methods
            if name == "self":
                continue
            # Inside generate_schema
            # dict is necessary to avoid overwriting the typemap
            param_schema = dict(self._TYPE_MAP.get(param_type, {"type": "string"}))
            logger.debug(f"Parameter schema: {param_schema}")

            # Handle default values
            if param.default is Parameter.empty:
                required.append(name)

            parameters[name] = param_schema
            logger.debug(f"Parameters: {parameters}")

        logger.debug(f"Parameters: {parameters}")

        # Get description from docstring
        description = inspect.getdoc(func) or f"Function {func.__name__}"
        return ToolSchema(
            name=func.__name__,
            description=description,
            parameters=parameters,
            required=required,
        )


class LLMSchemaGenerator(SchemaGenerator):
    """Schema generator that uses LLM to generate high-quality descriptions"""

    def __init__(self, llm_client: "LLMClient"):
        self.llm = llm_client
        self.basic_generator = BasicSchemaGenerator()

    def generate_schema(self, func: Callable) -> ToolSchema:
        # First get basic schema for structure
        basic_schema = self.basic_generator.generate_schema(func)
        try:
            source = getsource(func)
            info = {"source": inspect.getsource(func), "basic_schema": basic_schema}
            content = SCHEMA_PROMPT_TEMPLATE.substitute(info)
            messages = [{"role": "user", "content": content}]
            try:
                response = self.llm(messages)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                return basic_schema

            try:
                adapter = None
                if self.llm.config.client_type == ClientType.ANTHROPIC:
                    adapter = AnthropicAdapter
                elif self.llm.config.client_type == ClientType.OLLAMA:
                    adapter = LlamaAdapter
                if adapter is None:
                    raise ValueError(f"Unsupported client type: {self.llm.config.client_type}")
                content = adapter.get_content(response)
                enhanced = json.loads(content)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
                enhanced = json.loads(json_match.group(1))
                if not enhanced:
                    logger.warning("LLM enahncement failed, using basic schema")
                    return basic_schema
            try:
                basic_schema.description = enhanced['description']
                for key in enhanced['parameters'].keys():
                    basic_schema.parameters[key]["description"] = enhanced['parameters'][key]['description']
                return basic_schema
            except KeyError as e:
                logger.error(f"LLM response missing key: {e}")
                return basic_schema
        except Exception as e:
            logger.error(f"Error generating schema: {e}")
            return basic_schema

