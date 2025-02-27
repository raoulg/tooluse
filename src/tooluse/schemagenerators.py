import inspect
import json
from abc import ABC, abstractmethod
from inspect import Parameter, getsource, signature
from string import Template
from typing import TYPE_CHECKING, Any, Callable, Dict, List, get_type_hints

from loguru import logger

if TYPE_CHECKING:
    from tooluse.llm import LLMClient

# Template for each parameter's details
PARAMETER_TEMPLATE = Template("""- ${name}:
  Annotation: ${annotation}
  Type hint: ${type_hint}
  Default: ${default}
  Parameter kind: ${kind}""")

# Main prompt template
SCHEMA_PROMPT_TEMPLATE = Template("""Given this Python function information:

Function name: ${name}
Signature: ${signature}
Return type: ${return_type}

Parameters:
${param_details}

${source_info}
${doc_info}

Basic schema: ${basic_schema}

Please provide:
1. A clear, detailed description of what this function does
2. For each parameter, provide:
   - A detailed description
   - Any constraints or expectations
   - Example values
   - Any additional type information not captured in the basic schema

Format your response as a JSON object with this structure:
{
    "description": "main function description",
    "parameters": {
        "param_name": {
            "description": "detailed description",
            "examples": ["example1", "example2"],
            "additional_type_info": "any complex typing information"
        }
    }
}
Respond with only this schema, and nothing else
""")


class ToolSchema:
    """Represents the schema for a tool, following Anthropic's format"""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        required: List[str],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.required = required

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Anthropic's expected format"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required,
            },
        }

    def __repr__(self) -> str:
        dict_schema = self.to_dict()
        return f"ToolSchema({dict_schema})"

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string for easy viewing/editing"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "ToolSchema":
        """Create schema from JSON string"""
        data = json.loads(json_str)
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=data["parameters"]["properties"],
            required=data["parameters"]["required"],
        )


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
        # Add more type mappings as needed
    }

    def generate_schema(self, func: Callable) -> ToolSchema:
        sig = signature(func)
        type_hints = get_type_hints(func)

        parameters = {}
        required = []

        for name, param in sig.parameters.items():
            param_type = type_hints.get(name, Any)

            # Skip self parameter for methods
            if name == "self":
                continue

            param_schema = self._TYPE_MAP.get(param_type, {"type": "string"})

            # Handle default values
            if param.default is Parameter.empty:
                required.append(name)

            parameters[name] = param_schema

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

    def get_function_info(self, func: Callable) -> dict:
        """Gather all available function information"""
        sig = signature(func)
        type_hints = get_type_hints(func)

        # Get detailed parameter info
        param_details = []
        for name, param in sig.parameters.items():
            details = PARAMETER_TEMPLATE.substitute(
                name=name,
                annotation=str(param.annotation),
                type_hint=str(type_hints.get(name, "No type hint")),
                default="None" if param.default is param.empty else str(param.default),
                kind=str(param.kind),
            )
            param_details.append(details)

        info = {
            "name": func.__name__,
            "signature": str(sig),
            "param_details": "\n".join(param_details),
            "return_type": str(type_hints.get("return", "No return type hint")),
        }

        # Try to get source, but don't fail if we can't
        try:
            info["source_info"] = f"\nSource code:\n{getsource(func)}"
        except (TypeError, OSError):
            info["source_info"] = ""

        # Add docstring if available
        doc = func.__doc__ or ""
        info["doc_info"] = f"\nDocumentation: {doc}" if doc else ""
        logger.debug(f"Function info: {info}")

        return info

    def generate_schema(self, func: Callable) -> ToolSchema:
        # First get basic schema for structure
        basic_schema = self.basic_generator.generate_schema(func)

        # Get all available function info
        info = self.get_function_info(func)
        # Add basic schema to info
        info["basic_schema"] = json.dumps(basic_schema.to_dict(), indent=2)

        # Create prompt using template
        content = SCHEMA_PROMPT_TEMPLATE.substitute(info)
        messages = [{"role": "user", "content": content}]

        try:
            response = self.llm(messages)

            # Get response content
            content = response.message.content

            # Try to find and parse JSON from the response
            try:
                enhanced = json.loads(content)
            except json.JSONDecodeError:
                import re

                json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
                if not json_match:
                    return basic_schema
                enhanced = json.loads(json_match.group(1))

            # Update schema while preserving types from basic schema
            if "description" in enhanced:
                basic_schema.description = enhanced["description"]

            if "parameters" in enhanced:
                for param_name, param_info in enhanced["parameters"].items():
                    if param_name in basic_schema.parameters:
                        # Start with original type information
                        param_schema = basic_schema.parameters[param_name]

                        # Add enhanced information
                        param_schema.update(
                            {
                                "description": param_info.get("description", ""),
                                "examples": param_info.get("examples", []),
                            }
                        )

                        # Add any additional type information if provided
                        if "additional_type_info" in param_info:
                            param_schema["type_details"] = param_info[
                                "additional_type_info"
                            ]

                        basic_schema.parameters[param_name] = param_schema

            return basic_schema

        except Exception as e:
            print(f"LLM schema enhancement failed: {e}")
            return basic_schema
