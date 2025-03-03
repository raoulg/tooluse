import json
from typing import Any, Optional, Union
from unittest.mock import patch

import pytest
from loguru import logger

from tooluse.schemagenerators import ToolSchema


def test_basic_schema_creation(basic_schema):
    """Test basic creation of ToolSchema"""
    assert basic_schema.name == "test_function"
    assert basic_schema.description == "A test function"
    assert basic_schema.parameters == {"param1": {"type": "string"}}
    assert basic_schema.required == ["param1"]


def test_schema_to_dict(basic_schema):
    """Test conversion of schema to dictionary format"""
    expected_dict = {
        "name": "test_function",
        "description": "A test function",
        "parameters": {"param1": {"type": "string"}},
        "required": ["param1"],
        }

    assert basic_schema.to_dict() == expected_dict


def test_schema_json_serialization(basic_schema):
    """Test JSON serialization and deserialization"""
    json_str = basic_schema.to_json()
    recreated_schema = ToolSchema.from_json(json_str)

    assert basic_schema.to_dict() == recreated_schema.to_dict()


def test_simple_function_schema(basic_generator):
    """Test schema generation for a simple function"""

    def test_func(a: int, b: str, c: float = 0.0) -> str:
        """Test function docstring"""
        return f"{a} {b} {c}"

    schema = basic_generator.generate_schema(test_func)

    assert schema.name == "test_func"
    assert schema.description == "Test function docstring"
    assert set(schema.required) == {"a", "b"}
    assert schema.parameters["a"] == {"type": "integer"}
    assert schema.parameters["b"] == {"type": "string"}
    assert schema.parameters["c"] == {"type": "number"}


def test_method_schema(basic_generator):
    """Test schema generation for a class method"""

    class TestClass:
        def test_method(self, x: int) -> None:
            pass

    schema = basic_generator.generate_schema(TestClass().test_method)

    assert schema.name == "test_method"
    assert len(schema.parameters) == 1
    assert "self" not in schema.parameters


def test_unknown_type_handling(basic_generator):
    """Test handling of unknown type annotations"""
    from typing import Generic, TypeVar

    T = TypeVar("T")

    class GenericContainer(Generic[T]):
        def __init__(self, value: T):
            self.value = value

    def func_with_generic_type(x: GenericContainer[str], y: T, z: Any):
        pass

    schema = basic_generator.generate_schema(func_with_generic_type)

    # Generic types and TypeVars should default to string type
    assert schema.parameters["x"] == {"type": "string"}
    assert schema.parameters["y"] == {"type": "string"}
    assert schema.parameters["z"] == {"type": "string"}


@patch("inspect.getsource")
def test_successful_llm_enhancement(mock_getsource, llm_mock_generator, mock_llm):
    """Test successful LLM enhancement of schema"""

    def test_func(x: int) -> str:
        """Test function"""
        return str(x)

    mock_getsource.return_value = "def test_func(x: int) -> str: return str(x)"

    # Mock LLM response
    enhanced_schema = {
        "description": "Enhanced description",
        "parameters": {
            "x": {
                "description": "Enhanced parameter description",
            }
        },
    }
    mock_llm.return_value.message.content = json.dumps(enhanced_schema)

    schema = llm_mock_generator.generate_schema(test_func)

    assert schema.description == "Enhanced description"
    assert "description" in schema.parameters["x"]


@patch("inspect.getsource")
def test_llm_failure_fallback(mock_getsource, llm_mock_generator, mock_llm):
    """Test fallback to basic schema when LLM fails"""

    def test_func(x: int) -> str:
        return str(x)

    mock_getsource.return_value = "def test_func(x: int) -> str: return str(x)"
    mock_llm.side_effect = Exception("LLM failed")

    schema = llm_mock_generator.generate_schema(test_func)

    # Should still get a valid basic schema
    assert isinstance(schema, ToolSchema)
    assert "x" in schema.parameters
    assert schema.parameters["x"]["type"] == "integer"

# Example of parametrized test
@pytest.mark.parametrize(
    "input_type,expected_type",
    [
        (int, {"type": "integer"}),
        (str, {"type": "string"}),
        (float, {"type": "number"}),
        (bool, {"type": "boolean"}),
        (list[int], {"type": "string"}),  # Complex types default to string
        (dict[str, int], {"type": "string"}),
        (Optional[int], {"type": "string"}),
        (Union[str, int], {"type": "string"}),
    ],
)
def test_type_mapping(basic_generator, input_type, expected_type):
    """Test type mapping for different Python types"""

    def test_func(x: input_type):
        pass

    schema = basic_generator.generate_schema(test_func)
    logger.info(schema)
    assert schema.parameters["x"]["type"] == expected_type["type"]
