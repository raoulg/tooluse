"""
Test script for MCP tool integration
Connects to the calculator server and tests tool operations.
"""
import asyncio
from pathlib import Path

from loguru import logger
from llm_tooluse import MCPToolLoader, ToolCollection
import sys

logger.remove()
logger.add(sys.stderr, level="DEBUG")

async def test_basic_connection():
    """Test connecting to calculator server and listing tools"""
    logger.info("=== Test 1: Basic Connection ===")

    loader = MCPToolLoader()

    # Connect to calculator server
    calculator_path = Path("demo/servers/calc_server.py").resolve()


    collection = await loader.load_server(
        name="calculator",
        target=calculator_path,
    )

    logger.info(f"Loaded tools: {collection}")
    logger.info(f"Number of tools: {len(collection)}")

    # List schemas
    schemas = collection.get_schemas()
    for schema in schemas:
        logger.info(f"Tool Schema {schema.name}")

    await loader.cleanup()
    logger.success("Basic connection test passed!")
    return collection


async def test_tool_execution():
    """Test executing tools"""
    logger.info("\n=== Test 2: Tool Execution ===")

    loader = MCPToolLoader()

    calculator_path = Path("demo/servers/calc_server.py").resolve()
    collection = await loader.load_server(
        name="calculator",
        target=calculator_path,
    )

    # Test addition
    result = await collection("add", a=5, b=3)
    logger.info(f"add(5, 3) = {result}")
    assert float(result) == 8.0, f"Expected 8.0, got {result}"

    # Test subtraction
    result = await collection("subtract", a=10, b=4)
    logger.info(f"subtract(10, 4) = {result}")
    assert float(result) == 6.0, f"Expected 6.0, got {result}"

    # Test multiplication
    result = await collection("multiply", a=7, b=6)
    logger.info(f"multiply(7, 6) = {result}")
    assert float(result) == 42.0, f"Expected 42.0, got {result}"

    # Test division
    result = await collection("divide", a=20, b=5)
    logger.info(f"divide(20, 5) = {result}")
    assert float(result) == 4.0, f"Expected 4.0, got {result}"

    logger.success("\n=== Test 2: All calculations passed!")

    await loader.cleanup()


async def test_set_operations():
    """Test collection set operations"""
    logger.info("\n=== Test 3: Set Operations ===")

    loader = MCPToolLoader()
    calculator_path = Path("demo/servers/calc_server.py").resolve()
    full_collection = await loader.load_server(
        name="calculator",
        target=calculator_path,
    )

    logger.info(f"Full collection: {full_collection}")

    # Create subset with only add and subtract
    subset = full_collection - ["multiply", "divide"]
    logger.info(f"Subset (add, subtract): {subset}")
    assert len(subset) == 2
    assert "add" in subset
    assert "subtract" in subset
    assert "multiply" not in subset
    assert "divide" not in subset

    # Test that we can still call tools in subset
    result = await subset("add", a=1, b=2)
    logger.info(f"Subset call: add(1, 2) = {result}")
    assert float(result) == 3.0

    # Test that we can't call removed tools
    try:
        await subset("multiply", a=2, b=3)
        logger.error("Should not be able to call removed tool!")
        assert False
    except ValueError as e:
        logger.info(f"Correctly blocked removed tool: {e}")

    logger.success("=== Test 3: Set operations work correctly!")

    await loader.cleanup()


async def test_collection_union():
    """Test combining collections"""
    logger.info("\n=== Test 4: Collection Union ===")

    loader = MCPToolLoader()

    calculator_path = Path("demo/servers/calc_server.py").resolve()
    collection = await loader.load_server(
        name="calculator",
        target=calculator_path,
    )

    # Create two subsets
    subset1 = collection - ["multiply", "divide"]
    subset2 = collection - ["add", "subtract"]

    logger.info(f"Subset 1: {subset1}")
    logger.info(f"Subset 2: {subset2}")

    # Combine them
    combined = subset1 * subset2
    logger.info(f"Combined: {combined}")

    assert len(combined) == 4
    assert "add" in combined
    assert "subtract" in combined
    assert "multiply" in combined
    assert "divide" in combined

    logger.success("=== Test 4: Collection union works!")

    await loader.cleanup()


async def main():
    """Run all tests"""
    logger.info("Starting MCP Tool Integration Tests\n")

    try:
        await test_basic_connection()
        await test_tool_execution()
        await test_set_operations()
        await test_collection_union()

        logger.success("\n✅ All tests passed!")

    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())