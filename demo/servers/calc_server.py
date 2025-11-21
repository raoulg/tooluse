"""
Minimal Calculator MCP Server
A simple MCP server with basic calculator tools for testing.
"""
from fastmcp import FastMCP

# Create the FastMCP server
mcp = FastMCP("calculator")


@mcp.tool()
async def add(a: float, b: float) -> float:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Sum of a and b
    """
    return a + b


@mcp.tool()
async def subtract(a: float, b: float) -> float:
    """Subtract b from a.

    Args:
        a: First number
        b: Number to subtract

    Returns:
        Difference of a and b
    """
    return a - b


@mcp.tool()
async def multiply(a: float, b: float) -> float:
    """Multiply two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        Product of a and b
    """
    return a * b


@mcp.tool()
async def divide(a: float, b: float) -> float:
    """Divide a by b.

    Args:
        a: Numerator
        b: Denominator

    Returns:
        Quotient of a and b

    Raises:
        ValueError: If b is zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


if __name__ == "__main__":
    # Run the server using stdio transport
    mcp.run()