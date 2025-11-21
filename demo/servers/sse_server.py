from fastmcp import FastMCP


mcp = FastMCP("online_calculator")

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
    # docker build -t sse_example demo/
    # docker run -d -p 8000:8000 sse_example
    mcp.run(transport="http", host="0.0.0.0", port=8000)