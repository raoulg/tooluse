def calculator(operation: str, operand1: float, operand2: float) -> float:
    if operation == "add":
        return operand1 + operand2
    elif operation == "subtract":
        return operand1 - operand2
    elif operation == "multiply":
        return operand1 * operand2
    elif operation == "divide":
        if operand2 == 0:
            raise ValueError("Cannot divide by zero.")
        return operand1 / operand2
    else:
        raise ValueError(f"Unsupported operation: {operation}")


calculator_schema = {
    "name": "calculator",
    "description": "A simple calculator that performs basic arithmetic operations.",
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "The arithmetic operation to perform.",
            },
            "operand1": {"type": "number", "description": "The first operand."},
            "operand2": {"type": "number", "description": "The second operand."},
        },
        "required": ["operation", "operand1", "operand2"],
    },
}


def add(x: float, y: float) -> float:
    """
    Add x plus y
    """
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        x = float(x)
        y = float(y)
    return x + y


def subtract(x: float, y: float) -> float:
    """
    Subtract y from x
    """
    return x - y
