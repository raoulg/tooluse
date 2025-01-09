from anthropic import Anthropic
from anthropic.types import TextBlock, ToolUseBlock
from dotenv import load_dotenv


def haiku(client) -> str:
    msg = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": "Hi there! Please write me a haiku about a pet chicken",
            }
        ],
    )
    return msg.content[0].text


def calculator(operation, operand1, operand2):
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


def calculator_tool(tool_use):
    tool_inputs = tool_use.input
    operation = tool_inputs["operation"]
    operand1 = tool_inputs["operand1"]
    operand2 = tool_inputs["operand2"]
    result = calculator(operation, operand1, operand2)
    return result


def query(client: Anthropic, content: str, model: str, tools):
    response = client.messages.create(
        model=model,
        system="You have access to tools, but only use them when necessary. If a tool is not required, respond as normal",
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        max_tokens=400,
        tools=tools,
    )
    return response


def reaction(response) -> None:
    if response.stop_reason == "tool_use":
        tool_use = response.content[-1]
        assert isinstance(tool_use, ToolUseBlock)
        if tool_use.name == "calculator":
            print("Claude wants to use the calculator tool")
            try:
                result = calculator_tool(tool_use)
                print(f"Result is: {result}")
                print(response.content[0])
            except ValueError as e:
                print(f"Error: {e}")
    elif response.stop_reason == "end_turn":
        print("Claude did not want to use a tool")
        assert isinstance(response.content[0], TextBlock)
        print(response.content[0].text)


def main():
    load_dotenv()
    client = Anthropic()
    model = "claude-3-haiku-20240307"
    content = "Multiply 1984135 by 9343116. Only respond with the result"
    tools = [calculator_schema]
    response = query(client, content, model, tools)
    reaction(response)
    content = "Give me a haiku about the disclosure of alien presence"
    response = query(client, content, model, tools)
    reaction(response)


if __name__ == "__main__":
    main()
