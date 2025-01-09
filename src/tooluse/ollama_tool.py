from ollama import ChatResponse, chat


def add_two_numbers(a: int, b: int) -> int:
    """
    Add two numbers

    Args:
      a (int): The first number
      b (int): The second number

    Returns:
      int: The sum of the two numbers
    """
    return a + b


def subtract_two_numbers(a: int, b: int) -> int:
    """
    Subtract two numbers
    """
    return a - b


subtract_two_numbers_tool = {
    "type": "function",
    "function": {
        "name": "subtract_two_numbers",
        "description": "Subtract two numbers",
        "parameters": {
            "type": "object",
            "required": ["a", "b"],
            "properties": {
                "a": {"type": "integer", "description": "The first number"},
                "b": {"type": "integer", "description": "The second number"},
            },
        },
    },
}


def reaction(response, messages, available_functions):
    print("\nStart reaction ======")
    print("[DEBUG] Response:", response)
    print("[DEBUG] messages:", messages)
    if response.message.tool_calls:
        print("[DEBUG] Tool calls:", response.message.tool_calls)
        # There may be multiple tool calls in the response
        for tool in response.message.tool_calls:
            # Ensure the function is available, and then call it
            if function_to_call := available_functions.get(tool.function.name):
                print("Calling function:", tool.function.name)
                print("Arguments:", tool.function.arguments)
                output = function_to_call(**tool.function.arguments)
                print("Function output:", output)
            else:
                print("Function", tool.function.name, "not found")

    # Only needed to chat with the model using the tool call results
    if response.message.tool_calls:
        # Add the function response to messages for the model to use
        messages.append(response.message)
        messages.append(
            {"role": "tool", "content": str(output), "name": tool.function.name}
        )

        # Get final response from model with function outputs
        print("[DEBUG] sending messages to model:", messages)
        final_response = chat("llama3.1", messages=messages)
        print("\n\nFinal response:", final_response.message.content)

    else:
        print("\nNo tool calls in response")
        print("[DEBUG] sending messages to model:", messages)
        print(response.message)
        final_response = chat("llama3.1", messages=messages)
        print("\nFinal response:", final_response.message.content)
        print("====\n")


def main():
    messages = [{"role": "user", "content": "What is three plus one?"}]
    print("\nPrompt:", messages[0]["content"])

    available_functions = {
        "add_two_numbers": add_two_numbers,
        "subtract_two_numbers": subtract_two_numbers,
    }

    response: ChatResponse = chat(
        "llama3.1",
        messages=messages,
        tools=[add_two_numbers, subtract_two_numbers],
    )
    reaction(response, messages, available_functions)

    messages = [
        {
            "role": "system",
            "content": "You have access to tools, but only use them when necessary. If a tool is not required, respond as normal",
        },
        {
            "role": "user",
            "content": "Give me a haiku about the disclosure of alien presence",
        },
    ]
    print("\nPrompt:", messages[0]["content"])
    response: ChatResponse = chat(
        "llama3.1",
        messages=messages,
    )
    reaction(response, messages, available_functions)


if __name__ == "__main__":
    main()
