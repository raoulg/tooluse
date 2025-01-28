from ollama import chat


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
