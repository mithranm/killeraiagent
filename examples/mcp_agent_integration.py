# killeraiagent/mcp_agent_integration.py

import asyncio
import json
from killeraiagent.mcp_client import MCPClient
from killeraiagent.model import create_model  # from your existing models.py

def extract_tool_call(response_text: str):
    """
    Extract a tool call from the LLM response text.
    This function looks for a marker in the text indicating a tool call.
    Format convention: the response should include a substring in the form:
      {{TOOL_CALL: {"tool": "tool_name", "args": { ... }} }}
    Returns the parsed JSON dict if found; otherwise, returns None.
    """
    start_marker = "{{TOOL_CALL:"
    end_marker = "}}"
    start_idx = response_text.find(start_marker)
    if start_idx == -1:
        return None
    start_idx += len(start_marker)
    end_idx = response_text.find(end_marker, start_idx)
    if end_idx == -1:
        return None
    json_str = response_text[start_idx:end_idx].strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

async def integration_loop():
    # Initialize an LLM via your models module.
    # The create_model function is assumed to create an instance (e.g., of LlamaCppLLM) with a generate() method.
    # Replace "path/to/model.gguf" with your actual model path and "llamacpp" with your backend.
    llm = create_model(model_path="path/to/model.gguf", backend="llamacpp")
    if not llm.is_loaded:
        llm.load()
    
    # Instantiate the generic MCP client to connect to the MCP server.
    mcp_client = MCPClient("http://localhost:8001")
    
    # Cache the available tools from the server.
    tools_info = await mcp_client.list_tools(force_refresh=True)
    print("Indexed Tools from MCP Server:")
    print(json.dumps(tools_info, indent=2))
    
    # Start a simple conversation. Here we simulate an initial input that prompts the LLM.
    conversation_history = "User: What is the weather like today?\n"
    
    # For the sake of demonstration, assume the LLM's generate() method returns a string.
    # We simulate that the LLM sometimes outputs an instruction to call a tool in the following format:
    #   "... some text ... {{TOOL_CALL: {\"tool\": \"search\", \"args\": {\"query\": \"weather forecast\"}}}} ..."
    #
    # In a real integration, the LLM might output natural language and you’d use further processing.
    
    simulated_input = conversation_history + "Please provide the weather forecast.\n" \
                      "{{TOOL_CALL: {\"tool\": \"search\", \"args\": {\"query\": \"weather forecast\"}}}}"
    
    # Call the LLM generate method with the simulated input.
    llm_response_text, _ = llm.generate(simulated_input)
    print("LLM Response:")
    print(llm_response_text)
    
    # Check if the LLM response includes a tool call.
    tool_call = extract_tool_call(llm_response_text)
    if tool_call:
        tool_name = tool_call.get("tool")
        tool_args = tool_call.get("args", {})
        print(f"LLM requested tool call: {tool_name} with arguments: {tool_args}")
        
        # Use the MCP client to call the requested tool.
        tool_result = await mcp_client.call_tool(tool_name, tool_args)
        print("Result from tool call:")
        print(json.dumps(tool_result, indent=2))
        
        # Append the result into the conversation history and ask the LLM to generate a final answer.
        conversation_history += f"Tool ({tool_name}) output: {tool_result.get('result')}\n"
        final_response_text, _ = llm.generate(conversation_history)
        print("LLM Final Response:")
        print(final_response_text)
    else:
        print("No tool call detected in the LLM response.")
    
    await mcp_client.close()

if __name__ == "__main__":
    asyncio.run(integration_loop())
