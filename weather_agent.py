import asyncio
from openai_mcp_agent import Agent, InferenceClient

# This file demonstrates using the Agent with an explicit InferenceClient
# Updated to use the new InferenceClient class for configurable inference providers

def make_current_weather_agent():
    """
    Create a weather agent with explicit OpenAI inference client.
    This demonstrates how to configure the inference provider explicitly.
    """
    # Create an explicit OpenAI inference client
    # This allows for customization of model, API key, base URL, etc.
    inference_client = InferenceClient(
        model="gpt-4o",  # Specify the model explicitly
        # api_key="your-api-key",  # Optional: override env var
        # base_url="https://api.openai.com/v1",  # Optional: custom base URL
    )
    
    # Create agent with explicit inference client and tool filtering
    return Agent(
        server_spec="weather_mcp_server.py", 
        tools=["get_current_weather"],
        inference_client=inference_client
    )

def make_weather_agent_with_custom_model():
    """
    Create a weather agent using a different model (gpt-4o-mini for faster/cheaper responses).
    """
    inference_client = InferenceClient(model="gpt-4o-mini")
    return Agent(
        server_spec="weather_mcp_server.py",
        tools=["get_current_weather"], 
        inference_client=inference_client
    )

def make_remote_weather_agent(host="localhost", port=3000):
    """
    Create a weather agent that connects to a remote MCP weather server via SSE.
    
    Args:
        host: Host where the remote weather server is running
        port: Port where the remote weather server is running
        
    Usage:
        1. Start the remote server: python weather_mcp_server.py --transport sse
        2. Use this function to connect to it
    """
    inference_client = InferenceClient(model="gpt-4o-mini")
    return Agent(
        server_spec=f"remote:http://{host}:{port}",
        tools=["get_current_weather"],
        inference_client=inference_client
    )

# Example of how to use Azure OpenAI (commented out since it requires actual Azure credentials)
def make_azure_weather_agent():
    """
    Example of creating a weather agent with Azure OpenAI.
    Uncomment and configure with actual Azure credentials to use.
    """
    # azure_client = InferenceClient(
    #     api_key="your-azure-api-key",
    #     base_url="https://your-resource.openai.azure.com/openai/v1",
    #     model="gpt-4o"
    # )
    # return Agent(
    #     server_spec="weather_mcp_server.py",
    #     tools=["get_current_weather"],
    #     inference_client=azure_client
    # )
    pass

# Simple test main function
async def main():
    print("=== Weather Agent with Explicit InferenceClient ===\n")
    
    # Test 1: Standard OpenAI with explicit client
    print("1. Testing with explicit OpenAI client (gpt-4o):")
    async with make_current_weather_agent() as agent:
        print(f"   Using model: {agent.inference_client.model}")
        print(f"   Using base URL: {agent.inference_client.base_url or 'OpenAI default'}")
        response = await agent.chat("What's the weather in Boston?")
        print(f"   Response: {response}\n")
    
    # Test 2: Using gpt-4o-mini for faster responses
    print("2. Testing with gpt-4o-mini for faster responses:")
    async with make_weather_agent_with_custom_model() as agent:
        print(f"   Using model: {agent.inference_client.model}")
        response = await agent.chat("What's the weather in San Francisco?")
        print(f"   Response: {response}\n")
    
    # Test 3: Remote server (optional - requires separate server process)
    print("3. Testing remote MCP server connection (optional):")
    print("   Note: This requires running the weather server in SSE mode:")
    print("   python weather_mcp_server.py --transport sse")
    print("   Skipping remote test for now - enable manually if server is running")
    # 
    # Uncomment below to test remote server (ensure server is running first):
    # try:
    #     async with make_remote_weather_agent() as agent:
    #         print(f"   Connected to remote server at localhost:3000")
    #         print(f"   Using model: {agent.inference_client.model}")
    #         response = await agent.chat("What's the weather in New York?")
    #         print(f"   Response: {response}\n")
    # except Exception as e:
    #     print(f"   Could not connect to remote server: {e}\n")
    
    print("=== Tests completed ===")

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    if exit_code:
        exit(exit_code)