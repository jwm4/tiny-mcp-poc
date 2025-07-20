import asyncio
from openai_mcp_agent import Agent, InferenceClient

# Mostly generate by Cursor using Claude 4 Sonnet.

def make_remote_weather_agent(
    mcp_host="localhost",
    mcp_port=3000,
    inference_base_url="https://api.openai.com/v1",
    inference_model="gpt-4o",
):
    """
    Create a weather agent that connects to a remote MCP weather server via SSE.

    Args:
        host: Host where the remote weather server is running
        port: Port where the remote weather server is running

    Usage:
        1. Start the remote server: python weather_mcp_server.py --transport sse
        2. Use this function to connect to it
    """
    inference_client = InferenceClient(
        model=inference_model
    )
    return Agent(
        server_spec=f"remote:http://{mcp_host}:{mcp_port}",
        tools=["get_current_weather"],
        inference_client=inference_client,
    )


# Simple test main function
async def main():
    try:
        async with make_remote_weather_agent() as agent:
            print(f"   Using model: {agent.inference_client.model}")
            print(
                f"   Using base URL: {agent.inference_client.base_url or 'OpenAI default'}"
            )
            response = await agent.chat("What's the weather in Boston?")
            print(f"   Response: {response}\n")
    except Exception as e:
        print(f"   Could not connect to remote server: {e}\n")

    print("=== Tests completed ===")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    if exit_code:
        exit(exit_code)
