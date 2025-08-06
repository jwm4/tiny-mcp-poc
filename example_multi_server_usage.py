#!/usr/bin/env python3
"""
Example usage of the multi-server OpenAI MCP Agent.

This demonstrates how to connect to multiple MCP servers simultaneously
and use tools from all of them in a single conversation.
"""

import asyncio
from openai_mcp_agent import Agent, LocalMCPServerConfig, RemoteMCPServerConfig


async def example_multi_server_usage():
    """Example showing multi-server agent usage."""
    
    print("=== Multi-Server Agent Example ===\n")
    
    # Example 1: Multiple local servers
    print("1. Creating agent with multiple local servers...")
    
    server_configs = [
        LocalMCPServerConfig("weather_mcp_server.py"),
        LocalMCPServerConfig("nps_mcp_server.py"),
    ]
    
    agent = await Agent.create(server_configs=server_configs)
    if not agent:
        print("Failed to create agent with local servers")
        return
    
    print(f"Agent created with {len(agent.server_infos)} servers")
    print(f"Available tools: {agent.all_tool_names}")
    
    # Test a query that might use tools from different servers
    response = await agent.chat("What's the weather like in San Francisco?")
    print(f"Weather response: {response}")
    
    await agent.cleanup()
    print()
    
    # Example 2: Using server specs (string format)
    print("2. Creating agent with server specs...")
    
    server_specs = [
        "weather_mcp_server.py[get_current_weather]",  # Only use weather tool
        "nps_mcp_server.py",  # Use all tools from NPS server
    ]
    
    agent2 = await Agent.create(server_specs=server_specs)
    if not agent2:
        print("Failed to create agent with server specs")
        return
    
    print(f"Agent created with {len(agent2.server_infos)} servers")
    print(f"Available tools: {agent2.all_tool_names}")
    
    await agent2.cleanup()
    print()
    
    # Example 3: Mixed local and remote servers (if remote server is available)
    print("3. Example configuration for mixed servers (not executed):")
    print("   - Local weather server")
    print("   - Remote HTTP server at localhost:8000")
    print("   - Tool filtering applied")
    
    example_mixed_configs = [
        LocalMCPServerConfig("weather_mcp_server.py", tools=["get_current_weather"]),
        RemoteMCPServerConfig("http://localhost:8000", transport_type="sse"),
    ]
    
    print(f"Would create agent with {len(example_mixed_configs)} servers")
    print()


async def interactive_multi_server_demo():
    """Interactive demo with multiple servers."""
    
    print("=== Interactive Multi-Server Demo ===\n")
    
    # You can modify these server specs based on what's available
    server_specs = [
        "weather_mcp_server.py",
        # Add more servers as needed:
        # "nps_mcp_server.py",
        # "remote:http://localhost:8000",
    ]
    
    agent = await Agent.create(server_specs=server_specs)
    if not agent:
        print("Failed to create multi-server agent")
        return
    
    print(f"Multi-server agent ready with {len(agent.server_infos)} server(s)")
    print(f"Available tools: {agent.all_tool_names}")
    print("\nType 'quit' to exit\n")
    
    try:
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            try:
                response = await agent.chat(user_input)
                print(f"Agent: {response}\n")
            except Exception as e:
                print(f"Error: {e}\n")
    
    finally:
        await agent.cleanup()
        print("Multi-server agent closed.")


if __name__ == "__main__":
    print("Choose demo mode:")
    print("1. Basic multi-server examples")
    print("2. Interactive multi-server demo")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(example_multi_server_usage())
    elif choice == "2":
        asyncio.run(interactive_multi_server_demo())
    else:
        print("Invalid choice. Running basic examples...")
        asyncio.run(example_multi_server_usage()) 