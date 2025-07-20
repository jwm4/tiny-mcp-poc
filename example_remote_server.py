#!/usr/bin/env python3
"""
Example demonstrating remote MCP server support in the OpenAI MCP Agent.

This script shows how to connect to remote MCP servers using different transport types:
1. HTTP transport for standard HTTP-based MCP servers
2. SSE (Server-Sent Events) transport for real-time connections
3. Configuration parsing for different URL formats
"""

import asyncio
from openai_mcp_agent import (
    Agent,
    RemoteMCPServerConfig,
    parse_server_config
)


async def example_http_remote_server():
    """Example 1: Connecting to HTTP remote MCP server"""
    print("=" * 60)
    print("Example 1: HTTP Remote MCP Server")
    print("=" * 60)
    
    try:
        # Create remote server config for HTTP transport
        config = RemoteMCPServerConfig(
            "http://localhost:8000",
            transport_type="sse",  # Use SSE for real-time communication
            tools=["get_weather"]  # Optional: filter tools
        )
        
        # Create agent with remote server
        agent = Agent(server_config=config)
        print(f"✓ Agent created for HTTP remote server")
        print(f"  Server: {config.url}")
        print(f"  Transport: {config.transport_type}")
        print(f"  Tools: {config.tools}")
        
        # Note: We won't actually initialize since we don't have a real server
        print("  (Not initializing - no real server available)")
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_sse_remote_server():
    """Example 2: Connecting to SSE remote MCP server"""
    print("\n" + "=" * 60)
    print("Example 2: SSE Remote MCP Server")
    print("=" * 60)
    
    try:
        # Create remote server config for SSE transport
        config = RemoteMCPServerConfig(
            "https://api.example.com/mcp/sse",
            transport_type="sse"
        )
        
        # Create agent with remote server
        agent = Agent(server_config=config)
        print(f"✓ Agent created for SSE remote server")
        print(f"  Server: {config.url}")
        print(f"  Transport: {config.transport_type}")
        
        # Note: We won't actually initialize since we don't have a real server
        print("  (Not initializing - no real server available)")
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_remote_server_parsing():
    """Example 3: Remote server configuration parsing"""
    print("\n" + "=" * 60)
    print("Example 3: Remote Server Configuration Parsing")
    print("=" * 60)
    
    test_specs = [
        "remote:http://localhost:8000",
        "remote:https://api.example.com/mcp",
        "remote:http://localhost:8000/sse",
        "remote:ws://localhost:8000",  # WebSocket (not supported yet)
        "remote:http://localhost:3000[get_weather,get_forecast]",
    ]
    
    for spec in test_specs:
        try:
            config = parse_server_config(spec)
            print(f"✓ '{spec}'")
            print(f"  Type: {type(config).__name__}")
            print(f"  URL: {getattr(config, 'url', 'N/A')}")
            print(f"  Transport: {getattr(config, 'transport_type', 'N/A')}")
            print(f"  Tools: {getattr(config, 'tools', 'All tools')}")
            print()
        except Exception as e:
            print(f"✗ '{spec}': {e}")
            print()


async def example_remote_server_with_context_manager():
    """Example 4: Using remote server with context manager"""
    print("\n" + "=" * 60)
    print("Example 4: Remote Server with Context Manager")
    print("=" * 60)
    
    try:
        # This would be the typical usage pattern
        print("Typical usage pattern:")
        print("""
async with Agent(server_spec="remote:http://localhost:8000") as agent:
    response = await agent.chat("What tools do you have?")
    print(response)
        """)
        
        # Create agent without actually connecting
        agent = Agent(server_spec="remote:http://localhost:8000")
        print(f"✓ Agent created for remote server")
        url = getattr(agent.server_config, 'url', 'N/A')
        transport = getattr(agent.server_config, 'transport_type', 'N/A')
        print(f"  Would connect to: {url}")
        print(f"  Using transport: {transport}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_remote_server_tool_filtering():
    """Example 5: Remote server with tool filtering"""
    print("\n" + "=" * 60)
    print("Example 5: Remote Server with Tool Filtering")
    print("=" * 60)
    
    try:
        # Method 1: Using server spec with embedded tools
        agent1 = Agent(server_spec="remote:http://localhost:8000[get_weather,get_forecast]")
        print(f"✓ Agent with embedded tools: {agent1.server_config.tools}")
        
        # Method 2: Using separate tools parameter
        agent2 = Agent(
            server_spec="remote:http://localhost:8000",
            tools=["get_weather", "get_forecast"]
        )
        print(f"✓ Agent with separate tools: {agent2.server_config.tools}")
        
        # Method 3: Using RemoteMCPServerConfig directly
        config = RemoteMCPServerConfig(
            "http://localhost:8000",
            tools=["get_weather", "get_forecast"]
        )
        agent3 = Agent(server_config=config)
        print(f"✓ Agent with config tools: {agent3.server_config.tools}")
        
    except Exception as e:
        print(f"✗ Error: {e}")


async def example_command_line_usage():
    """Example 6: Command line usage for remote servers"""
    print("\n" + "=" * 60)
    print("Example 6: Command Line Usage")
    print("=" * 60)
    
    print("Command line examples for remote servers:")
    print()
    print("# Basic remote server connection")
    print("python openai_mcp_agent.py --server remote:http://localhost:8000")
    print()
    print("# Remote server with HTTPS")
    print("python openai_mcp_agent.py --server remote:https://api.example.com/mcp")
    print()
    print("# Remote server with SSE endpoint")
    print("python openai_mcp_agent.py --server remote:http://localhost:8000/sse")
    print()
    print("# Remote server with tool filtering")
    print("python openai_mcp_agent.py --server remote:http://localhost:8000 --tools get_weather")
    print()
    print("# Remote server with embedded tools")
    print("python openai_mcp_agent.py --server 'remote:http://localhost:8000[get_weather,get_forecast]'")


async def main():
    """Run all examples"""
    print("OpenAI MCP Agent - Remote Server Examples")
    print("Note: These examples demonstrate configuration but don't connect to real servers")
    print()
    
    await example_http_remote_server()
    await example_sse_remote_server()
    await example_remote_server_parsing()
    await example_remote_server_with_context_manager()
    await example_remote_server_tool_filtering()
    await example_command_line_usage()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
    print()
    print("To test with a real remote server, you would need to:")
    print("1. Set up an MCP server that supports HTTP/SSE transport")
    print("2. Use one of the MCP server proxy tools (like mcp-streamablehttp-proxy)")
    print("3. Replace the example URLs with your actual server URLs")
    print("4. Remove the initialization skip logic in the examples")


if __name__ == "__main__":
    asyncio.run(main()) 