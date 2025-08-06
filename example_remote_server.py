#!/usr/bin/env python3
"""
Example demonstrating remote MCP server usage with the OpenAI MCP Agent.

This script shows how to:
1. Connect to remote MCP servers via HTTP and SSE
2. Use different transport types
3. Handle remote server configurations
4. Apply tool filtering with remote servers

Requirements:
- Set OPENAI_API_KEY environment variable
- Have a remote MCP server running (e.g., weather_mcp_server.py --transport sse)
- Install dependencies: pip install openai mcp httpx

Usage:
    # Start a remote server first:
    python weather_mcp_server.py --transport sse --port 8000
    
    # Then run this example:
    python example_remote_server.py
"""

import asyncio
from openai_mcp_agent import Agent, RemoteMCPServerConfig, InferenceClient

async def example_http_server():
    """Example connecting to HTTP-based remote MCP server."""
    
    print("🌐 HTTP Remote Server Example")
    print("=" * 50)
    
    # Create remote server configuration
    config = RemoteMCPServerConfig(
        url="http://localhost:8000",
        transport_type="sse",  # Use SSE transport
        tools=["get_current_weather"]  # Filter to specific tools
    )
    
    agent = Agent(server_configs=config)
    
    try:
        await agent.initialize()
        print(f"Connected to remote server at {config.url}")
        print(f"Available tools: {agent.all_tool_names}")
        
        response = await agent.chat("What's the weather in San Francisco?")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the remote server is running:")
        print("python weather_mcp_server.py --transport sse --port 8000")
    
    finally:
        await agent.cleanup()

async def example_sse_server():
    """Example connecting to SSE-based remote MCP server."""
    
    print("\n📡 SSE Remote Server Example")
    print("=" * 50)
    
    # Create remote server configuration for SSE
    config = RemoteMCPServerConfig(
        url="http://localhost:8000/sse",  # Explicit SSE endpoint
        transport_type="sse"
    )
    
    agent = Agent(server_configs=config)
    
    try:
        await agent.initialize()
        print(f"Connected via SSE to {config.url}")
        print(f"Available tools: {agent.all_tool_names}")
        
        response = await agent.chat("Check the weather in London")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the remote server is running with SSE transport")
    
    finally:
        await agent.cleanup()

async def example_server_specs():
    """Example using server specification strings for remote servers."""
    
    print("\n🔧 Server Specs Example")
    print("=" * 50)
    
    try:
        # Simple remote server spec
        async with Agent(server_specs="remote:http://localhost:8000") as agent:
            print("Connected using server spec string")
            response = await agent.chat("What's the weather in Tokyo?")
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

async def example_custom_inference_client():
    """Example using custom inference client with remote server."""
    
    print("\n🤖 Custom Inference Client Example")
    print("=" * 50)
    
    # Use faster/cheaper model for testing
    inference_client = InferenceClient(model="gpt-4o-mini")
    
    agent = Agent(server_specs="remote:http://localhost:8000")
    
    try:
        await agent.initialize()
        print(f"Using model: {agent.inference_client.model}")
        
        response = await agent.chat("Brief weather update for Paris")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        await agent.cleanup()

async def example_tool_filtering():
    """Example demonstrating tool filtering with remote servers."""
    
    print("\n🔍 Tool Filtering Example")
    print("=" * 50)
    
    try:
        # Method 1: Filter in server spec
        agent1 = Agent(server_specs="remote:http://localhost:8000[get_weather,get_forecast]")
        
        # Method 2: Filter via tools parameter
        agent2 = Agent(
            server_specs="remote:http://localhost:8000",
            tools=["get_current_weather"]
        )
        
        # Method 3: Filter via config object
        config = RemoteMCPServerConfig(
            url="http://localhost:8000",
            tools=["get_current_weather"]
        )
        agent3 = Agent(server_configs=config)
        
        print("Created agents with different tool filtering approaches")
        print("(Not connecting to avoid multiple connections)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def example_error_handling():
    """Example demonstrating error handling with remote servers."""
    
    print("\n⚠️ Error Handling Example")
    print("=" * 50)
    
    # Try connecting to non-existent server
    try:
        agent = Agent(server_specs="remote:http://localhost:9999")
        await agent.initialize()
        
    except Exception as e:
        print(f"❌ Expected error connecting to non-existent server: {e}")
    
    # Try invalid URL format
    try:
        agent = Agent(server_specs="remote:invalid-url")
        await agent.initialize()
        
    except Exception as e:
        print(f"❌ Expected error with invalid URL: {e}")

def print_setup_instructions():
    """Print instructions for setting up remote server."""
    
    print("🚀 Remote MCP Server Setup Instructions")
    print("=" * 60)
    print()
    print("To run these examples, you need a remote MCP server running.")
    print("Here's how to start one:")
    print()
    print("1. Start the weather MCP server in SSE mode:")
    print("   python weather_mcp_server.py --transport sse --port 8000")
    print()
    print("2. The server will be available at:")
    print("   - HTTP: http://localhost:8000")
    print("   - SSE:  http://localhost:8000/sse")
    print()
    print("3. Then run this example script:")
    print("   python example_remote_server.py")
    print()
    print("Note: Make sure OPENAI_API_KEY is set in your environment")
    print("=" * 60)

async def main():
    """Run all remote server examples."""
    
    print_setup_instructions()
    
    print("\n" + "=" * 60)
    print("Running Remote Server Examples")
    print("=" * 60)
    
    # Note: These examples assume a remote server is running
    # If no server is available, they will show connection errors
    
    examples = [
        example_http_server,
        example_sse_server,
        example_server_specs,
        example_custom_inference_client,
        example_tool_filtering,
        example_error_handling,
    ]
    
    for example in examples:
        try:
            await example()
            print("\n" + "─" * 50)
            
        except Exception as e:
            print(f"❌ Example failed: {e}")
            print("─" * 50)
    
    print("\n✅ All remote server examples completed!")
    print("\nNote: Connection errors are expected if no remote server is running.")
    print("Start a remote server to see successful connections.")

if __name__ == "__main__":
    asyncio.run(main()) 