#!/usr/bin/env python3

# This class was written by Cursor using Claude 4 Sonnet.
"""
Example demonstrating the refactored Agent class usage.

This script shows different ways to create and use the Agent class:
1. Using constructor with server_spec
2. Using constructor with server_config object
3. Using factory method Agent.create()
4. Using async context manager
5. Using tool filtering
"""

import asyncio
from openai_mcp_agent import (
    Agent,
    LocalMCPServerConfig,
    parse_server_config
)


async def example_constructor_with_server_spec():
    """Example 1: Using constructor with server specification"""
    print("=" * 60)
    print("Example 1: Constructor with server specification")
    print("=" * 60)
    
    try:
        # Create agent with server spec
        agent = Agent(server_spec="weather_mcp_server.py[get_current_weather]")
        
        # Initialize the agent
        await agent.initialize()
        
        # Use the agent
        response = await agent.chat("What's the weather in London?")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'agent' in locals():
            await agent.cleanup()


async def example_constructor_with_config():
    """Example 2: Using constructor with server config object"""
    print("\n" + "=" * 60)
    print("Example 2: Constructor with server config object")
    print("=" * 60)
    
    try:
        # Create server config
        server_config = LocalMCPServerConfig(
            "weather_mcp_server.py", 
            tools=["get_current_weather"]
        )
        
        # Create agent with config
        agent = Agent(server_config=server_config)
        
        # Initialize the agent
        await agent.initialize()
        
        # Use the agent
        response = await agent.chat("What's the weather in Paris?")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'agent' in locals():
            await agent.cleanup()


async def example_factory_method():
    """Example 3: Using Agent.create() factory method"""
    print("\n" + "=" * 60)
    print("Example 3: Using Agent.create() factory method")
    print("=" * 60)
    
    try:
        # Create and initialize agent in one step
        agent = await Agent.create(
            server_spec="weather_mcp_server.py",
            tools=["get_current_weather"]
        )
        
        if agent:
            # Use the agent
            response = await agent.chat("What's the weather in Tokyo?")
            print(f"Response: {response}")
        else:
            print("Failed to create agent")
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'agent' in locals() and agent:
            await agent.cleanup()


async def example_async_context_manager():
    """Example 4: Using async context manager"""
    print("\n" + "=" * 60)
    print("Example 4: Using async context manager")
    print("=" * 60)
    
    try:
        # Use agent with async context manager (automatic cleanup)
        async with Agent(server_spec="weather_mcp_server.py") as agent:
            response = await agent.chat("What's the weather in New York?")
            print(f"Response: {response}")
        # Agent is automatically cleaned up here
        
    except Exception as e:
        print(f"Error: {e}")


async def example_tool_filtering():
    """Example 5: Using tool filtering"""
    print("\n" + "=" * 60)
    print("Example 5: Tool filtering")
    print("=" * 60)
    
    try:
        # Create agent with specific tools
        agent = Agent(
            server_spec="weather_mcp_server.py",
            tools=["get_current_weather"]  # Only this tool will be available
        )
        
        await agent.initialize()
        
        response = await agent.chat("What's the weather in Sydney?")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'agent' in locals():
            await agent.cleanup()


async def example_multiple_messages():
    """Example 6: Multiple messages with same agent"""
    print("\n" + "=" * 60)
    print("Example 6: Multiple messages with same agent")
    print("=" * 60)
    
    try:
        async with Agent(server_spec="weather_mcp_server.py") as agent:
            # Multiple interactions with the same agent
            cities = ["Berlin", "Madrid", "Rome"]
            
            for city in cities:
                response = await agent.chat(f"What's the weather in {city}?")
                print(f"{city}: {response[:100]}...")  # Truncate for readability
                
    except Exception as e:
        print(f"Error: {e}")


async def example_single_message_function():
    """Example 7: Using Agent for single messages"""
    print("\n" + "=" * 60)
    print("Example 7: Agent for single messages")
    print("=" * 60)
    
    try:
        # For one-off messages, use the Agent with context manager
        async with Agent(server_spec="weather_mcp_server.py", tools=["get_current_weather"]) as agent:
            response = await agent.chat("What's the weather in Mumbai?")
            print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run all examples"""
    print("Agent Class Usage Examples")
    print("Note: These examples require weather_mcp_server.py to be available")
    print()
    
    await example_constructor_with_server_spec()
    await example_constructor_with_config()
    await example_factory_method()
    await example_async_context_manager()
    await example_tool_filtering()
    await example_multiple_messages()
    await example_single_message_function()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main()) 