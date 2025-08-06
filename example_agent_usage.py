#!/usr/bin/env python3
"""
Comprehensive example demonstrating various ways to use the OpenAI MCP Agent.

This script shows:
1. Basic agent usage with tool filtering
2. Custom server configuration
3. Different initialization patterns
4. Error handling
5. Conversation management
6. Remote server usage examples

Requirements:
- Set OPENAI_API_KEY environment variable
- Have weather_mcp_server.py available
- Install dependencies: pip install openai mcp

Usage:
    python example_agent_usage.py
"""

import asyncio
import os
from openai_mcp_agent import Agent, LocalMCPServerConfig, InferenceClient

async def basic_usage_example():
    """Demonstrate basic agent usage with tool filtering."""
    
    print("🤖 Basic Agent Usage Example")
    print("=" * 50)
    
    # Create agent with specific tools only
    agent = Agent(server_specs="weather_mcp_server.py[get_current_weather]")
    
    try:
        await agent.initialize()
        
        print("\nAsking about weather in San Francisco...")
        response = await agent.chat("What's the weather like in San Francisco?")
        print(f"Agent: {response}")
        
        print("\nAsking a general question...")
        response = await agent.chat("What's the capital of France?")
        print(f"Agent: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        await agent.cleanup()

async def custom_config_example():
    """Demonstrate custom server configuration."""
    
    print("\n🔧 Custom Configuration Example")
    print("=" * 50)
    
    # Create custom server configuration
    server_config = LocalMCPServerConfig(
        script_path="weather_mcp_server.py",
        command="python",
        tools=["get_current_weather"]  # Only allow weather tool
    )
    
    agent = Agent(server_configs=server_config)
    
    try:
        await agent.initialize()
        
        print(f"Available tools: {agent.all_tool_names}")
        
        response = await agent.chat("Check the weather in New York")
        print(f"Agent: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        await agent.cleanup()

async def factory_method_example():
    """Demonstrate using the factory method."""
    
    print("\n🏭 Factory Method Example")
    print("=" * 50)
    
    # Use factory method with custom inference client
    inference_client = InferenceClient(model="gpt-4o-mini")
    
    agent = await Agent.create(
        server_specs="weather_mcp_server.py",
        inference_client=inference_client
    )
    
    if not agent:
        print("❌ Failed to create agent")
        return
    
    try:
        print(f"Using model: {agent.inference_client.model}")
        response = await agent.chat("What's the weather in London?")
        print(f"Agent: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        await agent.cleanup()

async def context_manager_example():
    """Demonstrate using agent as context manager."""
    
    print("\n🔄 Context Manager Example")
    print("=" * 50)
    
    try:
        # Agent automatically initializes and cleans up
        async with Agent(server_specs="weather_mcp_server.py") as agent:
            print("Agent initialized automatically")
            
            # Multiple interactions in same conversation
            response1 = await agent.chat("What's the weather in Tokyo?")
            print(f"First query: {response1}")
            
            response2 = await agent.chat("How about in Paris?")
            print(f"Second query: {response2}")
            
        print("Agent cleaned up automatically")
        
    except Exception as e:
        print(f"❌ Error: {e}")

async def tool_filtering_example():
    """Demonstrate different tool filtering approaches."""
    
    print("\n🔧 Tool Filtering Example")
    print("=" * 50)
    
    try:
        # Method 1: Filter in server spec
        async with Agent(server_specs="weather_mcp_server.py", tools=["get_current_weather"]) as agent:
            print(f"Filtered tools: {agent.all_tool_names}")
            response = await agent.chat("Check weather in Miami")
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

async def error_handling_example():
    """Demonstrate error handling patterns."""
    
    print("\n⚠️ Error Handling Example")
    print("=" * 50)
    
    # Try to connect to non-existent server
    try:
        agent = Agent(server_specs="nonexistent_server.py")
        success = await agent.initialize()
        
        if not success:
            print("❌ Failed to initialize agent with non-existent server")
        else:
            await agent.cleanup()
            
    except Exception as e:
        print(f"❌ Expected error: {e}")
    
    # Try with invalid server spec
    try:
        agent = Agent(server_specs="invalid:spec:format")
        await agent.initialize()
        
    except Exception as e:
        print(f"❌ Expected error with invalid spec: {e}")

async def conversation_example():
    """Demonstrate multi-turn conversation."""
    
    print("\n💬 Conversation Example")
    print("=" * 50)
    
    try:
        async with Agent(server_specs="weather_mcp_server.py") as agent:
            
            # Simulate a conversation
            queries = [
                "What's the weather in Seattle?",
                "How about the temperature?",
                "Is it good weather for hiking?",
            ]
            
            for i, query in enumerate(queries, 1):
                print(f"\nTurn {i}: {query}")
                response = await agent.chat(query)
                print(f"Agent: {response}")
                
    except Exception as e:
        print(f"❌ Error: {e}")

def check_requirements():
    """Check if required environment variables and dependencies are available."""
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    try:
        import openai
        import mcp
        print("✅ Required dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install with: pip install openai mcp")
        return False

async def main():
    """Run all examples."""
    
    print("🚀 OpenAI MCP Agent - Comprehensive Examples")
    print("=" * 60)
    
    if not check_requirements():
        print("\n❌ Requirements not met. Please fix the above issues.")
        return
    
    # Run all examples
    examples = [
        basic_usage_example,
        custom_config_example, 
        factory_method_example,
        context_manager_example,
        tool_filtering_example,
        error_handling_example,
        conversation_example,
    ]
    
    for example in examples:
        try:
            await example()
            print("\n" + "─" * 60)
            
        except Exception as e:
            print(f"❌ Example failed: {e}")
            print("─" * 60)
    
    print("\n✅ All examples completed!")

if __name__ == "__main__":
    asyncio.run(main()) 