#!/usr/bin/env python3
"""
Example script demonstrating how to use the NPS MCP Server.

This script shows how to:
1. Search for national parks
2. Get alerts for specific parks
3. Find campgrounds and visitor centers
4. Get upcoming events

Requirements:
- Set NPS_API_KEY environment variable (optional, will use DEMO_KEY if not set)
- Install dependencies: pip install fastmcp httpx

Usage:
    python example_nps_usage.py
"""

import asyncio
import json
from openai_mcp_agent import Agent

async def demonstrate_nps_tools():
    """Demonstrate the NPS MCP server tools."""
    
    print("🏞️  NPS MCP Server Demo")
    print("=" * 50)
    
    # Initialize the agent with the NPS MCP server
    # The server will run as a subprocess using stdio transport
    agent = Agent(server_specs="nps_mcp_server.py")
    
    try:
        # Initialize the agent
        await agent.initialize()
        
        print("\n1. 🔍 Searching for parks in California...")
        response = await agent.chat("Search for 3 national parks in California using the search_parks tool")
        print("California Parks:")
        print(response)
        
        print("\n2. 🚨 Getting alerts for Yellowstone...")
        response = await agent.chat("Get current alerts for Yellowstone National Park (park code: yell)")
        print("Yellowstone Alerts:")
        print(response)
        
        print("\n3. 🏕️ Finding campgrounds in Grand Canyon...")
        response = await agent.chat("Find campgrounds in Grand Canyon National Park (park code: grca)")
        print("Grand Canyon Campgrounds:")
        print(response)
        
        print("\n4. 🏛️ Finding visitor centers in Acadia...")
        response = await agent.chat("Find visitor centers in Acadia National Park (park code: acad)")
        print("Acadia Visitor Centers:")
        print(response)
        
        print("\n5. 📅 Getting events for Yosemite...")
        response = await agent.chat("Get upcoming events for Yosemite National Park (park code: yose)")
        print("Yosemite Events:")
        print(response)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTips:")
        print("- Make sure you have an internet connection")
        print("- Set NPS_API_KEY environment variable for full functionality")
        print("- Check that fastmcp and httpx are installed")
    
    finally:
        if 'agent' in locals():
            await agent.cleanup()

def main():
    """Main function to run the demo."""
    print("Starting NPS MCP Server Demo...")
    print("This will demonstrate various National Park Service API calls.")
    print("\nNote: Using DEMO_KEY if NPS_API_KEY not set (limited functionality)")
    print("Get a free API key at: https://www.nps.gov/subjects/developer/get-started.htm")
    
    # Run the async demo
    asyncio.run(demonstrate_nps_tools())

if __name__ == "__main__":
    main() 