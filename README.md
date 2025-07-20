# OpenAI MCP Agent

A flexible AI assistant that connects OpenAI's GPT models with MCP (Model Context Protocol) servers, enabling dynamic tool integration and configurable server connections.  This README was written by Cursor using Claude 4 Sonnet.

## Features

- **Configurable MCP Servers**: Support for local and remote MCP servers
- **Multiple Transport Types**: Local (stdio), Remote HTTP, and Remote SSE
- **Tool Filtering**: Select specific tools from MCP servers  
- **Multi-Provider Inference**: Support for OpenAI, Azure OpenAI, and other OpenAI-compatible providers
- **Flexible API**: Multiple ways to create and use agents
- **Async Context Management**: Automatic resource cleanup
- **Command Line Interface**: Easy-to-use CLI with comprehensive options

## Installation

```bash
pip install openai mcp
export OPENAI_API_KEY="your-openai-api-key"
```

## Inference Client Configuration

The agent supports multiple inference providers through the `InferenceClient` class. This allows you to use OpenAI, Azure OpenAI, or any OpenAI-compatible API.

### Default OpenAI Configuration

```python
from openai_mcp_agent import InferenceClient

# Uses OPENAI_API_KEY environment variable
client = InferenceClient()

# Custom model
client = InferenceClient(model="gpt-4o-mini")
```

### Azure OpenAI Configuration

```python
client = InferenceClient(
    api_key="your-azure-api-key",
    base_url="https://your-resource.openai.azure.com/openai/v1",
    model="gpt-4o"
)
```

### Other OpenAI-Compatible Providers

```python
# Local inference server (e.g., Ollama)
client = InferenceClient(
    api_key="not-needed",
    base_url="http://localhost:11434/v1",
    model="llama2"
)

# Other providers like Anthropic via proxy, etc.
client = InferenceClient(
    api_key="your-api-key",
    base_url="https://api.provider.com/v1",
    model="custom-model"
)
```

## Quick Start

### Using the Agent Class

```python
import asyncio
from openai_mcp_agent import Agent, InferenceClient

async def main():
    # Create and use an agent with default OpenAI
    async with Agent(server_spec="weather_mcp_server.py") as agent:
        response = await agent.chat("What's the weather in London?")
        print(response)

asyncio.run(main())
```

### Using Custom Inference Client

```python
import asyncio
from openai_mcp_agent import Agent, InferenceClient

async def main():
    # Create custom inference client
    client = InferenceClient(model="gpt-4o-mini")
    
    # Create agent with custom client
    async with Agent(server_spec="weather_mcp_server.py", inference_client=client) as agent:
        response = await agent.chat("What's the weather in London?")
        print(response)

asyncio.run(main())
```

### Command Line Usage

```bash
# Basic usage
python openai_mcp_agent.py --server weather_mcp_server.py

# With tool filtering
python openai_mcp_agent.py --server weather_mcp_server.py --tools get_current_weather

# Using server specification with embedded tools
python openai_mcp_agent.py --server 'weather_mcp_server.py[get_current_weather]'
```

## Agent Class API

### Constructor Options

```python
# Option 1: Server specification string
agent = Agent(server_spec="weather_mcp_server.py[get_current_weather]")

# Option 2: Server configuration object
from openai_mcp_agent import LocalMCPServerConfig
config = LocalMCPServerConfig("weather_mcp_server.py", tools=["get_current_weather"])
agent = Agent(server_config=config)

# Option 3: Direct parameters
agent = Agent(
    server_spec="weather_mcp_server.py",
    tools=["get_current_weather"]
)

# Option 4: With custom inference client
from openai_mcp_agent import InferenceClient
client = InferenceClient(model="gpt-4o-mini")
agent = Agent(
    server_spec="weather_mcp_server.py",
    inference_client=client
)

# Option 5: All parameters
agent = Agent(
    server_spec="weather_mcp_server.py",
    tools=["get_current_weather"],
    inference_client=InferenceClient(model="gpt-4o-mini")
)
```

### Usage Patterns

#### 1. Factory Method (Recommended)
```python
# Create and initialize in one step
agent = await Agent.create(server_spec="weather_mcp_server.py")
if agent:
    response = await agent.chat("Hello!")
    await agent.cleanup()

# With custom inference client
from openai_mcp_agent import InferenceClient
client = InferenceClient(model="gpt-4o-mini")
agent = await Agent.create(server_spec="weather_mcp_server.py", inference_client=client)
if agent:
    response = await agent.chat("Hello!")
    await agent.cleanup()
```

#### 2. Async Context Manager (Recommended)
```python
# Automatic cleanup
async with Agent(server_spec="weather_mcp_server.py") as agent:
    response = await agent.chat("Hello!")
# Agent is automatically cleaned up here
```

#### 3. Manual Management
```python
# Manual initialization and cleanup
agent = Agent(server_spec="weather_mcp_server.py")
await agent.initialize()
response = await agent.chat("Hello!")
await agent.cleanup()
```

#### 4. One-off Messages
```python
# For single messages
async with Agent(server_spec="weather_mcp_server.py") as agent:
    response = await agent.chat("What's the weather?")
```

## Server Configuration

### Local Servers

```python
# Simple path (uses python by default)
Agent(server_spec="weather_mcp_server.py")

# Explicit local with custom command
Agent(server_spec="local:my_server.js:node")

# With tool filtering
Agent(server_spec="weather_mcp_server.py[get_current_weather,get_forecast]")
```

### Server Configuration Objects

```python
from openai_mcp_agent import LocalMCPServerConfig, RemoteMCPServerConfig

# Local server configurations
config = LocalMCPServerConfig("weather_mcp_server.py")

# With custom command
config = LocalMCPServerConfig("my_server.js", command="node")

# With tool filtering
config = LocalMCPServerConfig("weather_mcp_server.py", tools=["get_current_weather"])

# With additional arguments
config = LocalMCPServerConfig("server.py", args=["--debug", "--port", "8080"])

# Remote server configurations
config = RemoteMCPServerConfig("http://localhost:8000", transport_type="sse")
config = RemoteMCPServerConfig("https://api.example.com/mcp", transport_type="sse")
config = RemoteMCPServerConfig("http://localhost:3000", transport_type="sse")

# Remote server with tool filtering
config = RemoteMCPServerConfig("http://localhost:8000", tools=["get_weather"])
```

### Tool Filtering

Tools can be filtered in several ways:

```python
# 1. In server specification
Agent(server_spec="weather_mcp_server.py[get_current_weather,get_forecast]")

# 2. As separate parameter
Agent(server_spec="weather_mcp_server.py", tools=["get_current_weather"])

# 3. In server config object
config = LocalMCPServerConfig("weather_mcp_server.py", tools=["get_current_weather"])
Agent(server_config=config)
```

## Command Line Interface

### Basic Options

```bash
# Required server argument
python openai_mcp_agent.py --server weather_mcp_server.py

# With tool filtering
python openai_mcp_agent.py --server weather_mcp_server.py --tools get_current_weather

# Help
python openai_mcp_agent.py --help
```

### Server Specification Formats

```bash
# Local servers
--server weather_mcp_server.py                    # Default: python
--server local:weather_mcp_server.py              # Explicit local
--server local:my_server.js:node                  # Custom command

# Tool filtering
--server 'weather_mcp_server.py[get_current_weather]'
--server weather_mcp_server.py --tools get_current_weather,get_forecast

# Remote servers
--server remote:http://localhost:8000             # HTTP transport
--server remote:https://api.example.com/mcp       # HTTPS transport
--server remote:http://localhost:8000/sse         # SSE transport
--server remote:ws://localhost:8000               # WebSocket (not supported yet)
```

## Advanced Usage

### Multiple Conversations

```python
async with Agent(server_spec="weather_mcp_server.py") as agent:
    # Multiple interactions with conversation history
    response1 = await agent.chat("What's the weather in London?")
    response2 = await agent.chat("What about Paris?")
    response3 = await agent.chat("Compare them")
```

### Error Handling

```python
try:
    async with Agent(server_spec="weather_mcp_server.py") as agent:
        response = await agent.chat("What's the weather?")
except Exception as e:
    print(f"Error: {e}")
```

### Custom System Messages

```python
# The Agent class uses a default system message, but you can modify it
agent = Agent(server_spec="weather_mcp_server.py")
await agent.initialize()

# Modify the system message
agent.messages[0] = {
    "role": "system",
    "content": "You are a weather expert assistant."
}
```

## API Reference

### Agent Class

#### Constructor
```python
Agent(server_config=None, server_spec=None, tools=None, inference_client=None)
```

**Parameters:**
- `server_config`: MCPServerConfig object
- `server_spec`: Server specification string
- `tools`: List of tool names to use
- `inference_client`: InferenceClient instance (uses default OpenAI if None)

#### Methods
- `async initialize()`: Initialize the agent
- `async chat(message)`: Send a message and get response (alias for process_message)
- `async process_message(message)`: Process a message and return response
- `async cleanup()`: Clean up resources

#### Class Methods
- `async Agent.create(...)`: Factory method to create and initialize

#### Context Manager
- `async with Agent(...) as agent:`: Automatic initialization and cleanup

### InferenceClient Class

```python
InferenceClient(api_key=None, base_url=None, model=None)
```

**Parameters:**
- `api_key`: API key for authentication (defaults to OPENAI_API_KEY env var)
- `base_url`: Base URL for the API (defaults to OpenAI's URL)
- `model`: Model to use (defaults to "gpt-4o")

**Methods:**
- `create_response(input_data, tools=None, previous_response_id=None, instructions=None, model=None)`: Create response using Responses API

### Configuration Classes

#### LocalMCPServerConfig
```python
LocalMCPServerConfig(script_path, command="python", args=None, tools=None)
```

#### RemoteMCPServerConfig
```python
RemoteMCPServerConfig(url, transport_type="sse", tools=None)
```

**Transport Types:**
- `"sse"`: Server-Sent Events (default)
- `"websocket"`: WebSocket (not supported yet)
- Transport is auto-detected from URL scheme if not specified

### Utility Functions

```python
# Parse server specification string
parse_server_config(server_spec, tools_override=None)
```

## Examples

- **`example_agent_usage.py`**: Comprehensive usage examples covering all patterns, including tool filtering, different initialization methods, and various usage scenarios.
- **`example_remote_server.py`**: Complete guide to using remote MCP servers with different transport types, configuration options, and real-world examples.

## Requirements

- Python 3.7+
- OpenAI API key
- MCP server implementation (local or remote)
- Required packages: `openai`, `mcp`, `requests`, `httpx`

## Testing with Remote Servers

To test with real remote MCP servers, you have several options:

### Option 1: Using mcp-streamablehttp-proxy

```bash
# Install the proxy
pip install mcp-streamablehttp-proxy

# Proxy your local MCP server to HTTP
mcp-streamablehttp-proxy python -m weather_mcp_server

# Connect to the proxied server
python openai_mcp_agent.py --server remote:http://localhost:3000
```

### Option 2: Direct Remote Server

If you have an MCP server that natively supports HTTP/SSE:

```bash
# Connect to remote server
python openai_mcp_agent.py --server remote:https://api.example.com/mcp

# Connect with specific transport
python openai_mcp_agent.py --server remote:http://localhost:8000/sse
```

### Option 3: Custom Server Setup

See `example_remote_server.py` for comprehensive examples of remote server configurations.

 