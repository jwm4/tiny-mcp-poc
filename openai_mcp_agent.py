# openai_mcp_agent.py
import json
import os
import asyncio
import argparse
from urllib.parse import urlparse
from openai import OpenAI

# Import MCP client libraries
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client  # type: ignore
from mcp.client.streamable_http import streamablehttp_client  # type: ignore
import httpx

# This class was written by Google Gemini but then massively rewritten over and over by
# Cursor using Claude 4 Sonnet after numerous change requests.
# Converted to use OpenAI Responses API instead of Chat Completions API.

# --- 1. Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

DEFAULT_MODEL = "gpt-4o"


# --- 2. Inference Client Class ---
class InferenceClient:
    """Client for making inference requests to OpenAI-compatible APIs."""
    
    def __init__(self, api_key=None, base_url=None, model=None):
        """
        Initialize the inference client.
        
        Args:
            api_key (str, optional): API key for authentication. Defaults to OPENAI_API_KEY env var.
            base_url (str, optional): Base URL for the API. Defaults to OpenAI's URL.
            model (str, optional): Default model to use. Defaults to DEFAULT_MODEL.
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.base_url = base_url
        self.model = model or DEFAULT_MODEL
        
        if not self.api_key:
            raise ValueError("API key must be provided either as parameter or OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client with optional base_url for compatibility with other providers
        client_params = {"api_key": self.api_key}
        if self.base_url:
            client_params["base_url"] = self.base_url
            
        self.client = OpenAI(**client_params)
    
    def create_response(self, input_data, tools=None, previous_response_id=None, instructions=None, model=None):
        """
        Create a response using the Responses API.
        
        Args:
            input_data: Input data for the API call
            tools (list, optional): Tools available to the model
            previous_response_id (str, optional): ID of previous response for conversation continuity
            instructions (str, optional): System instructions for the model
            model (str, optional): Model to use, overrides default
            
        Returns:
            Response object or None on error
        """
        try:
            # Use provided model or fall back to instance default
            model_to_use = model or self.model
            
            # Build the request parameters
            params = {
                "model": model_to_use,
                "input": input_data
            }
            
            if tools:
                params["tools"] = tools
            
            if previous_response_id:
                params["previous_response_id"] = previous_response_id
                
            if instructions:
                params["instructions"] = instructions

            response = self.client.responses.create(**params)
            return response
        except Exception as e:
            print(f"Error calling Responses API: {e}")
            return None
    
    def __repr__(self):
        base_info = f"model={self.model}"
        if self.base_url:
            base_info += f", base_url={self.base_url}"
        return f"InferenceClient({base_info})"


# Create default inference client for backward compatibility
_default_client = None

def get_default_client():
    """Get or create the default inference client."""
    global _default_client
    if _default_client is None:
        _default_client = InferenceClient()
    return _default_client


# --- 3. Server Configuration Classes ---
class MCPServerConfig:
    """Base class for MCP server configurations."""
    pass


class LocalMCPServerConfig(MCPServerConfig):
    """Configuration for local MCP servers."""
    def __init__(self, script_path, command="python", args=None, tools=None):
        self.script_path = script_path
        self.command = command
        self.args = args or []
        self.tools = tools  # List of tool names to use, or None for all tools


class RemoteMCPServerConfig(MCPServerConfig):
    """Configuration for remote MCP servers."""
    def __init__(self, url, transport_type="sse", tools=None):
        self.url = url
        self.transport_type = transport_type  # "sse" or "websocket"
        self.tools = tools  # List of tool names to use, or None for all tools


# --- 3. JSON Processing Functions ---
def convert_messages_to_responses_input(messages):
    """Convert Chat Completions messages format to Responses API input format."""
    input_items = []
    
    for message in messages:
        if message["role"] == "system":
            # System messages become instructions in Responses API
            continue  # We'll handle this separately in the main function
        elif message["role"] == "user":
            input_items.append({
                "role": "user",
                "content": [{"type": "input_text", "text": message["content"]}]
            })
        elif message["role"] == "assistant":
            if message.get("tool_calls"):
                # Handle assistant messages with tool calls
                input_items.append({
                    "role": "assistant", 
                    "content": [{"type": "output_text", "text": message.get("content", "")}]
                })
                # Tool calls will be handled by the Responses API automatically
            else:
                input_items.append({
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": message["content"]}]
                })
        elif message["role"] == "tool":
            # Tool responses in Responses API format
            input_items.append({
                "type": "function_call_output",
                "call_id": message["tool_call_id"],
                "output": message["content"]
            })
    
    return input_items


def process_responses_api_output(response):
    """Process Responses API output and extract relevant information."""
    if not response or not response.output:
        return None, []
    
    # Extract the last assistant message
    assistant_message = None
    tool_calls = []
    
    for output_item in response.output:
        if output_item.type == "message" and output_item.role == "assistant":
            if output_item.content:
                for content_item in output_item.content:
                    if content_item.type == "output_text":
                        assistant_message = content_item.text
        elif output_item.type == "function_call":
            # Parse arguments if they're a JSON string
            try:
                if isinstance(output_item.arguments, str):
                    function_args = json.loads(output_item.arguments)
                else:
                    function_args = output_item.arguments
            except json.JSONDecodeError:
                print(f"Warning: Could not parse tool arguments: {output_item.arguments}")
                function_args = output_item.arguments
            
            tool_calls.append({
                "id": output_item.call_id,
                "function_name": output_item.name,
                "function_args": function_args
            })
    
    return assistant_message, tool_calls


def build_user_input(content):
    """Build a user input for the Responses API."""
    return {
        "role": "user",
        "content": [{"type": "input_text", "text": content}]
    }


def build_tool_response_input(tool_call_id, tool_output):
    """Build a tool response input for the Responses API."""
    return {
        "type": "function_call_output",
        "call_id": tool_call_id,
        "output": str(tool_output)
    }


def process_mcp_tools_response(server_tools_response, tool_filter=None):
    """
    Process MCP server tools response and convert to OpenAI format.
    
    Args:
        server_tools_response: Response from MCP server list_tools call
        tool_filter (list, optional): List of tool names to include. If None, includes all tools.
    
    Returns:
        list: List of tools in OpenAI format
    """
    mcp_tools = []

    for tool in server_tools_response.tools:
        # Apply tool filtering if specified
        if tool_filter is not None and tool.name not in tool_filter:
            continue
            
        # Note: Adjusted for actual MCP Tool structure
        mcp_tools.append(
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema,  # Adjusted attribute name
            }
        )

    return mcp_tools


def extract_mcp_tool_result(tool_output_obj):
    """Extract the actual result from MCP tool output."""
    # Handle different possible result structures
    if hasattr(tool_output_obj, "content"):
        if (
            isinstance(tool_output_obj.content, list)
            and len(tool_output_obj.content) > 0
        ):
            return tool_output_obj.content[0].text
        return str(tool_output_obj.content)
    return str(tool_output_obj)


# --- 4. Server Configuration Parsing ---
def parse_server_config(server_spec, tools_override=None):
    """
    Parse server specification string into MCPServerConfig object.
    
    Args:
        server_spec (str): Server specification in format:
            - For local: "local:path/to/server.py" or "local:path/to/server.py:python"
            - For remote: "remote:http://localhost:8000" or "remote:ws://localhost:8000"
            - Simple path: "path/to/server.py" (defaults to local)
            - Tool filtering: "server.py[tool1,tool2]" or "local:server.py:python[tool1,tool2]"
        tools_override (str, optional): Comma-separated list of tools to override any embedded tools
    
    Returns:
        MCPServerConfig: Appropriate server configuration object
    """
    if not server_spec:
        raise ValueError("Server specification cannot be empty")
    
    # Extract tool list if present in server spec
    tools = None
    if '[' in server_spec and server_spec.endswith(']'):
        tool_start = server_spec.rfind('[')
        tool_spec = server_spec[tool_start+1:-1]
        server_spec = server_spec[:tool_start]
        tools = [tool.strip() for tool in tool_spec.split(',') if tool.strip()]
    
    # Parse tools override if provided
    if tools_override:
        tools = [tool.strip() for tool in tools_override.split(',') if tool.strip()]
        print(f"Overriding tools with command line specification: {tools}")
    
    # Handle simple path (default to local)
    if ":" not in server_spec or (server_spec.count(":") == 1 and not server_spec.startswith(("local:", "remote:"))):
        return LocalMCPServerConfig(server_spec, tools=tools)
    
    # Parse prefixed specifications
    if server_spec.startswith("local:"):
        spec_parts = server_spec[6:].split(":", 1)  # Remove "local:" prefix
        script_path = spec_parts[0]
        command = spec_parts[1] if len(spec_parts) > 1 else "python"
        return LocalMCPServerConfig(script_path, command, tools=tools)
    
    elif server_spec.startswith("remote:"):
        url = server_spec[7:]  # Remove "remote:" prefix
        parsed_url = urlparse(url)
        
        if parsed_url.scheme in ["ws", "wss"]:
            transport_type = "websocket"
        elif parsed_url.scheme in ["http", "https"]:
            transport_type = "sse"
        else:
            raise ValueError(f"Unsupported remote server scheme: {parsed_url.scheme}")
        
        return RemoteMCPServerConfig(url, transport_type, tools=tools)
    
    else:
        raise ValueError(f"Invalid server specification format: {server_spec}")


# --- 5. MCP Client Setup ---
async def setup_mcp_client(server_config=None):
    """
    Sets up an MCP client session to connect to a server.
    
    Args:
        server_config (MCPServerConfig): Server configuration object
    
    Returns:
        tuple: (session, tools, exit_stack) or (None, [], None) on error
    """
    if server_config is None:
        server_config = LocalMCPServerConfig("weather_mcp_server.py")
    
    exit_stack = AsyncExitStack()
    try:
        if isinstance(server_config, LocalMCPServerConfig):
            # Handle local MCP server
            server_params = StdioServerParameters(
                command=server_config.command,
                args=[server_config.script_path] + server_config.args
            )

            read_stream, write_stream = await exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            
            print(f"Connecting to local MCP server: {server_config.script_path}")

        elif isinstance(server_config, RemoteMCPServerConfig):
            # Handle remote MCP server
            print(f"Connecting to remote MCP server: {server_config.url} (transport: {server_config.transport_type})")
            
            if server_config.transport_type == "sse":
                # Use SSE client to connect to remote server
                sse_url = server_config.url
                if not sse_url.endswith('/sse'):
                    sse_url = sse_url.rstrip('/') + '/sse'
                
                read_stream, write_stream = await exit_stack.enter_async_context(
                    sse_client(sse_url)
                )
                print("Connected to remote MCP server via SSE")
                
            elif server_config.transport_type == "websocket":
                # WebSocket support is not available in the MCP library
                print("WebSocket transport is not supported by the MCP library")
                print("Please use HTTP or SSE transport instead")
                await exit_stack.aclose()
                return None, [], None
                
            else:  # HTTP transport
                # Use streamable HTTP client
                async_gen = streamablehttp_client(server_config.url)
                read_stream, write_stream, get_session_id = await exit_stack.enter_async_context(
                    async_gen
                )
                print("Connected to remote MCP server via HTTP")
        
        else:
            raise ValueError(f"Unsupported server configuration type: {type(server_config)}")

        # Create an MCP session using the streams
        session = await exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )

        # Initialize the MCP session before making any requests
        await session.initialize()

        print("Discovering tools from MCP server...")
        server_tools_response = await session.list_tools()
        
        # Apply tool filtering if specified in server config
        tool_filter = getattr(server_config, 'tools', None)
        mcp_tools = process_mcp_tools_response(server_tools_response, tool_filter)
        
        # Log available vs selected tools
        all_tool_names = [tool.name for tool in server_tools_response.tools]
        selected_tool_names = [t['name'] for t in mcp_tools]
        
        print(f"Available tools from server: {all_tool_names}")
        if tool_filter:
            print(f"Using filtered tools: {selected_tool_names}")
            # Check for invalid tool names
            invalid_tools = [t for t in tool_filter if t not in all_tool_names]
            if invalid_tools:
                print(f"Warning: Requested tools not found on server: {invalid_tools}")
        else:
            print(f"Using all available tools: {selected_tool_names}")

        return session, mcp_tools, exit_stack
    except Exception as e:
        print(f"Error setting up MCP client: {e}")
        await exit_stack.aclose()
        return None, [], None


# --- 6. Deprecated - Removed call_responses_api function
# This functionality is now handled by the InferenceClient class


# --- 7. Tool Execution Functions ---
async def execute_mcp_tool(mcp_session, function_name, function_args):
    """Execute an MCP tool and return the result."""
    try:
        print(f"Calling MCP tool: {function_name} with args: {function_args}")
        tool_output_obj = await mcp_session.call_tool(
            name=function_name, arguments=function_args
        )
        tool_output = extract_mcp_tool_result(tool_output_obj)
        print(f"MCP tool output: {tool_output}")
        return tool_output
    except Exception as e:
        print(f"Error executing MCP tool '{function_name}': {e}")
        raise


async def process_tool_calls_responses_api(tool_calls, mcp_session, tools_for_llm, previous_response_id, inference_client):
    """Process tool calls for Responses API and return tool outputs."""
    tool_outputs = []
    
    for tool_call_info in tool_calls:
        function_name = tool_call_info["function_name"]
        function_args = tool_call_info["function_args"]
        tool_call_id = tool_call_info["id"]

        # Check if this is an MCP tool
        if function_name in [t["name"] for t in tools_for_llm]:
            try:
                tool_output = await execute_mcp_tool(
                    mcp_session, function_name, function_args
                )
                tool_response = build_tool_response_input(tool_call_id, tool_output)
                tool_outputs.append(tool_response)
            except Exception as e:
                error_output = build_tool_response_input(
                    tool_call_id, f"Error executing tool '{function_name}': {e}"
                )
                tool_outputs.append(error_output)
        else:
            print(f"Agent: Error - Unknown tool requested by LLM: {function_name}")
            error_output = build_tool_response_input(tool_call_id, "Unknown tool")
            tool_outputs.append(error_output)
    
    # Make a follow-up call to the Responses API with tool outputs
    if tool_outputs:
        response = inference_client.create_response(
            input_data=tool_outputs,
            tools=tools_for_llm,
            previous_response_id=previous_response_id
        )
        if response:
            assistant_message, _ = process_responses_api_output(response)
            return assistant_message, response.id
    
    return "No response after tool execution.", None


# --- 8. Agent Class ---
class Agent:
    """OpenAI MCP Agent - AI assistant with configurable MCP server support using Responses API."""

    def __init__(self, server_config=None, server_spec=None, tools=None, inference_client=None):
        """
        Initialize the Agent with server configuration.

        Args:
            server_config (MCPServerConfig): Server configuration object
            server_spec (str): Server specification string (alternative to server_config)
            tools (list): List of tool names to use (overrides tools in config)
            inference_client (InferenceClient, optional): Inference client to use for API calls
        """
        # Parse server configuration
        if server_config is None and server_spec is not None:
            server_config = parse_server_config(server_spec)
        
        if server_config is None:
            server_config = LocalMCPServerConfig("weather_mcp_server.py")
        
        # Override tools if specified
        if tools is not None:
            server_config.tools = tools
        
        self.server_config = server_config
        self.inference_client = inference_client or get_default_client()
        self.mcp_session = None
        self.tools_for_llm = []
        self.exit_stack = None
        self.last_response_id = None
        self.instructions = "You are a helpful AI assistant. You can answer general questions and use available tools to help users."
        self.is_initialized = False

    async def initialize(self):
        """Initialize the agent with MCP client and tools."""
        if self.is_initialized:
            return True

        self.mcp_session, self.tools_for_llm, self.exit_stack = await setup_mcp_client(self.server_config)
        if not self.mcp_session:
            return False

        self.is_initialized = True
        return True

    async def process_message(self, user_input):
        """
        Process a single user message and return the agent's response.

        Args:
            user_input (str): The user's message

        Returns:
            str: The agent's response
        """
        if not self.is_initialized:
            raise RuntimeError(
                "Agent not initialized. Call initialize() first."
            )

        # Build user input for Responses API
        input_data = build_user_input(user_input)

        # Get response from Responses API
        response = self.inference_client.create_response(
            input_data=[input_data],
            tools=self.tools_for_llm,
            previous_response_id=self.last_response_id,
            instructions=self.instructions
        )

        if not response:
            return "Could not get a response from OpenAI Responses API."

        # Update last response ID for conversation continuity
        self.last_response_id = response.id

        # Process the response
        assistant_message, tool_calls = process_responses_api_output(response)

        if tool_calls:
            # Process tool calls and get final response
            final_response, new_response_id = await process_tool_calls_responses_api(
                tool_calls, self.mcp_session, self.tools_for_llm, self.last_response_id, self.inference_client
            )
            if new_response_id:
                self.last_response_id = new_response_id
            return final_response
        else:
            return assistant_message or "No content in response"

    async def chat(self, user_input):
        """
        Alias for process_message for more intuitive API.
        
        Args:
            user_input (str): The user's message

        Returns:
            str: The agent's response
        """
        return await self.process_message(user_input)

    async def cleanup(self):
        """Clean up the agent resources."""
        if self.exit_stack:
            await self.exit_stack.aclose()
        self.is_initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    @classmethod
    async def create(cls, server_config=None, server_spec=None, tools=None, inference_client=None):
        """
        Factory method to create and initialize an Agent.

        Args:
            server_config (MCPServerConfig): Server configuration object
            server_spec (str): Server specification string (alternative to server_config)
            tools (list): List of tool names to use (overrides tools in config)
            inference_client (InferenceClient, optional): Inference client to use for API calls

        Returns:
            Agent: Initialized agent, or None if initialization failed
        """
        agent = cls(server_config, server_spec, tools, inference_client)
        success = await agent.initialize()
        return agent if success else None


# --- 9. Interactive Console Interface ---
async def run_interactive_agent(server_config=None):
    """Run the agent in interactive console mode."""
    agent = await Agent.create(server_config)
    if not agent:
        print("Failed to start MCP client. Exiting.")
        return

    print(f"\nAgent ({agent.inference_client.model}) ready. Type 'quit' or 'exit' to end.")

    try:
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting agent.")
                break

            try:
                response = await agent.process_message(user_input)
                print(f"Agent: {response}")
            except Exception as e:
                print(f"Agent: Error processing message: {e}")

    finally:
        await agent.cleanup()


# --- 10. Command Line Interface ---
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="OpenAI MCP Agent - AI assistant with configurable MCP server support using Responses API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Server specification formats:
  Local servers:
    - path/to/server.py                    (default: python)
    - local:path/to/server.py              (default: python)
    - local:path/to/server.py:node         (custom command)
  
  Tool filtering:
    - server.py[tool1,tool2]               (use specific tools)
    - local:server.py:python[get_weather]  (combined format)
  
  Remote servers:
    - remote:http://localhost:8000         (HTTP transport)
    - remote:https://example.com/mcp       (HTTPS transport)
    - remote:http://localhost:8000/sse     (SSE transport)
    - remote:ws://localhost:8000           (WebSocket - not supported yet)

Examples:
  python openai_mcp_agent.py --server weather_mcp_server.py
  python openai_mcp_agent.py --server 'weather_mcp_server.py[get_current_weather]'
  python openai_mcp_agent.py --server local:my_server.py:node
  python openai_mcp_agent.py --server weather_mcp_server.py --tools get_current_weather
  python openai_mcp_agent.py --server remote:http://localhost:8000
        """
    )
    
    parser.add_argument(
        "--server", "-s",
        type=str,
        required=True,
        help="MCP server specification (required)"
    )
    
    parser.add_argument(
        "--tools", "-t",
        type=str,
        help="Comma-separated list of tools to use (overrides tools in server spec)"
    )
    
    return parser.parse_args()


# --- 11. Main Entry Point ---
async def main():
    """Main entry point - runs interactive agent with configurable server."""
    args = parse_arguments()
    
    try:
        server_config = parse_server_config(args.server, args.tools)
        await run_interactive_agent(server_config)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nGoodbye!")
        return 0


# --- Run the agent ---
if __name__ == "__main__":
    exit_code = asyncio.run(main())
    if exit_code:
        exit(exit_code)
