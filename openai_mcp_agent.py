# openai_mcp_agent.py
import json
import os
import asyncio
import argparse
import logging
from urllib.parse import urlparse
from openai import OpenAI
from typing import NamedTuple, List, Optional, Dict, Any

# Import MCP client libraries
from mcp import StdioServerParameters
from mcp.client.session import ClientSession
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client  # type: ignore
from mcp.client.streamable_http import streamablehttp_client  # type: ignore
import httpx

# Configure logging
logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO):
    """Configure logging for the application."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger.setLevel(level)

# Initialize logging with default level
setup_logging()

# This class was written by Google Gemini but then massively rewritten over and over by
# Cursor using Claude 4 Sonnet after numerous change requests.
# Converted to use OpenAI Responses API instead of Chat Completions API.

# --- 1. Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

DEFAULT_MODEL = "gpt-4o"


# --- 2. Server Information Structure ---
class ServerInfo(NamedTuple):
    """Information about a connected MCP server."""
    config: 'MCPServerConfig'
    session: ClientSession
    tools: List[Dict[str, Any]]
    tool_names: List[str]  # Quick lookup for tool names


# --- 3. Inference Client Class ---
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
        if self.base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
    
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
            logger.error(f"Error calling Responses API: {e}")
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
                logger.warning(f"Warning: Could not parse tool arguments: {output_item.arguments}")
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
        logger.info(f"Overriding tools with command line specification: {tools}")
    
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
async def setup_mcp_clients(server_configs=None):
    """
    Sets up MCP client sessions to connect to multiple servers.
    
    Args:
        server_configs (List[MCPServerConfig]): List of server configuration objects
    
    Returns:
        tuple: (server_infos, exit_stack) or ([], None) on error
        where server_infos is a List[ServerInfo]
    """
    if server_configs is None:
        server_configs = [LocalMCPServerConfig("weather_mcp_server.py")]
    
    if not isinstance(server_configs, list):
        server_configs = [server_configs]
    
    exit_stack = AsyncExitStack()
    server_infos = []
    
    try:
        for server_config in server_configs:
            try:
                if isinstance(server_config, LocalMCPServerConfig):
                    # Handle local MCP server
                    server_params = StdioServerParameters(
                        command=server_config.command,
                        args=[server_config.script_path] + server_config.args,
                        env=os.environ.copy()  # Pass current environment to subprocess
                    )

                    read_stream, write_stream = await exit_stack.enter_async_context(
                        stdio_client(server_params)
                    )
                    
                    logger.info(f"Connecting to local MCP server: {server_config.script_path}")

                elif isinstance(server_config, RemoteMCPServerConfig):
                    # Handle remote MCP server
                    logger.info(f"Connecting to remote MCP server: {server_config.url} (transport: {server_config.transport_type})")
                    
                    if server_config.transport_type == "sse":
                        # Use SSE client to connect to remote server
                        sse_url = server_config.url
                        if not sse_url.endswith('/sse'):
                            sse_url = sse_url.rstrip('/') + '/sse'
                        
                        read_stream, write_stream = await exit_stack.enter_async_context(
                            sse_client(sse_url)
                        )
                        logger.info("Connected to remote MCP server via SSE")
                        
                    elif server_config.transport_type == "websocket":
                        # WebSocket support is not available in the MCP library
                        logger.warning("WebSocket transport is not supported by the MCP library")
                        logger.warning("Please use HTTP or SSE transport instead")
                        continue  # Skip this server instead of failing all
                        
                    else:  # HTTP transport
                        # Use streamable HTTP client
                        async_gen = streamablehttp_client(server_config.url)
                        read_stream, write_stream, get_session_id = await exit_stack.enter_async_context(
                            async_gen
                        )
                        logger.info("Connected to remote MCP server via HTTP")
                
                else:
                    logger.warning(f"Unsupported server configuration type: {type(server_config)}")
                    continue

                # Create an MCP session using the streams
                session = await exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )

                # Initialize the MCP session before making any requests
                await session.initialize()

                logger.info(f"Discovering tools from MCP server ({server_config})...")
                server_tools_response = await session.list_tools()
                
                # Apply tool filtering if specified in server config
                tool_filter = getattr(server_config, 'tools', None)
                mcp_tools = process_mcp_tools_response(server_tools_response, tool_filter)
                
                # Log available vs selected tools
                all_tool_names = [tool.name for tool in server_tools_response.tools]
                selected_tool_names = [t['name'] for t in mcp_tools]
                
                logger.info(f"Available tools from server: {all_tool_names}")
                if tool_filter:
                    logger.info(f"Using filtered tools: {selected_tool_names}")
                    # Check for invalid tool names
                    invalid_tools = [t for t in tool_filter if t not in all_tool_names]
                    if invalid_tools:
                        logger.warning(f"Warning: Requested tools not found on server: {invalid_tools}")
                else:
                    logger.info(f"Using all available tools: {selected_tool_names}")

                # Create ServerInfo for this server
                server_info = ServerInfo(
                    config=server_config,
                    session=session,
                    tools=mcp_tools,
                    tool_names=selected_tool_names
                )
                server_infos.append(server_info)
                
            except Exception as e:
                logger.error(f"Error setting up MCP client for server {server_config}: {e}")
                continue  # Continue with other servers
        
        if not server_infos:
            logger.warning("No servers were successfully connected")
            await exit_stack.aclose()
            return [], None
            
        return server_infos, exit_stack
        
    except Exception as e:
        logger.error(f"Error setting up MCP clients: {e}")
        await exit_stack.aclose()
        return [], None


# Keep the old function for backward compatibility
async def setup_mcp_client(server_config=None):
    """
    Sets up an MCP client session to connect to a server.
    Deprecated: Use setup_mcp_clients for multiple server support.
    
    Args:
        server_config (MCPServerConfig): Server configuration object
    
    Returns:
        tuple: (session, tools, exit_stack) or (None, [], None) on error
    """
    server_infos, exit_stack = await setup_mcp_clients([server_config] if server_config else None)
    if server_infos:
        server_info = server_infos[0]
        return server_info.session, server_info.tools, exit_stack
    return None, [], None


# --- 6. Deprecated - Removed call_responses_api function
# This functionality is now handled by the InferenceClient class


# --- 7. Tool Execution Functions ---
async def execute_mcp_tool(mcp_session, function_name, function_args):
    """Execute an MCP tool and return the result."""
    try:
        logger.info(f"Calling MCP tool: {function_name} with args: {function_args}")
        tool_output_obj = await mcp_session.call_tool(
            name=function_name, arguments=function_args
        )
        tool_output = extract_mcp_tool_result(tool_output_obj)
        logger.info(f"MCP tool output: {tool_output}")
        return tool_output
    except Exception as e:
        logger.error(f"Error executing MCP tool '{function_name}': {e}")
        raise


# This function has been replaced by _process_tool_calls_and_get_response in the Agent class


# --- 8. Agent Class ---
class Agent:
    """OpenAI MCP Agent - AI assistant with configurable MCP server support using Responses API."""

    def __init__(self, server_configs=None, server_specs=None, tools=None, inference_client=None):
        """
        Initialize the Agent with server configurations.

        Args:
            server_configs (List[MCPServerConfig] or MCPServerConfig): Server configuration objects
            server_specs (List[str] or str): Server specification strings (alternative to server_configs)
            tools (list): List of tool names to use (overrides tools in all configs)
            inference_client (InferenceClient, optional): Inference client to use for API calls
        """
        # Parse server configurations
        if server_configs is None and server_specs is not None:
            if isinstance(server_specs, str):
                server_specs = [server_specs]
            server_configs = [parse_server_config(spec) for spec in server_specs]
        
        if server_configs is None:
            server_configs = [LocalMCPServerConfig("weather_mcp_server.py")]
        
        # Ensure server_configs is a list
        if not isinstance(server_configs, list):
            server_configs = [server_configs]
        
        # Override tools if specified (applies to all servers)
        if tools is not None:
            for config in server_configs:
                config.tools = tools
        
        self.server_configs = server_configs
        self.inference_client = inference_client or get_default_client()
        self.server_infos: List[ServerInfo] = []
        self.exit_stack = None
        self.last_response_id = None
        self.instructions = "You are a helpful AI assistant. You can answer general questions and use available tools to help users."
        self.is_initialized = False

    @property
    def all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools from all connected servers."""
        tools = []
        for server_info in self.server_infos:
            tools.extend(server_info.tools)
        return tools

    @property
    def all_tool_names(self) -> List[str]:
        """Get all tool names from all connected servers."""
        names = []
        for server_info in self.server_infos:
            names.extend(server_info.tool_names)
        return names

    def find_server_for_tool(self, tool_name: str) -> Optional[ServerInfo]:
        """Find which server provides a specific tool."""
        for server_info in self.server_infos:
            if tool_name in server_info.tool_names:
                return server_info
        return None

    async def initialize(self):
        """Initialize the agent with MCP clients and tools."""
        if self.is_initialized:
            return True

        self.server_infos, self.exit_stack = await setup_mcp_clients(self.server_configs)
        if not self.server_infos:
            return False

        logger.info(f"Agent initialized with {len(self.server_infos)} server(s)")
        logger.info(f"Total available tools: {self.all_tool_names}")
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
            tools=self.all_tools, # Use all_tools here
            previous_response_id=self.last_response_id,
            instructions=self.instructions
        )

        if not response:
            return "Could not get a response from OpenAI Responses API."

        # Update last response ID for conversation continuity
        self.last_response_id = response.id

        # Loop to handle multiple rounds of tool calls
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            # Process the response
            assistant_message, tool_calls = process_responses_api_output(response)

            if tool_calls:
                logger.debug(f"Iteration {iteration}: Processing {len(tool_calls)} tool calls")
                # Process tool calls and get the next response
                response = await self._process_tool_calls_and_get_response(
                    tool_calls, self.last_response_id
                )
                
                if not response:
                    return "Error processing tool calls."
                
                # Update response ID
                self.last_response_id = response.id
                iteration += 1
            else:
                # No tool calls, return the assistant message
                return assistant_message or "No content in response"
        
        return "Maximum iterations reached while processing tool calls."
    
    async def _process_tool_calls_and_get_response(self, tool_calls, previous_response_id):
        """Process tool calls and return the next response."""
        input_items = []
        
        # Create a mapping of tool names to servers for quick lookup
        tool_to_server = {}
        all_tools = []
        for server_info in self.server_infos:
            all_tools.extend(server_info.tools)
            for tool_name in server_info.tool_names:
                tool_to_server[tool_name] = server_info
        
        # Process each tool call and create input items for both the function call and its output
        for tool_call_info in tool_calls:
            function_name = tool_call_info["function_name"]
            function_args = tool_call_info["function_args"]
            tool_call_id = tool_call_info["id"]

            # Add the original function call to input
            function_call_input = {
                "type": "function_call",
                "call_id": tool_call_id,
                "name": function_name,
                "arguments": json.dumps(function_args) if isinstance(function_args, dict) else str(function_args)
            }
            input_items.append(function_call_input)

            # Check if this is an MCP tool and find the appropriate server
            server_info = tool_to_server.get(function_name)
            if server_info:
                try:
                    tool_output = await execute_mcp_tool(
                        server_info.session, function_name, function_args
                    )
                    tool_response = build_tool_response_input(tool_call_id, tool_output)
                    input_items.append(tool_response)
                except Exception as e:
                    error_output = build_tool_response_input(
                        tool_call_id, f"Error executing tool '{function_name}': {e}"
                    )
                    input_items.append(error_output)
            else:
                logger.warning(f"Agent: Error - Unknown tool requested by LLM: {function_name}")
                error_output = build_tool_response_input(tool_call_id, "Unknown tool")
                input_items.append(error_output)
        
        # Make a follow-up call to the Responses API with both function calls and their outputs
        if input_items:
            response = self.inference_client.create_response(
                input_data=input_items,
                tools=all_tools,
                previous_response_id=previous_response_id
            )
            return response
        
        return None

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
    async def create(cls, server_configs=None, server_specs=None, tools=None, inference_client=None):
        """
        Factory method to create and initialize an Agent.

        Args:
            server_configs (List[MCPServerConfig] or MCPServerConfig): Server configuration objects
            server_specs (List[str] or str): Server specification strings (alternative to server_configs)
            tools (list): List of tool names to use (overrides tools in all configs)
            inference_client (InferenceClient, optional): Inference client to use for API calls

        Returns:
            Agent: Initialized agent, or None if initialization failed
        """
        agent = cls(server_configs, server_specs, tools, inference_client)
        success = await agent.initialize()
        return agent if success else None


# --- 9. Interactive Console Interface ---
async def run_interactive_agent(server_configs):
    """Run the agent in interactive console mode."""
    agent = await Agent.create(server_configs)
    if not agent:
        logger.warning("Failed to start MCP client. Exiting.")
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
                logger.error(f"Error processing message: {e}")
                print(f"Agent: Error processing your message. Please try again.")

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

Multiple servers:
  Use multiple --server arguments to connect to multiple servers:
    --server weather_mcp_server.py --server nps_mcp_server.py
    --server 'weather.py[get_weather]' --server 'remote:http://localhost:8000'

Examples:
  python openai_mcp_agent.py --server weather_mcp_server.py
  python openai_mcp_agent.py --server 'weather_mcp_server.py[get_current_weather]'
  python openai_mcp_agent.py --server local:my_server.py:node
  python openai_mcp_agent.py --server weather_mcp_server.py --tools get_current_weather
  python openai_mcp_agent.py --server remote:http://localhost:8000
  python openai_mcp_agent.py --server weather.py --server nps_server.py
  python openai_mcp_agent.py --server weather.py --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--server", "-s",
        type=str,
        action="append",  # Allow multiple --server arguments
        required=True,
        help="MCP server specification (can be specified multiple times for multiple servers)"
    )
    
    parser.add_argument(
        "--tools", "-t",
        type=str,
        help="Comma-separated list of tools to use (applies to all servers, overrides tools in server specs)"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)"
    )
    
    return parser.parse_args()


# --- 11. Main Entry Point ---
async def main():
    """Main entry point - runs interactive agent with configurable servers."""
    args = parse_arguments()
    
    # Configure logging based on command line argument
    log_level = getattr(logging, args.log_level.upper())
    setup_logging(log_level)
    
    try:
        # Parse all server configurations
        server_configs = []
        for server_spec in args.server:
            server_config = parse_server_config(server_spec, args.tools)
            server_configs.append(server_config)
        
        await run_interactive_agent(server_configs)
    except ValueError as e:
        logger.error(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("\nGoodbye!")
        return 0


# --- Run the agent ---
if __name__ == "__main__":
    exit_code = asyncio.run(main())
    if exit_code:
        exit(exit_code)
