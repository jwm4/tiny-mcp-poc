# National Park Service MCP Server

A Model Context Protocol (MCP) server that provides access to the National Park Service APIs, allowing AI agents to retrieve information about national parks, alerts, campgrounds, events, and visitor centers.

## Features

The NPS MCP Server provides 5 main tools:

1. **search_parks** - Search for national parks by state, park code, or query string
2. **get_park_alerts** - Get current alerts for specific parks
3. **get_park_campgrounds** - Get campground information including amenities and reservations
4. **get_park_events** - Get upcoming events and programs
5. **get_visitor_centers** - Get visitor center locations and operating hours

## Quick Start

### 1. Get an API Key (Optional)

While the server works with a demo key, for full functionality get a free API key:
- Visit: https://www.nps.gov/subjects/developer/get-started.htm
- Register for a free API key
- Set the environment variable: `export NPS_API_KEY=your_api_key_here`

### 2. Install Dependencies

```bash
pip install fastmcp httpx
```

### 3. Run the Server

**Local mode (stdio transport):**
```bash
python nps_mcp_server.py
```

**Remote mode (HTTP/SSE transport):**
```bash
python nps_mcp_server.py --transport sse --port 3000
```

### 4. Use with an Agent

```python
from openai_mcp_agent import Agent

# Create agent with NPS server
agent = Agent(server_specs="nps_mcp_server.py")
await agent.initialize()

# Ask about parks
response = await agent.chat("Find national parks in California")
print(response)

await agent.cleanup()
```

Alternatively, you can use it with the CLI for
[openai_mcp_agent.py](./openai_mcp_agent.py), which will provide a simple interactive
chat session, e.g.:

```bash
python openai_mcp_agent.py --server nps_mcp_server.py
```

or:

```bash
python nps_mcp_server.py --transport sse --port 3000 &
python openai_mcp_agent.py --server remote:http://localhost:3000
```

## API Reference

### search_parks

Search for national parks with flexible filtering options.

**Parameters:**
- `state_code` (optional): Two-letter state code (e.g., 'CA', 'NY')
- `park_code` (optional): Four-letter park code (e.g., 'yell', 'acad')
- `query` (optional): Search query for park names or descriptions
- `limit` (optional): Maximum results to return (default: 10)

**Example:**
```python
# Search by state
await agent.chat("Search for parks in California")

# Search by park code
await agent.chat("Find information about Yellowstone (park code: yell)")

# Search by query
await agent.chat("Find parks with 'mountain' in the name")
```

### get_park_alerts

Get current alerts and announcements for a specific park.

**Parameters:**
- `park_code`: Four-letter park code (required)

**Example:**
```python
await agent.chat("Get alerts for Grand Canyon (park code: grca)")
```

### get_park_campgrounds

Get campground information including amenities, fees, and reservation details.

**Parameters:**
- `park_code`: Four-letter park code (required)
- `limit` (optional): Maximum results to return (default: 10)

**Example:**
```python
await agent.chat("Find campgrounds in Yosemite (park code: yose)")
```

### get_park_events

Get upcoming events, programs, and activities.

**Parameters:**
- `park_code`: Four-letter park code (required)
- `limit` (optional): Maximum results to return (default: 10)

**Example:**
```python
await agent.chat("What events are happening at Acadia (park code: acad)?")
```

### get_visitor_centers

Get visitor center locations, hours, and contact information.

**Parameters:**
- `park_code`: Four-letter park code (required)
- `limit` (optional): Maximum results to return (default: 10)

**Example:**
```python
await agent.chat("Find visitor centers in Zion (park code: zion)")
```

## Common Park Codes

Here are some popular national park codes:

- **yell** - Yellowstone National Park
- **grca** - Grand Canyon National Park
- **yose** - Yosemite National Park
- **acad** - Acadia National Park
- **zion** - Zion National Park
- **romo** - Rocky Mountain National Park
- **grsm** - Great Smoky Mountains National Park
- **olym** - Olympic National Park
- **glac** - Glacier National Park
- **arch** - Arches National Park

## Command Line Options

```bash
python nps_mcp_server.py --help
```

**Transport Modes:**
- `--transport stdio` (default): Local communication via stdin/stdout
- `--transport sse`: HTTP-based Server-Sent Events for remote clients

**Network Options:**
- `--host HOST`: Host to bind to (default: localhost)
- `--port PORT`: Port to bind to (default: 3000)

## Environment Variables

- `NPS_API_KEY`: Your NPS API key (optional, uses DEMO_KEY if not set)

## Rate Limits

The NPS API has the following rate limits:
- **1,000 requests per hour** per API key
- Rate limit headers are included in responses
- 429 status code returned when limits exceeded

## Example Usage

See `example_nps_usage.py` for a complete demonstration:

```bash
python example_nps_usage.py
```

## Error Handling

The server handles common errors gracefully:
- **Network errors**: Connection timeouts, DNS failures
- **API errors**: Invalid park codes, rate limiting
- **Authentication errors**: Invalid or missing API keys
- **Data errors**: Malformed responses, missing fields

All errors are returned as JSON with descriptive error messages.

## Data Sources

This server uses the official National Park Service API:
- **Base URL**: https://developer.nps.gov/api/v1
- **Documentation**: https://www.nps.gov/subjects/developer/api-documentation.htm
- **Data includes**: Parks, alerts, campgrounds, events, visitor centers, news, articles

## License

This project follows the same license as the sample-agent repository. 