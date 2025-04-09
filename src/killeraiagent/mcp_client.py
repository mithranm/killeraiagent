# killeraiagent/mcp_client.py

import httpx
import time

CACHE_DURATION = 60  # seconds; adjust as needed

class MCPClient:
    """
    A generic MCP client for connecting to an MCP server via HTTP.
    This client exposes only the basic methods: list_tools() and call_tool(),
    and caches the tools list to reduce HTTP round trips.
    """
    def __init__(self, base_url: str):
        """
        Initialize the MCP client with the base URL of the MCP server.
        For example: MCPClient("http://localhost:8001")
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url)
        self._cached_tools = None
        self._cache_timestamp = 0

    async def list_tools(self, force_refresh: bool = False) -> dict:
        """
        Retrieves the list of tools available on the MCP server.
        If a cached result is available and fresh, returns it; otherwise,
        makes an HTTP GET request to fetch the tool list.
        """
        current_time = time.time()
        if not force_refresh and self._cached_tools is not None and (current_time - self._cache_timestamp) < CACHE_DURATION:
            return self._cached_tools
        response = await self.client.get("/tools/list")
        response.raise_for_status()
        tools = response.json()
        self._cached_tools = tools
        self._cache_timestamp = current_time
        return tools

    async def call_tool(self, name: str, arguments: dict = None) -> dict:
        """
        Calls a tool on the MCP server by name, with a dictionary of arguments.
        """
        if arguments is None:
            arguments = {}
        payload = {"name": name, "arguments": arguments}
        response = await self.client.post("/tools/call", json=payload)
        response.raise_for_status()
        return response.json()

    async def close(self):
        """
        Closes the underlying HTTP client.
        """
        await self.client.aclose()
