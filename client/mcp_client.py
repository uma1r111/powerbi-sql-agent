# client/mcp_client.py

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

class NorthwindMCPClient:
    """
    An MCP Client that connects to our Northwind MCP Server.
    It exposes synchronous methods so it drops perfectly into our LangGraph nodes.
    """
    def __init__(self):
        # Tell the client how to start the MCP server process
        self.server_params = StdioServerParameters(
            command="python",
            args=["server/mcp_server.py"]
        )

    async def _run_tool(self, tool_name: str, arguments: dict) -> Any:
        """Core async method to spin up server, call tool, and close."""
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Execute the tool on the remote MCP server
                    result = await session.call_tool(tool_name, arguments=arguments)
                    
                    # FastMCP serializes dictionaries into JSON text
                    response_text = result.content[0].text
                    return json.loads(response_text)
                    
        except Exception as e:
            logger.error(f"MCP Client Error calling {tool_name}: {e}")
            return {"success": False, "error": str(e), "data": []}

    def execute_query(self, sql_query: str) -> Dict[str, Any]:
        """Synchronous wrapper for SQL execution"""
        return asyncio.run(self._run_tool("execute_sql_query", {"sql_query": sql_query}))

    def get_database_schema(self, table_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Synchronous wrapper for fetching schema"""
        return asyncio.run(self._run_tool("get_database_schema", {"table_names": table_names}))

# Create a global instance
mcp_client = NorthwindMCPClient()