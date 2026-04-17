# client/mcp_client.py

import asyncio
import json
import logging
import concurrent.futures
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

# Calculate absolute path to the server script
BASE_DIR = Path(__file__).resolve().parent.parent
SERVER_SCRIPT_PATH = str(BASE_DIR / "server" / "mcp_server.py")

def _sync_runner(coro):
    """
    Manually creates a new event loop for the thread.
    Crucially forces the Proactor policy on Windows to allow subprocesses.
    """
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Explicitly create and set the loop for this specific thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

class NorthwindMCPClient:
    """
    An MCP Client that connects to our Northwind MCP Server.
    Thread-safe, Windows-safe, and Path-safe for FastAPI.
    """
    def __init__(self):
        env = os.environ.copy()
        # Set PYTHONPATH so the server subprocess can import project modules
        env["PYTHONPATH"] = str(BASE_DIR)
        
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[SERVER_SCRIPT_PATH],  # Use the absolute path!
            env=env
        )

    async def _run_tool_async(self, tool_name: str, arguments: dict) -> Any:
        """Core async method to spin up server, call tool, and close."""
        try:
            logger.info(f"Starting MCP subprocess at: {SERVER_SCRIPT_PATH}")
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=arguments)
                    response_text = result.content[0].text
                    return json.loads(response_text)
        except Exception as e:
            logger.error(f"MCP Client Error calling {tool_name}: {e}")
            return {"success": False, "error": str(e), "data": []}

    def _run_sync(self, coro) -> Any:
        """Safely runs an async coroutine from a sync context"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If inside FastAPI's event loop, run in a background thread
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(_sync_runner, coro).result()
        else:
            # If running normally
            return _sync_runner(coro)

    def execute_query(self, sql_query: str) -> Dict[str, Any]:
        """Synchronous wrapper for SQL execution"""
        return self._run_sync(self._run_tool_async("execute_sql_query", {"sql_query": sql_query}))

    def get_database_schema(self, table_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Synchronous wrapper for fetching schema"""
        return self._run_sync(self._run_tool_async("get_database_schema", {"table_names": table_names}))

# Create a global instance
mcp_client = NorthwindMCPClient()   