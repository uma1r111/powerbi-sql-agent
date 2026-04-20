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
    
    NEW: Includes caching and fallback mechanisms.
    """
    def __init__(self):
        env = os.environ.copy()
        # Set PYTHONPATH so the server subprocess can import project modules
        env["PYTHONPATH"] = str(BASE_DIR)
        
        self.server_params = StdioServerParameters(
            command=sys.executable,
            args=[SERVER_SCRIPT_PATH],
            env=env
        )
        
        # NEW: Cache for static content (prompts don't change at runtime)
        self._prompt_cache = {}
        self._resource_cache = {}
        self._cache_enabled = True

    # ------------------------------------------------------------------------
    # ASYNC CORE METHODS (Talks to the Server)
    # ------------------------------------------------------------------------
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

    async def _read_resource_async(self, uri: str) -> str:
        """Core async method to fetch a static resource or stream."""
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.read_resource(uri)
                    # Extract the raw text from the resource content
                    return result.contents[0].text
        except Exception as e:
            logger.error(f"MCP Client Error reading resource {uri}: {e}")
            return f"Error reading resource: {e}"

    async def _get_prompt_async(self, prompt_name: str, arguments: dict = None) -> str:
        """
        Core async method to fetch an AI prompt instruction.
        
        Args:
            prompt_name: Name of the prompt (e.g., "northwind_query_rules")
            arguments: Optional arguments to pass to the prompt
        """
        try:
            async with stdio_client(self.server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.get_prompt(prompt_name, arguments=arguments or {})
                    
                    # Prompts return a list of messages. Extract text from all messages.
                    if result.messages and len(result.messages) > 0:
                        # Combine all message contents
                        text_parts = []
                        for msg in result.messages:
                            if hasattr(msg.content, 'text'):
                                text_parts.append(msg.content.text)
                        return "\n".join(text_parts)
                    return ""
        except Exception as e:
            logger.error(f"MCP Client Error getting prompt {prompt_name}: {e}")
            # NEW: Fallback to local file
            return self._load_fallback_prompt(prompt_name)

    # ------------------------------------------------------------------------
    # FALLBACK MECHANISMS (NEW)
    # ------------------------------------------------------------------------
    def _load_fallback_prompt(self, prompt_name: str) -> str:
        """
        Load prompt from local fallback file if MCP server unavailable.
        
        Args:
            prompt_name: Name of the prompt
        
        Returns:
            Prompt content from local file, or empty string if not found
        """
        fallback_path = BASE_DIR / "config" / f"{prompt_name}_fallback.txt"
        
        if fallback_path.exists():
            logger.warning(f"MCP unavailable, using fallback: {fallback_path}")
            with open(fallback_path, "r", encoding="utf-8") as f:
                return f.read()
        
        logger.error(f"No fallback found for prompt: {prompt_name}")
        return ""

    # ------------------------------------------------------------------------
    # SYNC RUNNER (FastAPI/LangGraph Bridge)
    # ------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------
    # PUBLIC SYNC API (What LangGraph Nodes actually call)
    # ------------------------------------------------------------------------
    def execute_query(self, sql_query: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Synchronous wrapper for SQL execution.
        
        Args:
            sql_query: SQL query to execute
            session_id: Session identifier for error tracking
        """
        return self._run_sync(
            self._run_tool_async("execute_sql_query", {
                "sql_query": sql_query,
                "session_id": session_id
            })
        )

    def get_database_schema(self, table_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Synchronous wrapper for fetching schema"""
        return self._run_sync(
            self._run_tool_async("get_database_schema", {"table_names": table_names})
        )

    def read_resource(self, uri: str, use_cache: bool = True) -> str:
        """
        Synchronous wrapper for reading an MCP resource.
        
        Args:
            uri: Resource URI (e.g., "db://logs/recent_errors")
            use_cache: Whether to use cached value for static resources
        
        Returns:
            Resource content as string
        """
        # Check cache for static resources
        if use_cache and self._cache_enabled and uri in self._resource_cache:
            logger.debug(f"Using cached resource: {uri}")
            return self._resource_cache[uri]
        
        # Fetch from server
        content = self._run_sync(self._read_resource_async(uri))
        
        # Cache static resources (schema overview doesn't change)
        if use_cache and self._cache_enabled and uri == "db://schema/overview":
            self._resource_cache[uri] = content
        
        return content

    def get_prompt(self, prompt_name: str, use_cache: bool = True, arguments: dict = None) -> str:
        """
        Synchronous wrapper for fetching an MCP prompt.
        
        Args:
            prompt_name: Name of the prompt (e.g., "northwind_query_rules")
            use_cache: Whether to use cached value (prompts are static)
            arguments: Optional arguments for dynamic prompts
        
        Returns:
            Prompt content as string
        """
        # Check cache
        cache_key = f"{prompt_name}:{json.dumps(arguments or {})}"
        if use_cache and self._cache_enabled and cache_key in self._prompt_cache:
            logger.debug(f"Using cached prompt: {prompt_name}")
            return self._prompt_cache[cache_key]
        
        # Fetch from server
        content = self._run_sync(self._get_prompt_async(prompt_name, arguments))
        
        # Cache result
        if use_cache and self._cache_enabled:
            self._prompt_cache[cache_key] = content
        
        return content
    
    def clear_cache(self):
        """Clear all cached prompts and resources (useful for testing)"""
        self._prompt_cache.clear()
        self._resource_cache.clear()
        logger.info("MCP client cache cleared")

# Create a global instance
mcp_client = NorthwindMCPClient()