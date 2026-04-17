import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# Add the project root to the Python path so imports work correctly
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the FastMCP wrapper from the official SDK
from mcp.server.fastmcp import FastMCP

# Import your existing tools
from tools.sql_tools import sql_executor
from tools.schema_tools import schema_inspector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Initialize the MCP Server
# Dependencies: pip install mcp
mcp = FastMCP(
    "Northwind_Database_Server",
    dependencies=["psycopg2-binary", "python-dotenv", "pandas"]
)

# 2. Expose SQL Execution as a Tool
@mcp.tool()
def execute_sql_query(sql_query: str) -> Dict[str, Any]:
    """
    Executes a PostgreSQL query against the Northwind database and returns the results.
    
    Args:
        sql_query: A valid, safe PostgreSQL query string to execute.
        
    Returns:
        A dictionary containing 'success' (bool), 'data' (list of rows), 
        'row_count' (int), and any 'error' messages.
    """
    logger.info(f"MCP Server executing query: {sql_query}")
    try:
        result = sql_executor.execute_query(sql_query)
        return result
    except Exception as e:
        logger.error(f"MCP SQL Execution failed: {e}")
        return {"success": False, "error": str(e), "data": []}

# 3. Expose Schema Inspection as a Tool
@mcp.tool()
def get_database_schema(table_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Retrieves the database schema context, including column names, descriptions, 
    and primary/foreign keys.
    
    Args:
        table_names: An optional list of specific table names (e.g., ['customers', 'orders']). 
                     If omitted, returns a high-level overview of the entire database.
                     
    Returns:
        A dictionary containing the schema details for the requested tables.
    """
    logger.info(f"MCP Server fetching schema for: {table_names or 'ALL TABLES'}")
    try:
        if table_names and len(table_names) > 0:
            return schema_inspector.get_multiple_tables_context(table_names)
        else:
            return schema_inspector.get_database_overview()
    except Exception as e:
        logger.error(f"MCP Schema fetch failed: {e}")
        return {"success": False, "error": str(e)}

# 4. Optional: Expose a dynamic "Resource" (Read-only data stream)
# MCP Resources allow clients to read internal data directly. 
# Here, we expose the raw schema text file as a resource.
@mcp.resource("db://schema/overview")
def read_schema_text() -> str:
    """Reads the raw database schema definition file."""
    schema_path = Path(project_root) / "database" / "schema.txt"
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Schema file not found."

if __name__ == "__main__":
    # Start the server using standard input/output (stdio)
    # This is the protocol required by MCP clients (like Claude Desktop or LangChain)
    logger.info("Starting Northwind MCP Server on stdio...")
    mcp.run(transport="stdio")