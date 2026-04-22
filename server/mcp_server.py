# server/mcp_server.py
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from collections import defaultdict

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mcp.server.fastmcp import FastMCP
from tools.sql_tools import sql_executor
from tools.schema_tools import schema_inspector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SESSION-SCOPED ERROR TRACKING ---
# KEY CHANGE: Errors now stored per session_id for privacy
ERRORS_BY_SESSION = defaultdict(list)
MAX_ERRORS_PER_SESSION = 10
ERROR_RETENTION_HOURS = 1

def log_failed_query(query: str, error_msg: str, session_id: str = "default"):
    """
    Store errors with timestamp, scoped to session.
    
    Args:
        query: The failed SQL query
        error_msg: Error message from database
        session_id: User session identifier (for privacy isolation)
    """
    timestamp = datetime.now()
    error_entry = {
        "timestamp": timestamp.isoformat(),
        "query": query,
        "error": error_msg
    }
    
    # Store in session-specific list
    ERRORS_BY_SESSION[session_id].append(error_entry)
    
    # Limit to last N errors
    if len(ERRORS_BY_SESSION[session_id]) > MAX_ERRORS_PER_SESSION:
        ERRORS_BY_SESSION[session_id].pop(0)
    
    # Cleanup old sessions (prevent memory leak)
    _cleanup_old_sessions()

def _cleanup_old_sessions():
    """Remove error logs older than ERROR_RETENTION_HOURS"""
    cutoff = datetime.now() - timedelta(hours=ERROR_RETENTION_HOURS)
    sessions_to_delete = []
    
    for session_id, errors in ERRORS_BY_SESSION.items():
        if errors:
            last_error_time = datetime.fromisoformat(errors[-1]["timestamp"])
            if last_error_time < cutoff:
                sessions_to_delete.append(session_id)
    
    for session_id in sessions_to_delete:
        del ERRORS_BY_SESSION[session_id]
        logger.info(f"Cleaned up old error log for session: {session_id}")

# --- INITIALIZE SERVER ---
mcp = FastMCP(
    "Northwind_Database_Server",
    dependencies=["psycopg2-binary", "python-dotenv", "pandas"]
)

# --- TOOLS ---
@mcp.tool()
def execute_sql_query(sql_query: str, session_id: str = "default") -> Dict[str, Any]:
    """
    Executes a PostgreSQL query against the Northwind database.
    
    Args:
        sql_query: Valid PostgreSQL query string
        session_id: Optional session identifier for error tracking
    """
    logger.info(f"MCP Server executing query for session {session_id}: {sql_query[:100]}...")
    try:
        result = sql_executor.execute_query(sql_query)
        
        # Log failures for error resource
        if isinstance(result, dict) and not result.get("success", True):
            error_msg = result.get("error", "Unknown DB Error")
            log_failed_query(sql_query, error_msg, session_id)
        
        return result
    except Exception as e:
        logger.error(f"MCP SQL Execution failed: {e}")
        log_failed_query(sql_query, str(e), session_id)
        return {"success": False, "error": str(e), "data": []}

@mcp.tool()
def get_database_schema(table_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Retrieves the database schema context.
    
    Args:
        table_names: Optional list of specific tables (e.g., ['customers', 'orders'])
                     If omitted, returns overview of all tables
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

# --- RESOURCES ---
@mcp.resource("db://schema/overview")
def read_schema_text() -> str:
    """
    Raw database schema file for manual inspection.
    This is the same content your schema_inspector uses internally.
    """
    schema_path = Path(project_root) / "database" / "schema.txt"
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Schema file not found."

@mcp.resource("db://logs/errors/{session_id}")
def get_recent_errors_log(session_id: str) -> str:
    """
    Returns a log of recently failed SQL queries for this session.
    
    CRITICAL: This is session-scoped for privacy. Pass session_id from client.
    
    Args:
        session_id: User session identifier (defaults to "default" for testing)
    
    Returns:
        Formatted error log showing last 10 failures with timestamps
    """
    errors = ERRORS_BY_SESSION.get(session_id, [])
    
    if not errors:
        return "✅ No recent SQL errors for this session. Everything is running smoothly!"
    
    # Format errors for LLM consumption
    log_lines = ["=== RECENT SQL ERRORS (Last 10) ===\n"]
    
    for i, error_entry in enumerate(errors, 1):
        timestamp = error_entry["timestamp"]
        query = error_entry["query"]
        error_msg = error_entry["error"]
        
        log_lines.append(f"Error #{i} at {timestamp}:")
        log_lines.append(f"  Query: {query}")
        log_lines.append(f"  Error: {error_msg}")
        log_lines.append("")  # Blank line
    
    # Add pattern detection hint
    if len(errors) >= 3:
        log_lines.append("⚠️ PATTERN DETECTED: Multiple failures in sequence.")
        log_lines.append("   Consider reviewing schema or query structure.\n")
    
    return "\n".join(log_lines)

# --- PROMPTS ---
@mcp.prompt()
def northwind_query_rules() -> List[Dict[str, Any]]:
    """
    Provides specific rules for querying Northwind accurately.
    
    Returns:
        List of message objects (MCP prompt format)
    """
    rules_content = """
=== NORTHWIND DATABASE QUERY RULES ===

You are generating SQL for a PostgreSQL Northwind database. Follow these rules STRICTLY:

## 1. STRING COMPARISONS (CRITICAL)
- ❌ NEVER use `=` for string comparisons
- ✅ ALWAYS use `ILIKE` for case-insensitive matching
  Example: `WHERE ship_country ILIKE 'germany'`

## 2. REVENUE CALCULATION (MOST COMMON ERROR)
- Revenue formula: `SUM(unit_price * quantity * (1 - discount))`
- ❌ WRONG: `SUM(unit_price * quantity)` (ignores discount)
- ✅ CORRECT: `SUM(unit_price * quantity * (1 - discount))`
- Source table: `order_details` (NOT orders)

## 3. COMMON TABLE/COLUMN MISTAKES
- ❌ orders.country → ✅ orders.ship_country
- ❌ customers.name → ✅ customers.company_name
- ❌ products.name → ✅ products.product_name
- ❌ employees.name → ✅ employees.first_name, employees.last_name

## 4. TERRITORY QUERIES (COMMON JOIN ERROR)
- To find employee territories, use this JOIN chain:
```sql
  employees 
  → employee_territories (junction table)
  → territories
  → region
```

## 5. COUNTRY VALUES (DATA QUALITY ISSUE)
- Country values use FULL names, not abbreviations:
  - ❌ 'UK' → ✅ 'United Kingdom'
  - ❌ 'US' → ✅ 'USA'
  - ❌ 'DE' → ✅ 'Germany'

## 6. RESULT LIMITS (PERFORMANCE)
- ALWAYS add `LIMIT` clause unless doing aggregation (COUNT, SUM, etc.)
- Default limit: 100 rows
- Exception: When query explicitly asks for "all" or uses aggregate functions

## 7. DATE FILTERING
- Use `EXTRACT(YEAR FROM order_date) = 1997` for year filtering
- Use `order_date::date` for date comparisons without time

## 8. JOIN ORDER MATTERS
- Core transaction chain: customers → orders → order_details → products
- NEVER skip order_details when calculating revenue from orders

## 9. COMMON BUSINESS LOGIC
- "Out of stock": `units_in_stock = 0`
- "Needs reordering": `units_in_stock <= reorder_level`
- "Discontinued": `discontinued = 1`

## 10. SCHEMA REFERENCE
Available tables: customers, orders, order_details, products, categories, 
                 suppliers, employees, shippers, territories, region, 
                 employee_territories

When in doubt, use the `get_database_schema` tool to verify exact column names.
"""
    
    # Return in MCP prompt format (list of message objects)
    return [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": rules_content
            }
        }
    ]

if __name__ == "__main__":
    logger.info("Starting Northwind MCP Server on stdio...")
    mcp.run(transport="stdio")