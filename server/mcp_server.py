# server/mcp_server.py
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from collections import defaultdict

project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mcp.server.fastmcp import FastMCP
from tools.sql_tools import sql_executor
from tools.schema_tools import schema_inspector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ERRORS_BY_SESSION = defaultdict(list)
MAX_ERRORS_PER_SESSION = 10
ERROR_RETENTION_HOURS = 1

def log_failed_query(query: str, error_msg: str, session_id: str = "default"):
    timestamp = datetime.now()
    ERRORS_BY_SESSION[session_id].append({
        "timestamp": timestamp.isoformat(),
        "query": query,
        "error": error_msg
    })
    if len(ERRORS_BY_SESSION[session_id]) > MAX_ERRORS_PER_SESSION:
        ERRORS_BY_SESSION[session_id].pop(0)
    _cleanup_old_sessions()

def _cleanup_old_sessions():
    cutoff = datetime.now() - timedelta(hours=ERROR_RETENTION_HOURS)
    to_delete = [
        sid for sid, errors in ERRORS_BY_SESSION.items()
        if errors and datetime.fromisoformat(errors[-1]["timestamp"]) < cutoff
    ]
    for sid in to_delete:
        del ERRORS_BY_SESSION[sid]

mcp = FastMCP(
    "Northwind_Database_Server",
    dependencies=["psycopg2-binary", "python-dotenv", "pandas"]
)

# --- TOOLS ---

@mcp.tool()
def execute_sql_query(sql_query: str, session_id: str = "default", max_rows: int = 50) -> Dict[str, Any]:
    """
    Executes a PostgreSQL query against the Northwind database.

    Args:
        sql_query: Valid PostgreSQL query string
        session_id: Optional session identifier for error tracking
        max_rows: Max rows to return (default 50, lower = fewer tokens)
    """
    logger.info(f"MCP executing query [{session_id}]: {sql_query[:80]}...")
    try:
        result = sql_executor.execute_query(sql_query, limit=max_rows)

        if isinstance(result, dict) and not result.get("success", True):
            log_failed_query(sql_query, result.get("error", "Unknown DB Error"), session_id)

        # Strip high-token fields the LLM doesn't need
        return _slim_sql_result(result)
    except Exception as e:
        logger.error(f"MCP SQL Execution failed: {e}")
        log_failed_query(sql_query, str(e), session_id)
        return {"success": False, "error": str(e), "data": []}


def _slim_sql_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only fields the LLM needs; drop verbose/redundant ones."""
    if result.get("success"):
        return {
            "success": True,
            "data": result.get("data", []),
            "row_count": result.get("row_count", 0),
            "column_names": result.get("column_names", []),
        }
    else:
        return {
            "success": False,
            "error": result.get("error") or result.get("message", "Unknown error"),
        }


@mcp.tool()
def get_database_schema(table_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Retrieves database schema.

    Args:
        table_names: Specific tables to fetch (e.g. ['customers', 'orders']).
                     Omit to get a slim overview of all tables.
    """
    logger.info(f"MCP fetching schema: {table_names or 'OVERVIEW'}")
    try:
        if table_names and len(table_names) > 0:
            raw = schema_inspector.get_multiple_tables_context(table_names)
            return _slim_multi_table(raw)
        else:
            raw = schema_inspector.get_database_overview()
            return _slim_overview(raw)
    except Exception as e:
        logger.error(f"MCP Schema fetch failed: {e}")
        return {"success": False, "error": str(e)}


def _slim_overview(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return only table names + one-line descriptions. Drops relationships/analyses."""
    summary = {
        name: info.get("description", "")
        for name, info in raw.get("table_summary", {}).items()
    }
    return {
        "success": True,
        "total_tables": raw.get("total_tables", 0),
        "tables": summary,
    }


def _slim_multi_table(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return only key_fields + column types. Drops sample_queries, join_suggestions, indirect_connections."""
    slim_tables = {}
    for tname, ctx in raw.get("tables", {}).items():
        slim_tables[tname] = {
            "description": ctx.get("description", ""),
            "key_fields": ctx.get("key_fields", []),
            "foreign_keys": ctx.get("relationships", {}).get("foreign_keys", {}),
        }
    return {
        "success": True,
        "tables": slim_tables,
        "relationships": raw.get("inter_table_relationships", {}),
    }


# --- RESOURCES ---

@mcp.resource("db://schema/overview")
def read_schema_text() -> str:
    """Raw schema file for manual inspection."""
    schema_path = Path(project_root) / "database" / "schema.txt"
    if schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Schema file not found."


@mcp.resource("db://logs/errors/{session_id}")
def get_recent_errors_log(session_id: str) -> str:
    """Recent failed SQL queries for this session (last 10, session-scoped)."""
    errors = ERRORS_BY_SESSION.get(session_id, [])
    if not errors:
        return "No recent SQL errors."

    lines = []
    for i, e in enumerate(errors, 1):
        lines.append(f"#{i} [{e['timestamp']}] {e['query']} => {e['error']}")

    if len(errors) >= 3:
        lines.append("WARNING: Multiple failures — check schema or query structure.")

    return "\n".join(lines)


# --- PROMPTS ---

@mcp.prompt()
def northwind_query_rules() -> List[Dict[str, Any]]:
    """Critical rules for querying the Northwind PostgreSQL database correctly."""
    rules = (
        "NORTHWIND SQL RULES:\n"
        "1. String match: use ILIKE, not =. E.g. WHERE ship_country ILIKE 'germany'\n"
        "2. Revenue: SUM(unit_price * quantity * (1 - discount)) from order_details\n"
        "3. Column names: ship_country (not country), company_name, product_name, first_name/last_name\n"
        "4. Countries: full names — 'United Kingdom' not 'UK', 'USA' not 'US'\n"
        "5. Always add LIMIT (default 50) unless using aggregation\n"
        "6. Year filter: EXTRACT(YEAR FROM order_date) = 1997\n"
        "7. Revenue join chain: customers→orders→order_details→products\n"
        "8. Territories: employees→employee_territories→territories→region\n"
        "9. Out of stock: units_in_stock=0 | Reorder: units_in_stock<=reorder_level\n"
        "Tables: customers,orders,order_details,products,categories,suppliers,"
        "employees,shippers,territories,region,employee_territories"
    )
    return [{"role": "user", "content": {"type": "text", "text": rules}}]


if __name__ == "__main__":
    logger.info("Starting Northwind MCP Server on stdio...")
    mcp.run(transport="stdio")
