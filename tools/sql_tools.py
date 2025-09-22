# tools/sql_tools.py

import psycopg2
import json
import logging
from typing import Dict, List, Any, Optional
from database.connection import get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLExecutorTool:
    """
    Tool for executing SQL queries and formatting results for the agent
    """
    
    def __init__(self):
        self.name = "SQLExecutorTool"
        self.description = "Executes SQL queries on the database and returns formatted results"
        
    def execute_query(self, query: str, limit: Optional[int] = 100) -> Dict[str, Any]:
        """
        Execute SQL query and return formatted results
        
        Args:
            query: SQL query string
            limit: Maximum number of rows to return (default 100)
            
        Returns:
            Dict containing success status, data, metadata, and any errors
        """
        try:
            # Add LIMIT if not present and it's a SELECT query
            if query.strip().upper().startswith('SELECT') and 'LIMIT' not in query.upper():
                query = f"{query.rstrip(';')} LIMIT {limit};"
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            logger.info(f"Executing query: {query}")
            cursor.execute(query)
            
            # Handle different query types
            if query.strip().upper().startswith(('SELECT', 'WITH')):
                # SELECT queries - fetch results
                results = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                
                # Format as list of dictionaries
                formatted_results = []
                for row in results:
                    formatted_results.append(dict(zip(column_names, row)))
                
                response = {
                    "success": True,
                    "query": query,
                    "data": formatted_results,
                    "row_count": len(results),
                    "column_names": column_names,
                    "message": f"Query executed successfully. Returned {len(results)} rows."
                }
                
            else:
                # INSERT, UPDATE, DELETE queries
                row_count = cursor.rowcount
                conn.commit()
                
                response = {
                    "success": True,
                    "query": query,
                    "data": None,
                    "row_count": row_count,
                    "column_names": None,
                    "message": f"Query executed successfully. {row_count} rows affected."
                }
            
            cursor.close()
            conn.close()
            
            return response
            
        except psycopg2.Error as db_error:
            logger.error(f"Database error: {db_error}")
            return {
                "success": False,
                "query": query,
                "data": None,
                "row_count": 0,
                "column_names": None,
                "error": str(db_error),
                "error_type": "database_error",
                "message": f"Database error: {db_error}"
            }
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {
                "success": False,
                "query": query,
                "data": None,
                "row_count": 0,
                "column_names": None,
                "error": str(e),
                "error_type": "execution_error",
                "message": f"Execution error: {e}"
            }
    
    def format_results_for_display(self, results: Dict[str, Any]) -> str:
        """
        Format query results for human-readable display
        """
        if not results["success"]:
            return f"❌ Query failed: {results['message']}"
        
        if results["data"] is None:
            return f"✅ {results['message']}"
        
        data = results["data"]
        if not data:
            return "✅ Query executed successfully but returned no results."
        
        # Create a simple table format
        output = f"✅ Query executed successfully. Found {len(data)} rows:\n\n"
        
        if len(data) > 0:
            # Get column headers
            headers = list(data[0].keys())
            
            # Create header row
            header_row = " | ".join(str(header).ljust(15)[:15] for header in headers)
            separator = "-" * len(header_row)
            
            output += header_row + "\n" + separator + "\n"
            
            # Add data rows (limit to first 10 for display)
            for i, row in enumerate(data[:10]):
                row_data = " | ".join(str(row.get(header, "")).ljust(15)[:15] for header in headers)
                output += row_data + "\n"
            
            if len(data) > 10:
                output += f"\n... and {len(data) - 10} more rows"
        
        return output
    
    def get_tool_spec(self) -> Dict[str, Any]:
        """
        Returns the tool specification for LangGraph
        """
        return {
            "name": "SQLExecutorTool",
            "description": "Executes SQL queries on the Northwind database and returns formatted results",
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "The SQL query to execute",
                    "required": True
                },
                "limit": {
                    "type": "integer", 
                    "description": "Maximum number of rows to return (default 100)",
                    "required": False,
                    "default": 100
                }
            }
        }

class DatabaseInfoTool:
    """
    Tool for getting database information and metadata
    """
    
    def __init__(self):
        self.name = "DatabaseInfoTool"
        self.description = "Provides database structure and metadata information"
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific table
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get column information
            cursor.execute("""
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns 
                WHERE table_name = %s AND table_schema = 'public'
                ORDER BY ordinal_position;
            """, (table_name,))
            
            columns = cursor.fetchall()
            
            if not columns:
                return {
                    "success": False,
                    "message": f"Table '{table_name}' not found"
                }
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
            row_count = cursor.fetchone()[0]
            
            # Format column information
            column_info = []
            for col_name, data_type, nullable, default in columns:
                column_info.append({
                    "name": col_name,
                    "type": data_type,
                    "nullable": nullable == "YES",
                    "default": default
                })
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "table_name": table_name,
                "columns": column_info,
                "row_count": row_count,
                "column_count": len(column_info)
            }
            
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to get table information: {e}"
            }
    
    def get_all_tables(self) -> List[str]:
        """
        Get list of all tables in the database
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                ORDER BY table_name;
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            
            cursor.close()
            conn.close()
            
            return tables
            
        except Exception as e:
            logger.error(f"Error getting table list: {e}")
            return []

# Tool instances that can be imported
sql_executor = SQLExecutorTool()
db_info = DatabaseInfoTool()

# Tool registry for LangGraph
TOOL_REGISTRY = {
    "SQLExecutorTool": sql_executor,
    "DatabaseInfoTool": db_info
}

def get_tool(tool_name: str):
    """Get a tool instance by name"""
    return TOOL_REGISTRY.get(tool_name)

def get_all_tool_specs() -> List[Dict[str, Any]]:
    """Get all tool specifications for LangGraph"""
    return [tool.get_tool_spec() for tool in TOOL_REGISTRY.values() if hasattr(tool, 'get_tool_spec')]