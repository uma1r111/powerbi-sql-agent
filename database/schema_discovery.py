import logging
from typing import Dict, List, Any
from database.connection import get_db_connection

logger = logging.getLogger(__name__)

class SchemaDiscovery:
    """
    Dynamically discovers database schema, replacing hardcoded contexts.
    """
    
    def __init__(self):
        self.conn = get_db_connection()
        if not self.conn:
            raise Exception("Failed to connect to database in SchemaDiscovery")
    
    def get_schema_context(self) -> Dict[str, Any]:
        """
        Generates a dictionary mimicking the old BUSINESS_CONTEXT
        but populated dynamically from the database.
        """
        schema_context = {}
        tables = self._get_all_tables()
        
        for table in tables:
            schema_context[table] = {
                "description": self._infer_table_purpose(table),
                "key_fields": self._get_primary_keys(table),
                "columns": self._get_columns(table),
                "sample_values": self._get_sample_values(table)
            }
            
        return schema_context

    def _get_all_tables(self) -> List[str]:
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """
        return [row['table_name'] for row in self._execute_query(query)]

    def _get_primary_keys(self, table_name: str) -> List[str]:
        query = """
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = %s::regclass AND i.indisprimary
        """
        try:
            results = self._execute_query(query, (table_name,))
            return [row['attname'] for row in results]
        except:
            return []

    def _get_columns(self, table_name: str) -> List[str]:
        query = "SELECT column_name FROM information_schema.columns WHERE table_name = %s"
        results = self._execute_query(query, (table_name,))
        return [row['column_name'] for row in results]

    def _get_sample_values(self, table_name: str) -> str:
        """Safe sample value extraction"""
        try:
            # Get first text-like column that isn't an ID
            col_query = """
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = %s AND data_type IN ('text', 'character varying')
                LIMIT 1
            """
            cols = self._execute_query(col_query, (table_name,))
            if not cols: return ""
            
            target_col = cols[0]['column_name']
            query = f"SELECT {target_col} FROM {table_name} LIMIT 3"
            rows = self._execute_query(query)
            return ", ".join([str(list(r.values())[0]) for r in rows])
        except:
            return ""

    def _infer_table_purpose(self, table_name: str) -> str:
        """Auto-generate descriptions based on naming conventions"""
        name = table_name.lower()
        if 'customer' in name: return "Customer details and contact info"
        if 'order' in name: return "Transaction headers and sales records"
        if 'detail' in name: return "Line item details for transactions"
        if 'product' in name: return "Product catalog and inventory status"
        if 'empl' in name: return "Internal staff and sales representatives"
        return f"Data table for {table_name}"

    def _execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Helper to run queries and return dicts"""
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
            return []
        except Exception as e:
            logger.error(f"Discovery error on {query}: {e}")
            return []
        finally:
            cursor.close()