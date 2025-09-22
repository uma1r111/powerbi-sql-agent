# tools/schema_tools.py

import logging
from typing import Dict, List, Any, Optional
from database.northwind_context import BUSINESS_CONTEXT, BUSINESS_SCENARIO
from database.relationships import (
    FOREIGN_KEY_RELATIONSHIPS, 
    TABLE_CONNECTIONS, 
    COMMON_JOIN_PATTERNS,
    get_related_tables,
    get_join_path
)
from database.sample_queries import SAMPLE_QUERIES, get_queries_with_tables

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaInspectorTool:
    """
    Tool for inspecting database schema and providing business context
    """
    
    def __init__(self):
        self.name = "SchemaInspectorTool"
        self.description = "Provides schema information with business context for query planning"
    
    def get_table_context(self, table_name: str) -> Dict[str, Any]:
        """
        Get comprehensive context for a specific table
        
        Args:
            table_name: Name of the table to inspect
            
        Returns:
            Dict containing business context, relationships, and sample queries
        """
        if table_name not in BUSINESS_CONTEXT:
            return {
                "success": False,
                "message": f"Table '{table_name}' not found in business context",
                "available_tables": list(BUSINESS_CONTEXT.keys())
            }
        
        table_info = BUSINESS_CONTEXT[table_name].copy()
        
        # Add relationship information
        related_tables = get_related_tables(table_name)
        table_info["relationships"] = {
            "direct_connections": related_tables["direct"],
            "indirect_connections": related_tables["indirect"],
            "foreign_keys": FOREIGN_KEY_RELATIONSHIPS.get(table_name, {}),
            "join_suggestions": self._get_join_suggestions(table_name)
        }
        
        # Add relevant sample queries
        relevant_queries = get_queries_with_tables([table_name])
        table_info["sample_queries"] = {
            query_id: query_info["natural_language"] 
            for query_id, query_info in relevant_queries.items()
        }
        
        return {
            "success": True,
            "table_name": table_name,
            "context": table_info
        }
    
    def get_multiple_tables_context(self, table_names: List[str]) -> Dict[str, Any]:
        """
        Get context for multiple tables and their relationships
        """
        contexts = {}
        relationships_map = {}
        
        for table_name in table_names:
            context = self.get_table_context(table_name)
            if context["success"]:
                contexts[table_name] = context["context"]
        
        # Find relationships between the specified tables
        for i, table1 in enumerate(table_names):
            for table2 in table_names[i+1:]:
                join_path = get_join_path(table1, table2)
                if "No known relationship" not in join_path:
                    relationships_map[f"{table1}_to_{table2}"] = join_path
        
        return {
            "success": True,
            "tables": contexts,
            "inter_table_relationships": relationships_map,
            "suggested_joins": self._suggest_joins_for_tables(table_names)
        }
    
    def get_database_overview(self) -> Dict[str, Any]:
        """
        Get high-level overview of the entire database
        """
        return {
            "success": True,
            "business_scenario": BUSINESS_SCENARIO,
            "total_tables": len(BUSINESS_CONTEXT),
            "table_summary": {
                name: {
                    "description": info["description"],
                    "row_count": info["row_count"],
                    "business_importance": info["business_importance"]
                }
                for name, info in BUSINESS_CONTEXT.items()
            },
            "key_relationships": self._get_key_relationships(),
            "common_analysis_patterns": BUSINESS_SCENARIO["common_analyses"]
        }
    
    def suggest_tables_for_query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Suggest relevant tables based on natural language query
        """
        query_lower = natural_language_query.lower()
        suggested_tables = []
        
        # Keyword mapping for table suggestion
        table_keywords = {
            "customers": ["customer", "client", "company", "contact"],
            "orders": ["order", "purchase", "sale", "transaction"],
            "order_details": ["product", "item", "quantity", "price", "revenue", "detail"],
            "products": ["product", "item", "inventory", "stock", "catalog"],
            "categories": ["category", "type", "group", "classification"],
            "suppliers": ["supplier", "vendor", "provider"],
            "employees": ["employee", "staff", "sales rep", "worker"],
            "shippers": ["shipping", "delivery", "freight", "shipper"]
        }
        
        # Score tables based on keyword matches
        table_scores = {}
        for table_name, keywords in table_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                table_scores[table_name] = score
        
        # Sort by relevance score
        suggested_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Get context for top suggestions
        top_suggestions = [table for table, score in suggested_tables[:3]]
        table_contexts = {}
        
        for table in top_suggestions:
            context = self.get_table_context(table)
            if context["success"]:
                table_contexts[table] = {
                    "description": context["context"]["description"],
                    "key_fields": context["context"]["key_fields"],
                    "relevance_score": dict(suggested_tables)[table]
                }
        
        return {
            "success": True,
            "query": natural_language_query,
            "suggested_tables": table_contexts,
            "all_scores": dict(suggested_tables)
        }
    
    def _get_join_suggestions(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get common join patterns involving this table
        """
        suggestions = []
        
        for pattern_name, pattern_info in COMMON_JOIN_PATTERNS.items():
            if table_name in pattern_info["tables"]:
                suggestions.append({
                    "pattern_name": pattern_name,
                    "description": pattern_info["description"],
                    "tables": pattern_info["tables"],
                    "join_condition": pattern_info.get("join_condition", 
                                                     pattern_info.get("join_conditions"))
                })
        
        return suggestions
    
    def _suggest_joins_for_tables(self, table_names: List[str]) -> List[Dict[str, Any]]:
        """
        Suggest optimal joins for a set of tables
        """
        suggestions = []
        
        for pattern_name, pattern_info in COMMON_JOIN_PATTERNS.items():
            pattern_tables = pattern_info["tables"]
            
            # Check if all tables in the pattern are in our requested tables
            if all(table in table_names for table in pattern_tables if not table.startswith('employees e')):
                suggestions.append({
                    "pattern_name": pattern_name,
                    "description": pattern_info["description"],
                    "tables": pattern_tables,
                    "sql_pattern": pattern_info.get("join_condition", 
                                                  pattern_info.get("join_conditions"))
                })
        
        return suggestions
    
    def _get_key_relationships(self) -> List[Dict[str, str]]:
        """
        Get the most important relationships in the database
        """
        key_relationships = [
            {
                "from": "customers",
                "to": "orders", 
                "type": "one_to_many",
                "description": "Customers place orders"
            },
            {
                "from": "orders",
                "to": "order_details",
                "type": "one_to_many", 
                "description": "Orders contain multiple line items"
            },
            {
                "from": "products",
                "to": "order_details",
                "type": "one_to_many",
                "description": "Products appear in order line items"
            },
            {
                "from": "categories",
                "to": "products",
                "type": "one_to_many",
                "description": "Categories contain products"
            },
            {
                "from": "suppliers", 
                "to": "products",
                "type": "one_to_many",
                "description": "Suppliers provide products"
            }
        ]
        
        return key_relationships
    
    def get_tool_spec(self) -> Dict[str, Any]:
        """
        Returns the tool specification for LangGraph
        """
        return {
            "name": "SchemaInspectorTool",
            "description": "Provides database schema information with business context for intelligent query planning",
            "parameters": {
                "operation": {
                    "type": "string",
                    "description": "Type of schema operation: 'table_context', 'multiple_tables', 'database_overview', 'suggest_tables'",
                    "required": True,
                    "enum": ["table_context", "multiple_tables", "database_overview", "suggest_tables"]
                },
                "table_name": {
                    "type": "string",
                    "description": "Name of the table (for table_context operation)",
                    "required": False
                },
                "table_names": {
                    "type": "array",
                    "description": "List of table names (for multiple_tables operation)", 
                    "required": False
                },
                "query": {
                    "type": "string",
                    "description": "Natural language query (for suggest_tables operation)",
                    "required": False
                }
            }
        }

# Tool instance
schema_inspector = SchemaInspectorTool()

# Export for LangGraph integration
SCHEMA_TOOL_REGISTRY = {
    "SchemaInspectorTool": schema_inspector
}