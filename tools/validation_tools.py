# tools/validation_tools.py

import re
import logging
import psycopg2
from typing import Dict, List, Any, Optional, Tuple
from database.connection import get_db_connection
from database.northwind_context import BUSINESS_CONTEXT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryValidatorTool:
    """
    Tool for validating SQL queries before execution
    """
    
    def __init__(self):
        self.name = "QueryValidatorTool"
        self.description = "Validates SQL queries for syntax, security, and business logic"
        
        # Valid table names from our schema
        self.valid_tables = set(BUSINESS_CONTEXT.keys())
        
        # Dangerous keywords that should be blocked
        self.dangerous_keywords = {
            'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 
            'INSERT', 'UPDATE', 'GRANT', 'REVOKE', 'EXEC'
        }
        
        # Common SQL keywords for syntax checking
        self.sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL',
            'ON', 'GROUP', 'BY', 'HAVING', 'ORDER', 'LIMIT', 'OFFSET',
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'DISTINCT', 'AS'
        }
    
    def validate_query(self, sql_query: str, schema_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Comprehensive query validation
        
        Args:
            sql_query: The SQL query to validate
            schema_context: Optional schema context for enhanced validation
            
        Returns:
            Dict containing validation results and suggestions
        """
        results = {
            "success": True,
            "errors": [],
            "warnings": [],
            "suggestions": [],
            "query": sql_query.strip(),
            "is_safe": True,
            "estimated_complexity": "low"
        }
        
        # Run all validation checks
        self._check_security(sql_query, results)
        self._check_syntax(sql_query, results)
        self._check_table_names(sql_query, results)
        self._check_query_structure(sql_query, results)
        self._estimate_complexity(sql_query, results)
        
        # If schema context provided, do enhanced validation
        if schema_context:
            self._check_column_names(sql_query, schema_context, results)
        
        # Final success determination
        results["success"] = len(results["errors"]) == 0
        
        return results
    
    def _check_security(self, query: str, results: Dict) -> None:
        """
        Check for dangerous SQL operations
        """
        query_upper = query.upper()
        
        for dangerous_keyword in self.dangerous_keywords:
            if dangerous_keyword in query_upper:
                results["errors"].append(f"Dangerous operation detected: {dangerous_keyword}")
                results["is_safe"] = False
        
        # Check for SQL injection patterns
        injection_patterns = [
            r"'\s*OR\s+'\d+'\s*=\s*'\d+'",  # '1'='1'
            r";\s*DROP\s+TABLE",             # ; DROP TABLE
            r"UNION\s+SELECT.*--",           # UNION SELECT with comment
            r"'\s*;\s*--"                    # '; --
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                results["errors"].append(f"Potential SQL injection pattern detected")
                results["is_safe"] = False
    
    def _check_syntax(self, query: str, results: Dict) -> None:
        """
        Basic syntax validation
        """
        query_stripped = query.strip()
        
        # Check if query is empty
        if not query_stripped:
            results["errors"].append("Query is empty")
            return
        
        # Check for basic SELECT structure
        if query_stripped.upper().startswith('SELECT'):
            # Must have FROM clause for SELECT statements
            if 'FROM' not in query_stripped.upper():
                results["errors"].append("SELECT statement must have a FROM clause")
            
            # Check for balanced parentheses
            if query_stripped.count('(') != query_stripped.count(')'):
                results["errors"].append("Unbalanced parentheses in query")
            
            # Check for proper semicolon usage
            semicolon_count = query_stripped.count(';')
            if semicolon_count > 1:
                results["warnings"].append("Multiple semicolons detected - may indicate multiple statements")
            elif semicolon_count == 0:
                results["suggestions"].append("Consider adding semicolon at end of query")
        
        # Check for common syntax issues
        if query_stripped.endswith(','):
            results["errors"].append("Query ends with comma - likely syntax error")
        
        # Check for unclosed quotes
        single_quote_count = query_stripped.count("'")
        if single_quote_count % 2 != 0:
            results["errors"].append("Unclosed single quotes detected")
    
    def _check_table_names(self, query: str, results: Dict) -> None:
        """
        Validate table names against known schema
        """
        # Extract potential table names from query
        # Simple regex to find words after FROM and JOIN
        table_patterns = [
            r'FROM\s+(\w+)',
            r'JOIN\s+(\w+)',
            r'UPDATE\s+(\w+)',
            r'INTO\s+(\w+)'
        ]
        
        found_tables = set()
        for pattern in table_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            found_tables.update(table.lower() for table in matches)
        
        # Check if tables exist in our schema
        for table in found_tables:
            if table not in self.valid_tables:
                # Check for common aliases or variations
                if not self._is_likely_alias(table, query):
                    results["warnings"].append(f"Table '{table}' not found in schema. Available tables: {', '.join(sorted(self.valid_tables))}")
    
    def _check_query_structure(self, query: str, results: Dict) -> None:
        """
        Check query structure and suggest improvements
        """
        query_upper = query.upper()
        
        # Check for potential performance issues
        if 'SELECT *' in query_upper:
            results["suggestions"].append("Consider selecting specific columns instead of using SELECT *")
        
        # Check for LIMIT clause on potentially large queries
        if 'JOIN' in query_upper and 'LIMIT' not in query_upper:
            results["suggestions"].append("Consider adding LIMIT clause for JOIN queries to prevent large result sets")
        
        # Check for proper GROUP BY usage
        if 'GROUP BY' in query_upper:
            if 'SELECT' in query_upper:
                # Basic check for GROUP BY with aggregation functions
                has_aggregation = any(func in query_upper for func in ['COUNT(', 'SUM(', 'AVG(', 'MIN(', 'MAX('])
                if not has_aggregation:
                    results["warnings"].append("GROUP BY clause detected but no aggregation functions found")
        
        # Check for ORDER BY without LIMIT
        if 'ORDER BY' in query_upper and 'LIMIT' not in query_upper:
            results["suggestions"].append("Consider adding LIMIT when using ORDER BY for better performance")
    
    def _check_column_names(self, query: str, schema_context: Dict, results: Dict) -> None:
        """
        Enhanced validation using schema context
        """
        # This would require more sophisticated parsing
        # For now, just basic checks
        if 'key_fields' in schema_context:
            key_fields = schema_context['key_fields']
            query_lower = query.lower()
            
            # Suggest using key fields if none are mentioned
            mentioned_fields = [field for field in key_fields if field.lower() in query_lower]
            if not mentioned_fields and len(key_fields) > 0:
                results["suggestions"].append(f"Consider using key fields: {', '.join(key_fields)}")
    
    def _estimate_complexity(self, query: str, results: Dict) -> None:
        """
        Estimate query complexity
        """
        query_upper = query.upper()
        complexity_score = 0
        
        # Count complexity indicators
        complexity_indicators = {
            'JOIN': 2,
            'SUBQUERY': 3,  # Approximated by presence of nested SELECT
            'GROUP BY': 2,
            'HAVING': 2,
            'ORDER BY': 1,
            'UNION': 3,
            'CASE WHEN': 2,
            'EXISTS': 2,
            'IN (SELECT': 3
        }
        
        for indicator, score in complexity_indicators.items():
            if indicator in query_upper:
                complexity_score += score
        
        # Count nested levels (rough approximation)
        nested_selects = query_upper.count('SELECT') - 1
        complexity_score += nested_selects * 2
        
        # Determine complexity level
        if complexity_score <= 2:
            results["estimated_complexity"] = "low"
        elif complexity_score <= 6:
            results["estimated_complexity"] = "medium"
        else:
            results["estimated_complexity"] = "high"
            results["suggestions"].append("High complexity query - consider breaking into smaller parts")
    
    def _is_likely_alias(self, table_name: str, query: str) -> bool:
        """
        Check if a table name is likely an alias
        """
        # Simple check for common alias patterns
        alias_patterns = [
            rf'\b\w+\s+{table_name}\b',  # table_name alias_name
            rf'\b\w+\s+AS\s+{table_name}\b'  # table_name AS alias_name
        ]
        
        for pattern in alias_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def validate_with_database(self, query: str) -> Dict[str, Any]:
        """
        Validate query by attempting to explain it (without executing)
        """
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Use EXPLAIN to validate without executing
            explain_query = f"EXPLAIN {query}"
            cursor.execute(explain_query)
            
            # If we get here, the query is valid
            explain_result = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {
                "success": True,
                "database_valid": True,
                "message": "Query validated successfully against database",
                "execution_plan": [str(row) for row in explain_result]
            }
            
        except psycopg2.Error as e:
            return {
                "success": False,
                "database_valid": False,
                "error": str(e),
                "message": f"Database validation failed: {e}"
            }
        except Exception as e:
            return {
                "success": False,
                "database_valid": False, 
                "error": str(e),
                "message": f"Validation error: {e}"
            }
    
    def get_tool_spec(self) -> Dict[str, Any]:
        """
        Returns the tool specification for LangGraph
        """
        return {
            "name": "QueryValidatorTool",
            "description": "Validates SQL queries for syntax, security, and business logic before execution",
            "parameters": {
                "sql_query": {
                    "type": "string",
                    "description": "The SQL query to validate",
                    "required": True
                },
                "schema_context": {
                    "type": "object",
                    "description": "Optional schema context for enhanced validation",
                    "required": False
                },
                "database_validation": {
                    "type": "boolean",
                    "description": "Whether to validate against the actual database using EXPLAIN",
                    "required": False,
                    "default": False
                }
            }
        }

class BusinessLogicValidator:
    """
    Validates queries against business rules and logic
    """
    
    def __init__(self):
        self.name = "BusinessLogicValidator"
        self.description = "Validates queries against business rules and constraints"
    
    def validate_business_logic(self, query: str, intent: str = "") -> Dict[str, Any]:
        """
        Validate query against business logic rules
        
        Args:
            query: SQL query to validate
            intent: Natural language description of what the query should do
            
        Returns:
            Validation results with business logic feedback
        """
        results = {
            "success": True,
            "business_valid": True,
            "warnings": [],
            "suggestions": [],
            "intent": intent
        }
        
        query_upper = query.upper()
        
        # Business rule validations
        self._check_revenue_calculations(query_upper, results)
        self._check_date_ranges(query_upper, results)
        self._check_common_business_patterns(query_upper, intent.lower(), results)
        
        results["business_valid"] = len([w for w in results["warnings"] if "business" in w.lower()]) == 0
        
        return results
    
    def _check_revenue_calculations(self, query: str, results: Dict) -> None:
        """
        Check if revenue calculations are done correctly
        """
        if "QUANTITY" in query and "UNIT_PRICE" in query:
            # Check if discount is being considered
            if "DISCOUNT" not in query:
                results["warnings"].append("Revenue calculation detected but discount not considered - may lead to incorrect totals")
                results["suggestions"].append("Include discount in revenue calculation: quantity * unit_price * (1 - discount)")
        
        if "SUM(" in query and ("QUANTITY" in query or "UNIT_PRICE" in query):
            # Suggest proper revenue calculation
            if "(1 - DISCOUNT)" not in query:
                results["suggestions"].append("For accurate revenue: SUM(quantity * unit_price * (1 - discount))")
    
    def _check_date_ranges(self, query: str, results: Dict) -> None:
        """
        Check for appropriate date range handling
        """
        if "ORDER_DATE" in query or "SHIPPED_DATE" in query:
            # Check if year filtering is reasonable
            current_year_pattern = r"199[0-9]|200[0-9]|201[0-9]|202[0-9]"
            if not re.search(current_year_pattern, query):
                results["suggestions"].append("Consider adding appropriate date range filters for better performance")
        
        # Check for date comparisons without indexes
        if ">" in query and ("DATE" in query):
            results["suggestions"].append("Date range queries may benefit from proper indexing")
    
    def _check_common_business_patterns(self, query: str, intent: str, results: Dict) -> None:
        """
        Check against common business query patterns
        """
        # Customer analysis patterns
        if "customer" in intent and "ORDER" in query:
            if "COUNT(" not in query and "SUM(" not in query:
                results["suggestions"].append("Customer analysis usually benefits from aggregation (COUNT, SUM)")
        
        # Product performance patterns  
        if "product" in intent and "performance" in intent:
            if "ORDER_DETAILS" not in query:
                results["warnings"].append("Product performance analysis typically requires order_details table")
        
        # Revenue analysis patterns
        if "revenue" in intent or "sales" in intent:
            required_tables = ["ORDER_DETAILS"]
            if not any(table in query for table in required_tables):
                results["warnings"].append("Revenue analysis requires order_details table for accurate calculations")

# Tool instances
query_validator = QueryValidatorTool()
business_validator = BusinessLogicValidator()

# Tool registry for LangGraph
VALIDATION_TOOL_REGISTRY = {
    "QueryValidatorTool": query_validator,
    "BusinessLogicValidator": business_validator
}

def validate_complete_query(sql_query: str, schema_context: Optional[Dict] = None, 
                          intent: str = "", database_check: bool = False) -> Dict[str, Any]:
    """
    Complete validation using all available validators
    """
    # Syntax and security validation
    syntax_results = query_validator.validate_query(sql_query, schema_context)
    
    # Business logic validation
    business_results = business_validator.validate_business_logic(sql_query, intent)
    
    # Optional database validation
    db_results = None
    if database_check and syntax_results["success"]:
        db_results = query_validator.validate_with_database(sql_query)
    
    # Combine results
    combined_results = {
        "success": syntax_results["success"] and business_results["business_valid"],
        "syntax_validation": syntax_results,
        "business_validation": business_results,
        "database_validation": db_results,
        "overall_score": _calculate_validation_score(syntax_results, business_results, db_results),
        "recommendations": _generate_recommendations(syntax_results, business_results)
    }
    
    return combined_results

def _calculate_validation_score(syntax_results: Dict, business_results: Dict, db_results: Optional[Dict]) -> int:
    """Calculate overall validation score (0-100)"""
    score = 100
    
    # Deduct points for errors and warnings
    score -= len(syntax_results.get("errors", [])) * 20
    score -= len(syntax_results.get("warnings", [])) * 10
    score -= len(business_results.get("warnings", [])) * 5
    
    # Bonus for database validation success
    if db_results and db_results.get("database_valid", False):
        score += 10
    
    return max(0, min(100, score))

def _generate_recommendations(syntax_results: Dict, business_results: Dict) -> List[str]:
    """Generate top recommendations from all validations"""
    recommendations = []
    
    # Priority: Errors first, then warnings, then suggestions
    recommendations.extend(syntax_results.get("errors", []))
    recommendations.extend(syntax_results.get("warnings", []))
    recommendations.extend(business_results.get("warnings", []))
    recommendations.extend(syntax_results.get("suggestions", [])[:3])  # Top 3 suggestions
    
    return recommendations[:5]  # Return top 5 recommendations