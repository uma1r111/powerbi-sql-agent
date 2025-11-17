# config/error_config.py

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

class ErrorSeverity(str, Enum):
    """Error severity levels"""
    CRITICAL = "critical"  # Complete failure, no recovery
    HIGH = "high"          # Significant issue, may recover
    MEDIUM = "medium"      # Recoverable with guidance
    LOW = "low"            # Warning, can proceed
    INFO = "info"          # Informational only

class ErrorCategory(str, Enum):
    """Main error categories"""
    SYNTAX = "syntax"
    SCHEMA = "schema"
    SEMANTIC = "semantic"
    BUSINESS_LOGIC = "business_logic"
    DATA = "data"
    INFRASTRUCTURE = "infrastructure"
    SECURITY = "security"
    VALIDATION = "validation"

@dataclass
class ErrorType:
    """Definition of an error type"""
    code: str
    category: ErrorCategory
    severity: ErrorSeverity
    retryable: bool
    user_message_template: str
    recovery_strategy: str
    technical_pattern: Optional[str] = None  # Regex pattern to match error

# Complete error type definitions
ERROR_TYPES: Dict[str, ErrorType] = {
    
    # ========== SYNTAX ERRORS ==========
    "syntax_missing_from": ErrorType(
        code="SYN001",
        category=ErrorCategory.SYNTAX,
        severity=ErrorSeverity.HIGH,
        retryable=True,
        user_message_template="The query is missing a FROM clause to specify which table to query.",
        recovery_strategy="add_from_clause",
        technical_pattern=r"SELECT.*(?!FROM)"
    ),
    
    "syntax_unbalanced_parentheses": ErrorType(
        code="SYN002",
        category=ErrorCategory.SYNTAX,
        severity=ErrorSeverity.HIGH,
        retryable=True,
        user_message_template="The query has unmatched parentheses. This usually means a missing opening or closing bracket.",
        recovery_strategy="fix_parentheses"
    ),
    
    "syntax_unclosed_quotes": ErrorType(
        code="SYN003",
        category=ErrorCategory.SYNTAX,
        severity=ErrorSeverity.HIGH,
        retryable=True,
        user_message_template="The query has an unclosed quote mark. Every opening quote needs a closing quote.",
        recovery_strategy="fix_quotes"
    ),
    
    "syntax_trailing_comma": ErrorType(
        code="SYN004",
        category=ErrorCategory.SYNTAX,
        severity=ErrorSeverity.MEDIUM,
        retryable=True,
        user_message_template="The query ends with a comma, which isn't valid SQL syntax.",
        recovery_strategy="remove_trailing_comma"
    ),
    
    "syntax_invalid_operator": ErrorType(
        code="SYN005",
        category=ErrorCategory.SYNTAX,
        severity=ErrorSeverity.MEDIUM,
        retryable=True,
        user_message_template="The query uses an invalid SQL operator or comparison.",
        recovery_strategy="fix_operator"
    ),
    
    # ========== SCHEMA ERRORS ==========
    "schema_table_not_found": ErrorType(
        code="SCH001",
        category=ErrorCategory.SCHEMA,
        severity=ErrorSeverity.HIGH,
        retryable=True,
        user_message_template="I couldn't find a table called '{table_name}' in the database.",
        recovery_strategy="suggest_similar_tables",
        technical_pattern=r"(relation|table) ['\"]?(\w+)['\"]? does not exist"
    ),
    
    "schema_column_not_found": ErrorType(
        code="SCH002",
        category=ErrorCategory.SCHEMA,
        severity=ErrorSeverity.HIGH,
        retryable=True,
        user_message_template="The column '{column_name}' doesn't exist in the {table_name} table.",
        recovery_strategy="suggest_similar_columns",
        technical_pattern=r"column ['\"]?(\w+)['\"]? does not exist"
    ),
    
    "schema_ambiguous_column": ErrorType(
        code="SCH003",
        category=ErrorCategory.SCHEMA,
        severity=ErrorSeverity.MEDIUM,
        retryable=True,
        user_message_template="The column name '{column_name}' is ambiguous - it exists in multiple tables.",
        recovery_strategy="qualify_column_name"
    ),
    
    "schema_invalid_join": ErrorType(
        code="SCH004",
        category=ErrorCategory.SCHEMA,
        severity=ErrorSeverity.HIGH,
        retryable=True,
        user_message_template="The tables can't be joined in the way specified.",
        recovery_strategy="fix_join_relationship"
    ),
    
    # ========== SEMANTIC ERRORS ==========
    "semantic_type_mismatch": ErrorType(
        code="SEM001",
        category=ErrorCategory.SEMANTIC,
        severity=ErrorSeverity.MEDIUM,
        retryable=True,
        user_message_template="Trying to compare values of incompatible types (like comparing text to a number).",
        recovery_strategy="fix_type_comparison"
    ),
    
    "semantic_aggregation_error": ErrorType(
        code="SEM002",
        category=ErrorCategory.SEMANTIC,
        severity=ErrorSeverity.MEDIUM,
        retryable=True,
        user_message_template="The aggregation function (SUM, COUNT, etc.) is used incorrectly.",
        recovery_strategy="fix_aggregation"
    ),
    
    "semantic_group_by_missing": ErrorType(
        code="SEM003",
        category=ErrorCategory.SEMANTIC,
        severity=ErrorSeverity.MEDIUM,
        retryable=True,
        user_message_template="Using aggregation requires a GROUP BY clause for non-aggregated columns.",
        recovery_strategy="add_group_by"
    ),
    
    # ========== BUSINESS LOGIC ERRORS ==========
    "business_missing_discount": ErrorType(
        code="BUS001",
        category=ErrorCategory.BUSINESS_LOGIC,
        severity=ErrorSeverity.LOW,
        retryable=True,
        user_message_template="Revenue calculations should include the discount factor to be accurate.",
        recovery_strategy="add_discount_calculation"
    ),
    
    "business_wrong_table": ErrorType(
        code="BUS002",
        category=ErrorCategory.BUSINESS_LOGIC,
        severity=ErrorSeverity.MEDIUM,
        retryable=True,
        user_message_template="This type of analysis typically requires the {required_table} table.",
        recovery_strategy="use_correct_table"
    ),
    
    "business_missing_required_column": ErrorType(
        code="BUS003",
        category=ErrorCategory.BUSINESS_LOGIC,
        severity=ErrorSeverity.MEDIUM,
        retryable=True,
        user_message_template="This calculation is missing the {column_name} column which is required for accuracy.",
        recovery_strategy="add_required_column"
    ),
    
    # ========== DATA ERRORS ==========
    "data_empty_result": ErrorType(
        code="DAT001",
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.LOW,
        retryable=False,
        user_message_template="The query executed successfully but found no matching data.",
        recovery_strategy="suggest_alternatives"
    ),
    
    "data_null_values": ErrorType(
        code="DAT002",
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.LOW,
        retryable=False,
        user_message_template="Some of the data contains NULL values which might affect the results.",
        recovery_strategy="handle_nulls"
    ),
    
    "data_constraint_violation": ErrorType(
        code="DAT003",
        category=ErrorCategory.DATA,
        severity=ErrorSeverity.MEDIUM,
        retryable=True,
        user_message_template="The operation violates a data constraint (like trying to insert duplicate values).",
        recovery_strategy="check_constraints"
    ),
    
    # ========== INFRASTRUCTURE ERRORS ==========
    "infra_connection_failed": ErrorType(
        code="INF001",
        category=ErrorCategory.INFRASTRUCTURE,
        severity=ErrorSeverity.CRITICAL,
        retryable=False,
        user_message_template="I couldn't connect to the database. The database might be down or unreachable.",
        recovery_strategy="check_connection",
        technical_pattern=r"(connection|connect) (refused|failed|timeout)"
    ),
    
    "infra_timeout": ErrorType(
        code="INF002",
        category=ErrorCategory.INFRASTRUCTURE,
        severity=ErrorSeverity.HIGH,
        retryable=True,
        user_message_template="The query took too long to execute and timed out.",
        recovery_strategy="simplify_query",
        technical_pattern=r"timeout|timed out"
    ),
    
    "infra_permission_denied": ErrorType(
        code="INF003",
        category=ErrorCategory.INFRASTRUCTURE,
        severity=ErrorSeverity.CRITICAL,
        retryable=False,
        user_message_template="I don't have permission to access this data.",
        recovery_strategy="request_permissions",
        technical_pattern=r"permission denied|access denied|unauthorized"
    ),
    
    # ========== SECURITY ERRORS ==========
    "security_sql_injection": ErrorType(
        code="SEC001",
        category=ErrorCategory.SECURITY,
        severity=ErrorSeverity.CRITICAL,
        retryable=False,
        user_message_template="This query contains patterns that look like a security threat and cannot be executed.",
        recovery_strategy="reject_query"
    ),
    
    "security_dangerous_operation": ErrorType(
        code="SEC002",
        category=ErrorCategory.SECURITY,
        severity=ErrorSeverity.CRITICAL,
        retryable=False,
        user_message_template="This operation could modify or delete data, which isn't allowed.",
        recovery_strategy="reject_query"
    ),
    
    # ========== VALIDATION ERRORS ==========
    "validation_failed": ErrorType(
        code="VAL001",
        category=ErrorCategory.VALIDATION,
        severity=ErrorSeverity.MEDIUM,
        retryable=True,
        user_message_template="The query didn't pass validation checks.",
        recovery_strategy="review_and_retry"
    ),
}

# Recovery strategy definitions
RECOVERY_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "add_from_clause": {
        "description": "Add missing FROM clause",
        "llm_guidance": "Add a FROM clause specifying the table name based on the user's query intent."
    },
    "fix_parentheses": {
        "description": "Balance parentheses",
        "llm_guidance": "Review the query and ensure every opening parenthesis has a matching closing one."
    },
    "suggest_similar_tables": {
        "description": "Suggest similar table names",
        "llm_guidance": "Find tables with similar names and suggest the closest match to the user."
    },
    "suggest_similar_columns": {
        "description": "Suggest similar column names",
        "llm_guidance": "List available columns in the table and suggest which one the user likely meant."
    },
    "add_discount_calculation": {
        "description": "Include discount in revenue calculation",
        "llm_guidance": "Update the revenue calculation to: quantity * unit_price * (1 - discount)"
    },
    "simplify_query": {
        "description": "Create a simpler version of the query",
        "llm_guidance": "Break down the complex query into a simpler version with fewer JOINs or conditions."
    },
    "reject_query": {
        "description": "Reject query for security reasons",
        "llm_guidance": "Do not execute this query. Explain to the user why it cannot be run."
    },
}