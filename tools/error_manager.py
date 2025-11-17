# tools/error_manager.py

import logging
import re
import json
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from difflib import get_close_matches

from config.error_config import (
    ErrorType, ERROR_TYPES, ErrorCategory, ErrorSeverity,
    RECOVERY_STRATEGIES
)

logger = logging.getLogger(__name__)

class ErrorDetail:
    """Detailed error information"""
    
    def __init__(
        self,
        error_type: ErrorType,
        raw_error: str,
        context: Dict[str, Any] = None,
        extracted_entities: Dict[str, str] = None
    ):
        self.error_type = error_type
        self.raw_error = raw_error
        self.context = context or {}
        self.extracted_entities = extracted_entities or {}
        self.timestamp = datetime.now()
        self.resolved = False
        self.resolution_attempt = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "error_code": self.error_type.code,
            "category": self.error_type.category,
            "severity": self.error_type.severity,
            "retryable": self.error_type.retryable,
            "raw_error": self.raw_error,
            "context": self.context,
            "extracted_entities": self.extracted_entities,
            "timestamp": self.timestamp.isoformat(),
            "resolved": self.resolved,
            "resolution_attempt": self.resolution_attempt
        }
    
    def get_user_message(self) -> str:
        """Get user-friendly error message"""
        message = self.error_type.user_message_template
        
        # Replace placeholders with extracted entities
        for key, value in self.extracted_entities.items():
            placeholder = f"{{{key}}}"
            if placeholder in message:
                message = message.replace(placeholder, str(value))
        
        return message
    
    def get_recovery_strategy(self) -> Dict[str, Any]:
        """Get recovery strategy details"""
        strategy_name = self.error_type.recovery_strategy
        return RECOVERY_STRATEGIES.get(strategy_name, {})

class ErrorManager:
    """Central error management system"""
    
    def __init__(self):
        self.error_log: List[ErrorDetail] = []
        self.session_error_counts: Dict[str, int] = {}
    
    def classify_error(
        self,
        error_message: str,
        error_context: Dict[str, Any] = None
    ) -> ErrorDetail:
        """
        Classify an error and return detailed error information
        
        Args:
            error_message: The raw error message
            error_context: Additional context (query, table, etc.)
            
        Returns:
            ErrorDetail object with classification and guidance
        """
        error_context = error_context or {}
        
        # Try to match error using patterns
        for error_key, error_type in ERROR_TYPES.items():
            if error_type.technical_pattern:
                match = re.search(error_type.technical_pattern, error_message, re.IGNORECASE)
                if match:
                    # Extract entities from regex groups
                    extracted_entities = self._extract_entities_from_match(match, error_type)
                    
                    error_detail = ErrorDetail(
                        error_type=error_type,
                        raw_error=error_message,
                        context=error_context,
                        extracted_entities=extracted_entities
                    )
                    
                    self._log_error(error_detail)
                    return error_detail
        
        # Fallback: generic validation error
        error_type = ERROR_TYPES["validation_failed"]
        error_detail = ErrorDetail(
            error_type=error_type,
            raw_error=error_message,
            context=error_context
        )
        
        self._log_error(error_detail)
        return error_detail
    
    def _extract_entities_from_match(self, match: re.Match, error_type: ErrorType) -> Dict[str, str]:
        """Extract entities like table names, column names from regex match"""
        entities = {}
        
        if error_type.code == "SCH001":  # Table not found
            if match.groups():
                entities["table_name"] = match.group(2) if len(match.groups()) >= 2 else match.group(1)
        
        elif error_type.code == "SCH002":  # Column not found
            if match.groups():
                entities["column_name"] = match.group(1)
        
        return entities
    
    def suggest_alternatives(
        self,
        error_detail: ErrorDetail,
        available_options: List[str]
    ) -> List[str]:
        """
        Suggest alternatives based on error type
        
        Args:
            error_detail: The classified error
            available_options: List of valid options (tables, columns, etc.)
            
        Returns:
            List of suggested alternatives
        """
        if error_detail.error_type.code in ["SCH001", "SCH002"]:  # Schema errors
            # Extract the wrong name
            wrong_name = error_detail.extracted_entities.get(
                "table_name" or "column_name", ""
            )
            
            if wrong_name and available_options:
                # Find close matches
                suggestions = get_close_matches(
                    wrong_name,
                    available_options,
                    n=3,
                    cutoff=0.6
                )
                return suggestions
        
        return []
    
    def format_user_friendly_message(
        self,
        error_detail: ErrorDetail,
        suggestions: List[str] = None
    ) -> str:
        """
        Format a complete user-friendly error message
        
        Args:
            error_detail: The classified error
            suggestions: Optional list of suggestions
            
        Returns:
            Formatted error message for the user
        """
        # Start with the base message
        message = f"🤔 {error_detail.get_user_message()}\n\n"
        
        # Add suggestions if available
        if suggestions:
            if error_detail.error_type.code == "SCH001":  # Table not found
                message += "**Did you mean one of these tables?**\n"
                for suggestion in suggestions:
                    message += f"• {suggestion}\n"
            
            elif error_detail.error_type.code == "SCH002":  # Column not found
                message += "**Available columns in this table:**\n"
                for suggestion in suggestions:
                    message += f"• {suggestion}\n"
        
        # Add recovery guidance
        if error_detail.error_type.retryable:
            message += "\n💡 I can try again with the correct information. "
            if suggestions:
                message += "Which one should I use?"
        else:
            message += "\n⚠️ This error cannot be automatically fixed."
        
        return message
    
    def should_retry(self, error_detail: ErrorDetail, current_attempts: int, max_attempts: int) -> bool:
        """Determine if error should trigger a retry"""
        if not error_detail.error_type.retryable:
            return False
        
        if current_attempts >= max_attempts:
            return False
        
        # Don't retry infrastructure errors multiple times
        if error_detail.error_type.category == ErrorCategory.INFRASTRUCTURE:
            return current_attempts < 1
        
        return True
    
    def get_recovery_guidance_for_llm(self, error_detail: ErrorDetail) -> str:
        """
        Get detailed guidance for LLM to fix the error
        
        Args:
            error_detail: The classified error
            
        Returns:
            Detailed guidance string for LLM prompt
        """
        strategy = error_detail.get_recovery_strategy()
        
        guidance = f"""
PREVIOUS QUERY FAILED - ERROR ANALYSIS:

Error Type: {error_detail.error_type.code} - {error_detail.error_type.category}
Problem: {error_detail.get_user_message()}

Recovery Strategy: {strategy.get('description', 'General retry')}

Specific Guidance:
{strategy.get('llm_guidance', 'Review the error and generate a corrected query.')}

Context:
- Original Error: {error_detail.raw_error}
- Extracted Entities: {error_detail.extracted_entities}
- Query Context: {error_detail.context}

Generate a corrected query that addresses this specific issue.
"""
        
        return guidance
    
    def _log_error(self, error_detail: ErrorDetail):
        """Log error to structured log"""
        self.error_log.append(error_detail)
        
        # Structured logging
        log_entry = error_detail.to_dict()
        logger.error(f"Classified Error: {json.dumps(log_entry, indent=2)}")
        
        # Track session counts
        error_code = error_detail.error_type.code
        self.session_error_counts[error_code] = self.session_error_counts.get(error_code, 0) + 1
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics for this session"""
        total_errors = len(self.error_log)
        
        by_category = {}
        by_severity = {}
        retryable_count = 0
        resolved_count = 0
        
        for error in self.error_log:
            # Count by category
            category = error.error_type.category
            by_category[category] = by_category.get(category, 0) + 1
            
            # Count by severity
            severity = error.error_type.severity
            by_severity[severity] = by_severity.get(severity, 0) + 1
            
            # Count retryable and resolved
            if error.error_type.retryable:
                retryable_count += 1
            if error.resolved:
                resolved_count += 1
        
        return {
            "total_errors": total_errors,
            "by_category": by_category,
            "by_severity": by_severity,
            "retryable": retryable_count,
            "resolved": resolved_count,
            "resolution_rate": (resolved_count / total_errors * 100) if total_errors > 0 else 0
        }

# Global error manager instance
error_manager = ErrorManager()