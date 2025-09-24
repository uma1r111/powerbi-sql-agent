import logging
from typing import Literal, Dict, Any

# Import our state management  
from state.agent_state import AgentState
from flow.graph import GraphState, graph_state_to_agent_state

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def should_continue_to_planner(state: GraphState) -> Literal["continue", "error"]:
    """
    Determine if processing should continue to the next node or stop due to errors
    
    Used after schema_inspector and planner nodes
    
    Args:
        state: Current GraphState
        
    Returns:
        "continue" if processing should continue, "error" if it should stop
    """
    agent_state = graph_state_to_agent_state(state)
    logger.info("Evaluating continuation condition")
    
    # Check for critical errors that should stop processing
    if len(agent_state.errors) > 0:
        logger.warning(f"Found {len(agent_state.errors)} errors, stopping processing")
        return "error"
    
    # Check if we have required information
    if not agent_state.selected_tables:
        logger.warning("No tables selected, cannot continue")
        agent_state.add_error("No relevant tables identified for the query")
        return "error"
    
    # Check if we're in a retry loop
    if agent_state.correction_attempts >= agent_state.max_correction_attempts:
        logger.warning(f"Max correction attempts ({agent_state.max_correction_attempts}) reached")
        agent_state.add_error("Maximum correction attempts exceeded")
        return "error"
    
    logger.info("Conditions met, continuing processing")
    return "continue"

def should_execute_query(state: GraphState) -> Literal["execute", "retry", "error"]:
    """
    Determine if the validated query should be executed, retried, or stopped
    
    Used after query_validator node
    
    Args:
        state: Current GraphState
        
    Returns:
        "execute" if query should be executed
        "retry" if query needs regeneration  
        "error" if processing should stop
    """
    agent_state = graph_state_to_agent_state(state)
    logger.info("Evaluating query execution condition")
    
    # Check if validation passed
    if agent_state.validation_passed:
        logger.info("Validation passed, proceeding to execution")
        return "execute"
    
    # Check if we can retry
    if agent_state.correction_attempts < agent_state.max_correction_attempts:
        logger.info(f"Validation failed, retrying (attempt {agent_state.correction_attempts + 1}/{agent_state.max_correction_attempts})")
        
        # Determine if the error is retryable
        validation_results = agent_state.validation_results
        
        if validation_results and "syntax_validation" in validation_results:
            syntax_errors = validation_results["syntax_validation"].get("errors", [])
            
            # Check for non-retryable errors
            non_retryable_patterns = [
                "dangerous operation",
                "sql injection",
                "unauthorized",
                "permission denied"
            ]
            
            for error in syntax_errors:
                if any(pattern in error.lower() for pattern in non_retryable_patterns):
                    logger.error(f"Non-retryable error detected: {error}")
                    agent_state.add_error(f"Security violation: {error}")
                    return "error"
        
        # Mark that we need correction and retry
        agent_state.needs_correction = True
        return "retry"
    
    # Max attempts reached
    logger.error("Max correction attempts reached, stopping")
    agent_state.add_error("Query validation failed after maximum retry attempts")
    return "error"

def should_retry_query(state: GraphState) -> Literal["success", "retry", "error"]:
    """
    Determine the next step after query execution
    
    Used after sql_executor node
    
    Args:
        state: Current GraphState
        
    Returns:
        "success" if execution succeeded and should proceed to output
        "retry" if execution failed but should retry
        "error" if execution failed and should stop
    """
    agent_state = graph_state_to_agent_state(state)
    logger.info("Evaluating post-execution condition")
    
    # Check if execution was successful
    if agent_state.execution_successful:
        logger.info("Query execution successful, proceeding to output formatting")
        return "success"
    
    # Execution failed - check if we can retry
    if agent_state.correction_attempts < agent_state.max_correction_attempts:
        # Analyze the type of error to determine if retry is worthwhile
        execution_results = agent_state.execution_results
        
        if execution_results and "error" in execution_results:
            error_message = execution_results["error"].lower()
            
            # Categorize errors
            syntax_errors = [
                "syntax error", "invalid sql", "parse error", "column does not exist",
                "table does not exist", "relation does not exist"
            ]
            
            permission_errors = [
                "permission denied", "access denied", "unauthorized", "forbidden"
            ]
            
            connection_errors = [
                "connection", "timeout", "network", "server"
            ]
            
            # Check error category
            if any(pattern in error_message for pattern in permission_errors):
                logger.error(f"Permission error, not retrying: {error_message}")
                agent_state.add_error("Database access permission error")
                return "error"
            
            elif any(pattern in error_message for pattern in connection_errors):
                logger.error(f"Connection error, not retrying: {error_message}")
                agent_state.add_error("Database connection error")
                return "error"
            
            elif any(pattern in error_message for pattern in syntax_errors):
                logger.info(f"Syntax error detected, will retry: {error_message}")
                agent_state.needs_correction = True
                return "retry"
            
            else:
                # Unknown error type - try once more
                logger.warning(f"Unknown error type, will retry once: {error_message}")
                agent_state.needs_correction = True
                return "retry"
    
    # Max attempts reached or non-retryable error
    logger.error("Query execution failed, cannot retry")
    return "error"

def should_continue_processing(state: AgentState) -> bool:
    """
    General utility function to check if processing should continue
    
    Args:
        state: Current agent state
        
    Returns:
        True if processing should continue, False otherwise
    """
    # Check basic continuation conditions
    if state.processing_complete:
        return False
    
    if state.correction_attempts >= state.max_correction_attempts:
        return False
    
    if state.requires_user_input:
        return False
    
    # Check if we have critical errors
    if state.errors:
        critical_error_patterns = [
            "security violation",
            "permission denied", 
            "database connection",
            "maximum.*exceeded"
        ]
        
        for error in state.errors:
            if any(pattern in error.lower() for pattern in critical_error_patterns):
                return False
    
    return True

def get_next_node_context(state: AgentState, current_node: str) -> Dict[str, Any]:
    """
    Get context information for the next node
    
    Args:
        state: Current agent state
        current_node: Name of the current node
        
    Returns:
        Context dictionary for the next node
    """
    context = {
        "previous_node": current_node,
        "state_summary": state.to_dict(),
        "should_continue": should_continue_processing(state),
        "retry_info": {
            "attempts": state.correction_attempts,
            "max_attempts": state.max_correction_attempts,
            "needs_correction": state.needs_correction
        }
    }
    
    # Add current plan context if available
    current_plan = getattr(state, '_current_plan', None)
    if current_plan:
        context["plan_info"] = {
            "current_step": current_plan.get_current_step().description if current_plan.get_current_step() else "No current step",
            "progress": current_plan.calculate_progress(),
            "plan_status": current_plan.status
        }
    
    return context

def log_transition(state: AgentState, from_node: str, to_node: str, condition: str):
    """
    Log node transitions for debugging and monitoring
    
    Args:
        state: Current agent state
        from_node: Source node name
        to_node: Destination node name  
        condition: Condition that triggered the transition
    """
    logger.info(f"Node transition: {from_node} -> {to_node} (condition: {condition})")
    
    # Log state summary
    logger.debug(f"State summary: Query='{state.user_query[:50]}...', "
                f"Tables={state.selected_tables}, "
                f"Errors={len(state.errors)}, "
                f"Attempts={state.correction_attempts}")
    
    # Log any warnings or errors
    if state.warnings:
        logger.warning(f"Active warnings: {state.warnings[-3:]}")  # Last 3 warnings
    
    if state.errors:
        logger.error(f"Active errors: {state.errors[-3:]}")  # Last 3 errors

def handle_error_recovery(state: AgentState, error_node: str) -> Dict[str, Any]:
    """
    Handle error recovery strategies
    
    Args:
        state: Current agent state
        error_node: Node where the error occurred
        
    Returns:
        Recovery strategy information
    """
    logger.info(f"Handling error recovery for node: {error_node}")
    
    recovery_strategy = {
        "can_recover": False,
        "recovery_action": "stop",
        "recovery_node": None,
        "user_message": "I encountered an error processing your request."
    }
    
    # Determine recovery based on error type and node
    if error_node == "schema_inspector":
        if state.correction_attempts == 0:
            recovery_strategy.update({
                "can_recover": True,
                "recovery_action": "retry_with_fallback",
                "recovery_node": "schema_inspector",
                "user_message": "Let me try a different approach to understand your query."
            })
    
    elif error_node == "planner":
        if state.selected_tables and state.correction_attempts < 2:
            recovery_strategy.update({
                "can_recover": True,
                "recovery_action": "simplify_query",
                "recovery_node": "planner",
                "user_message": "Let me try generating a simpler query."
            })
    
    elif error_node == "query_validator":
        if state.correction_attempts < 2:
            recovery_strategy.update({
                "can_recover": True,
                "recovery_action": "regenerate_query",
                "recovery_node": "planner",
                "user_message": "Let me regenerate the query with better validation in mind."
            })
    
    return recovery_strategy

# Export main functions
__all__ = [
    "should_continue_to_planner",
    "should_execute_query", 
    "should_retry_query",
    "should_continue_processing",
    "get_next_node_context",
    "log_transition",
    "handle_error_recovery"
]