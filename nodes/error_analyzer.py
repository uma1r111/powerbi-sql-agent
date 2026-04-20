# nodes/error_analyzer.py
"""
Error Analyzer Node - Fetches historical error context from MCP
"""

import logging
from state.agent_state import AgentState
from client.mcp_client import mcp_client

logger = logging.getLogger(__name__)

class ErrorAnalyzerNode:
    """
    Analyzes error history when retries are needed.
    Fetches aggregated error context from MCP to prevent retry loops.
    """
    
    def __init__(self):
        self.node_name = "error_analyzer"
    
    def __call__(self, state: AgentState) -> AgentState:
        """
        Fetch historical error context if execution failed.
        
        Args:
            state: Current agent state
        
        Returns:
            Updated state with error history
        """
        # Only run if we have a failure
        if state.execution_successful:
            logger.info("✅ Query successful, skipping error analysis")
            return state
        
        logger.info("🔍 Analyzing error history...")
        
        try:
            # Fetch session-scoped error log from MCP
            session_id = state.session_id
            
            # Use the dynamic URI we created in the server!
            resource_uri = f"db://logs/errors/{session_id}"
            
            error_history = mcp_client.read_resource(
                resource_uri,
                use_cache=False  # Don't cache error logs (they change)
            )
            
            # Store in state for Planner to use
            state.error_history = error_history
            
            # Pattern detection
            if self._is_stuck_in_loop(error_history):
                logger.warning("⚠️ Stuck in retry loop detected!")
                state.escalation_mode = True
                state.add_warning("Multiple similar errors detected. Consider schema review.")
            
            logger.info(f"✅ Error history loaded ({len(error_history)} chars)")
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch error history: {e}")
            state.error_history = ""
        
        return state
    
    def _is_stuck_in_loop(self, error_history: str) -> bool:
        """
        Detect if we're stuck in a retry loop.
        
        Args:
            error_history: Raw error log from MCP
        
        Returns:
            True if stuck in loop (3+ similar errors)
        """
        if not error_history or "Error #1" not in error_history:
            return False
        
        # Simple heuristic: 3+ errors in history = likely loop
        error_count = error_history.count("Error #")
        return error_count >= 3

# Create instance
error_analyzer_node = ErrorAnalyzerNode()