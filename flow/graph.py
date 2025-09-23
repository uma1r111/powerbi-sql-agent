# flow/graph.py

import logging
from typing import Dict, Any, Literal
from datetime import datetime

# LangGraph imports
from langgraph.graph import Graph, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# Import our state management
from state.agent_state import AgentState, StateManager
from state.plan_state import PlanManager, StepStatus

# Import our nodes
from nodes.schema_inspector import schema_inspector_node
from nodes.planner import planner_node

# Import tools for remaining nodes
from tools.sql_tools import sql_executor
from tools.validation_tools import query_validator, validate_complete_query

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryValidatorNode:
    """Node for validating SQL queries before execution"""
    
    def __init__(self):
        self.node_name = "query_validator"
    
    def __call__(self, state: AgentState) -> AgentState:
        logger.info("Validating SQL query")
        
        try:
            # Get current plan step
            current_plan = getattr(state, '_current_plan', None)
            if current_plan:
                current_step = current_plan.get_current_step()
                if current_step:
                    current_step.start_execution()
            
            # Validate the query
            validation_result = validate_complete_query(
                state.cleaned_sql,
                schema_context=state.schema_context,
                intent=state.business_intent,
                database_check=True
            )
            
            # Update state with validation results
            state.validation_results = validation_result
            state.validation_passed = validation_result["success"]
            
            # Add any warnings or errors
            if not validation_result["success"]:
                state.add_error("Query validation failed")
                state.needs_correction = True
                state.correction_attempts += 1
            
            # Complete current step
            if current_plan and current_step:
                current_step.complete_execution({
                    "validation_passed": state.validation_passed,
                    "validation_score": validation_result.get("overall_score", 0)
                })
            
            return state
            
        except Exception as e:
            error_msg = f"Query validation error: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            
            if current_plan and current_step:
                current_step.fail_execution(error_msg)
            
            return state

class SQLExecutorNode:
    """Node for executing validated SQL queries"""
    
    def __init__(self):
        self.node_name = "sql_executor"
    
    def __call__(self, state: AgentState) -> AgentState:
        logger.info("Executing SQL query")
        
        try:
            # Get current plan step
            current_plan = getattr(state, '_current_plan', None)
            if current_plan:
                current_step = current_plan.get_current_step()
                if current_step:
                    current_step.start_execution()
            
            # Execute the query
            start_time = datetime.now()
            execution_result = sql_executor.execute_query(state.cleaned_sql)
            end_time = datetime.now()
            
            # Calculate execution time
            state.execution_time = (end_time - start_time).total_seconds()
            
            # Update state with execution results
            state.execution_results = execution_result
            state.execution_successful = execution_result["success"]
            
            if execution_result["success"]:
                state.result_count = execution_result.get("row_count", 0)
                logger.info(f"Query executed successfully. Returned {state.result_count} rows")
            else:
                state.add_error(f"Query execution failed: {execution_result.get('error', 'Unknown error')}")
                state.needs_correction = True
                state.correction_attempts += 1
            
            # Complete current step
            if current_plan and current_step:
                current_step.complete_execution({
                    "execution_successful": state.execution_successful,
                    "result_count": state.result_count,
                    "execution_time": state.execution_time
                })
            
            return state
            
        except Exception as e:
            error_msg = f"Query execution error: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            
            if current_plan and current_step:
                current_step.fail_execution(error_msg)
            
            return state

class OutputFormatterNode:
    """Node for formatting and presenting results"""
    
    def __init__(self):
        self.node_name = "output_formatter"
    
    def __call__(self, state: AgentState) -> AgentState:
        logger.info("Formatting output")
        
        try:
            # Get current plan step
            current_plan = getattr(state, '_current_plan', None)
            if current_plan:
                current_step = current_plan.get_current_step()
                if current_step:
                    current_step.start_execution()
            
            # Format the response based on execution results
            if state.execution_successful:
                # Format successful results
                formatted_output = self._format_successful_results(state)
                response_message = f"Query executed successfully!\n\n{formatted_output}"
            else:
                # Format error response
                response_message = self._format_error_response(state)
            
            # Add AI response to conversation history
            state.add_ai_message(response_message)
            
            # Mark processing as complete
            state.processing_complete = True
            
            # Complete execution plan
            if current_plan:
                current_plan.complete_execution()
                if current_step:
                    current_step.complete_execution({"formatted_output": True})
            
            logger.info("Output formatting completed")
            return state
            
        except Exception as e:
            error_msg = f"Output formatting error: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            
            if current_plan and current_step:
                current_step.fail_execution(error_msg)
            
            return state
    
    def _format_successful_results(self, state: AgentState) -> str:
        """Format successful query results for display"""
        results = state.execution_results
        
        if not results.get("data"):
            return "Query executed successfully but returned no results."
        
        # Use the SQL executor's formatting method
        formatted_display = sql_executor.format_results_for_display(results)
        
        # Add some context
        context_info = f"""
SQL Query Used:
{state.cleaned_sql}

Execution Time: {state.execution_time:.2f} seconds
Tables Involved: {', '.join(state.selected_tables)}
"""
        
        return formatted_display + "\n" + context_info
    
    def _format_error_response(self, state: AgentState) -> str:
        """Format error response"""
        error_msg = "I encountered an issue processing your query:\n\n"
        
        # Add specific error details
        if state.errors:
            error_msg += f"Error: {state.errors[-1]}\n\n"
        
        # Add the attempted SQL for reference
        if state.cleaned_sql:
            error_msg += f"Attempted SQL:\n{state.cleaned_sql}\n\n"
        
        # Add suggestions if available
        if state.validation_results and "recommendations" in state.validation_results:
            recommendations = state.validation_results["recommendations"][:3]
            if recommendations:
                error_msg += "Suggestions:\n"
                for i, rec in enumerate(recommendations, 1):
                    error_msg += f"{i}. {rec}\n"
        
        return error_msg

class PowerBISQLAgent:
    """Main agent class that orchestrates the LangGraph workflow"""
    
    def __init__(self):
        self.graph = None
        self.memory = MemorySaver()
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        logger.info("Building LangGraph workflow")
        
        # Create node instances
        query_validator_node = QueryValidatorNode()
        sql_executor_node = SQLExecutorNode()
        output_formatter_node = OutputFormatterNode()
        
        # Define the workflow
        workflow = StateGraph(AgentState)
        
        # Add nodes to the graph
        workflow.add_node("schema_inspector", self._wrap_node(schema_inspector_node.execute))
        workflow.add_node("planner", self._wrap_node(planner_node.execute))
        workflow.add_node("query_validator", query_validator_node)
        workflow.add_node("sql_executor", sql_executor_node)
        workflow.add_node("output_formatter", output_formatter_node)
        
        # Define the flow (edges will be handled in edges.py)
        from flow.edge import should_continue_to_planner, should_retry_query, should_execute_query
        
        # Set entry point
        workflow.set_entry_point("schema_inspector")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "schema_inspector",
            should_continue_to_planner,
            {
                "continue": "planner",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "planner", 
            should_continue_to_planner,
            {
                "continue": "query_validator",
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "query_validator",
            should_execute_query,
            {
                "execute": "sql_executor",
                "retry": "planner", 
                "error": END
            }
        )
        
        workflow.add_conditional_edges(
            "sql_executor",
            should_retry_query,
            {
                "success": "output_formatter",
                "retry": "planner",
                "error": END
            }
        )
        
        workflow.add_edge("output_formatter", END)
        
        # Compile the graph
        self.graph = workflow.compile(checkpointer=self.memory)
        logger.info("LangGraph workflow built successfully")
    
    def _wrap_node(self, node_func):
        """Wrap node functions to handle state properly"""
        def wrapper(state: AgentState) -> AgentState:
            return node_func(state)
        return wrapper
    
    async def process_query(self, user_query: str, session_id: str = None) -> AgentState:
        """
        Process a user query through the complete workflow
        
        Args:
            user_query: The user's natural language query
            session_id: Optional session ID for conversation continuity
            
        Returns:
            Final state after processing
        """
        logger.info(f"Processing query: {user_query}")
        
        # Create initial state
        initial_state = StateManager.create_initial_state(user_query)
        
        if session_id:
            initial_state.session_id = session_id
        
        # Create execution plan
        plan = PlanManager.create_basic_sql_plan(user_query, [])  # Tables will be selected by schema inspector
        initial_state._current_plan = plan
        
        # Configure for graph execution
        config = {
            "configurable": {
                "thread_id": session_id or initial_state.session_id
            }
        }
        
        try:
            # Execute the workflow
            final_state = await self.graph.ainvoke(initial_state, config)
            
            logger.info(f"Query processing completed. Success: {final_state.processing_complete}")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            initial_state.add_error(f"Workflow execution failed: {e}")
            return initial_state
    
    def process_query_sync(self, user_query: str, session_id: str = None) -> AgentState:
        """Synchronous version of process_query"""
        logger.info(f"Processing query (sync): {user_query}")
        
        # Create initial state
        initial_state = StateManager.create_initial_state(user_query)
        
        if session_id:
            initial_state.session_id = session_id
        
        # Create execution plan  
        plan = PlanManager.create_basic_sql_plan(user_query, [])
        initial_state._current_plan = plan
        
        # Configure for graph execution
        config = {
            "configurable": {
                "thread_id": session_id or initial_state.session_id
            }
        }
        
        try:
            # Execute the workflow synchronously
            final_state = self.graph.invoke(initial_state, config)
            
            logger.info(f"Query processing completed. Success: {final_state.processing_complete}")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            initial_state.add_error(f"Workflow execution failed: {e}")
            return initial_state
    
    def get_conversation_history(self, session_id: str) -> list:
        """Get conversation history for a session"""
        try:
            # This would retrieve from the checkpointer
            # Implementation depends on your persistence needs
            return []
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            return []

# Create the main agent instance
agent = PowerBISQLAgent()

# Export main classes
__all__ = ["PowerBISQLAgent", "agent"]