# flow/graph.py

import logging
from typing import Dict, Any, Literal, TypedDict, List, Optional
from datetime import datetime
import pandas as pd

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage

# Import our state management
from state.agent_state import AgentState, StateManager
from state.plan_state import PlanManager, StepStatus

# Import our nodes
from nodes.schema_inspector import schema_inspector_node
from nodes.planner import planner_node

# Import tools for remaining nodes
from tools.sql_tools import sql_executor
from tools.validation_tools import query_validator, validate_complete_query

from tools.error_manager import error_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define TypedDict for LangGraph compatibility
class GraphState(TypedDict):
    user_query: str
    original_query: str
    selected_tables: List[str]
    schema_context: Dict[str, Any]
    business_intent: str
    generated_sql: str
    cleaned_sql: str
    validation_passed: bool
    execution_successful: bool
    result_count: int
    processing_complete: bool
    messages: List[BaseMessage]
    errors: List[str]
    warnings: List[str]
    validation_results: Dict[str, Any]
    execution_results: Dict[str, Any]
    needs_correction: bool
    correction_attempts: int
    max_correction_attempts: int
    execution_time: Optional[float]
    few_shot_examples: List[Dict[str, Any]]
    query_complexity: str
    table_relationships: Dict[str, Any]
    is_follow_up_query: bool
    session_id: str
    timestamp: str 
    requires_user_input: bool
    # Response formatting fields
    response_format: str
    show_sql: bool
    show_execution_details: bool
    conversation_context: Dict[str, Any]
    last_query_topic: str
    last_tables_used: List[str]
    last_result_summary: str
    last_user_intent: str
    active_entities: Dict[str, Any]
    context_window: List[Dict[str, Any]]

def agent_state_to_graph_state(agent_state: AgentState) -> GraphState:
    """Convert AgentState to GraphState"""
    return GraphState(
        user_query=agent_state.user_query,
        original_query=agent_state.original_query,
        selected_tables=agent_state.selected_tables,
        schema_context=agent_state.schema_context,
        business_intent=agent_state.business_intent,
        generated_sql=agent_state.generated_sql,
        cleaned_sql=agent_state.cleaned_sql,
        validation_passed=agent_state.validation_passed,
        execution_successful=agent_state.execution_successful,
        result_count=agent_state.result_count,
        processing_complete=agent_state.processing_complete,
        messages=agent_state.messages,
        errors=agent_state.errors,
        warnings=agent_state.warnings,
        validation_results=agent_state.validation_results,
        execution_results=agent_state.execution_results,
        needs_correction=agent_state.needs_correction,
        correction_attempts=agent_state.correction_attempts,
        max_correction_attempts=agent_state.max_correction_attempts,
        execution_time=agent_state.execution_time or None,
        few_shot_examples=agent_state.few_shot_examples,
        query_complexity=agent_state.query_complexity,
        table_relationships=agent_state.table_relationships,
        is_follow_up_query=agent_state.is_follow_up_query,
        session_id=str(agent_state.session_id),
        timestamp=str(agent_state.timestamp),
        requires_user_input=agent_state.requires_user_input,
        response_format=agent_state.response_format,
        show_sql=agent_state.show_sql,
        show_execution_details=agent_state.show_execution_details,
        conversation_context=agent_state.conversation_context,
        last_query_topic=agent_state.last_query_topic,
        last_tables_used=agent_state.last_tables_used,
        last_result_summary=agent_state.last_result_summary,
        last_user_intent=agent_state.last_user_intent,
        active_entities=agent_state.active_entities,
        context_window=agent_state.context_window
        
    )

def graph_state_to_agent_state(graph_state: GraphState) -> AgentState:
    """Convert GraphState to AgentState"""
    agent_state = AgentState(
        user_query=graph_state["user_query"],
        original_query=graph_state["original_query"],
        selected_tables=graph_state["selected_tables"],
        schema_context=graph_state["schema_context"],
        business_intent=graph_state["business_intent"],
        generated_sql=graph_state["generated_sql"],
        cleaned_sql=graph_state["cleaned_sql"],
        validation_passed=graph_state["validation_passed"],
        execution_successful=graph_state["execution_successful"],
        result_count=graph_state["result_count"],
        processing_complete=graph_state["processing_complete"],
        messages=graph_state["messages"],
        errors=graph_state["errors"],
        warnings=graph_state["warnings"],
        validation_results=graph_state["validation_results"],
        execution_results=graph_state["execution_results"],
        needs_correction=graph_state["needs_correction"],
        correction_attempts=graph_state["correction_attempts"],
        max_correction_attempts=graph_state["max_correction_attempts"],
        execution_time=graph_state.get("execution_time", None),
        few_shot_examples=graph_state["few_shot_examples"],
        query_complexity=graph_state["query_complexity"],
        table_relationships=graph_state["table_relationships"],
        is_follow_up_query=graph_state["is_follow_up_query"],
        session_id=graph_state["session_id"],
        #timestamp=datetime.fromisoformat(graph_state["timestamp"]) if graph_state["timestamp"] else datetime.now(),
        requires_user_input=graph_state["requires_user_input"],
        response_format=graph_state.get("response_format", "conversational"),
        show_sql=graph_state.get("show_sql", False),
        show_execution_details=graph_state.get("show_execution_details", False),
        conversation_context=graph_state.get("conversation_context", {}),
        last_query_topic=graph_state.get("last_query_topic", ""),
        last_tables_used=graph_state.get("last_tables_used", []),
        last_result_summary=graph_state.get("last_result_summary", ""),
        last_user_intent=graph_state.get("last_user_intent", ""),
        active_entities=graph_state.get("active_entities", {}),
        context_window=graph_state.get("context_window", [])
    )
    return agent_state

class QueryValidatorNode:
    """Node for validating SQL queries before execution"""
    
    def __init__(self):
        self.node_name = "query_validator"
    
    def __call__(self, state: GraphState) -> GraphState:
        logger.info("Validating SQL query")
        
        try:
            # Convert to AgentState
            agent_state = graph_state_to_agent_state(state)
            
            # Validate the query
            validation_result = validate_complete_query(
                agent_state.cleaned_sql,
                schema_context=agent_state.schema_context,
                intent=agent_state.business_intent,
                database_check=True
            )
            
            # Update state with validation results
            agent_state.validation_results = validation_result
            agent_state.validation_passed = validation_result["success"]
            
            # NEW: Extract detailed error information
            if not validation_result["success"] and "error_details" in validation_result:
                for error_detail in validation_result["error_details"]:
                    # Store detailed error info
                    error_info = {
                        "code": error_detail.error_type.code,
                        "category": error_detail.error_type.category,
                        "severity": error_detail.error_type.severity,
                        "user_message": error_detail.get_user_message(),
                        "retryable": error_detail.error_type.retryable
                    }
                    agent_state.add_error(error_detail.get_user_message())
                    
                    # Log structured error
                    logger.error(f"Validation error: {error_info}")
            
            # Add any warnings or errors (existing logic)
            if not validation_result["success"]:
                agent_state.needs_correction = True
                agent_state.correction_attempts += 1
            
            # Convert back to GraphState
            return agent_state_to_graph_state(agent_state)
            
        except Exception as e:
            error_msg = f"Query validation error: {str(e)}"
            logger.error(error_msg)
            agent_state = graph_state_to_agent_state(state)
            agent_state.add_error(error_msg)
            return agent_state_to_graph_state(agent_state)
        
class SQLExecutorNode:
    """Node for executing validated SQL queries"""
    
    def __init__(self):
        self.node_name = "sql_executor"
    
    def __call__(self, state: GraphState) -> GraphState:
        logger.info("Executing SQL query")
        
        try:
            agent_state = graph_state_to_agent_state(state)
            
            # Execute the query
            start_time = datetime.now()
            execution_result = sql_executor.execute_query(agent_state.cleaned_sql)
            end_time = datetime.now()
            
            # Calculate execution time
            agent_state.execution_time = (end_time - start_time).total_seconds()
            
            # Update state with execution results
            agent_state.execution_results = execution_result
            agent_state.execution_successful = execution_result["success"]
            
            if execution_result["success"]:
                agent_state.result_count = execution_result.get("row_count", 0)
                logger.info(f"Query executed successfully. Returned {agent_state.result_count} rows")
                
                # Update conversation context (existing)
                agent_state.update_conversation_context(agent_state.execution_results)
            else:
                # NEW: Handle errors with error manager
                error_detail = execution_result.get("error_detail")
                
                if error_detail:
                    # Use user-friendly message
                    user_message = execution_result.get("user_message", execution_result.get("error"))
                    agent_state.add_error(user_message)
                    
                    # Log structured error
                    logger.error(f"Execution error: {error_detail.to_dict()}")
                    
                    # Check if retryable
                    if error_detail.error_type.retryable:
                        agent_state.needs_correction = True
                        agent_state.correction_attempts += 1
                else:
                    # Fallback for unclassified errors
                    agent_state.add_error(f"Query execution failed: {execution_result.get('error', 'Unknown error')}")
                    agent_state.needs_correction = True
                    agent_state.correction_attempts += 1
            
            # Convert back to GraphState
            return agent_state_to_graph_state(agent_state)
            
        except Exception as e:
            error_msg = f"Query execution error: {str(e)}"
            logger.error(error_msg)
            agent_state = graph_state_to_agent_state(state)
            agent_state.add_error(error_msg)
            return agent_state_to_graph_state(agent_state)


class OutputFormatterNode:
    """Node for formatting and presenting results"""
    
    def __init__(self):
        self.node_name = "output_formatter"
        # Initialize LLM for conversational response generation
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7  # Slightly higher for more natural responses
        )
    
    def __call__(self, state: GraphState) -> GraphState:
        logger.info("Formatting output")
        
        try:
            # Convert to AgentState
            agent_state = graph_state_to_agent_state(state)
            
            # Format the response based on execution results
            if agent_state.execution_successful:
                formatted_output = self._format_successful_results(agent_state)
            else:
                formatted_output = self._format_error_response(agent_state)
            
            # Add AI response to conversation history
            agent_state.add_ai_message(formatted_output)
            
            # Mark processing as complete
            agent_state.processing_complete = True
            
            logger.info("Output formatting completed")
            
            # Convert back to GraphState
            return agent_state_to_graph_state(agent_state)
            
        except Exception as e:
            error_msg = f"Output formatting error: {str(e)}"
            logger.error(error_msg)
            agent_state = graph_state_to_agent_state(state)
            agent_state.add_error(error_msg)
            return agent_state_to_graph_state(agent_state)
    
    def _format_successful_results(self, state: AgentState) -> str:
        """Format successful query results"""
        results = state.execution_results
        
        if not results.get("data"):
            return "I didn't find any results matching your query. The query executed successfully but returned no data."
        
        data = results["data"]
        
        # Route to appropriate formatter based on response_format
        if state.response_format == "detailed":
            return self._format_detailed_response(state, data)
        else:
            return self._format_conversational_response(state, data)
        
    def _format_as_markdown(self, data: list) -> str:
        """Convert data list to a Markdown table"""
        if not data:
            return ""
        
        try:
            df = pd.DataFrame(data)
            
            # Format numbers: Add commas to integers, keep floats to 2 decimals
            # This makes "141396.74" look like "141,396.74"
            for col in df.select_dtypes(include=['float']).columns:
                df[col] = df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "")
            
            # If dataset is large, cap it for display (e.g., top 10 rows)
            if len(df) > 10:
                preview = df.head(10)
                markdown_table = preview.to_markdown(index=False)
                return f"{markdown_table}\n\n*(...and {len(df)-10} more rows)*"
            
            return df.to_markdown(index=False)
        except Exception as e:
            logger.error(f"Markdown formatting failed: {e}")
            return str(data) # Fallback    
    
    def _format_conversational_response(self, state: AgentState, data: list) -> str:
        """Generate natural language conversational response"""

        data_summary = self._prepare_data_summary(data, state.user_query)
        
        # Check if user explicitly wants full/complete list
        user_wants_full_list = any(
            phrase in state.user_query.lower() 
            for phrase in ["full list", "complete list", "all of them", "show all", "entire list", "show them all"]
        )
        
            # Add instruction override if needed
        if user_wants_full_list:
            instruction_override = "\n8. User asked for the COMPLETE list, so list ALL items, not just a sample."
        else:
            instruction_override = ""
        
        # Build prompt for conversational response
        prompt = f"""You are a helpful data analyst assistant. Convert this database query result into a natural, conversational response.

    User asked: "{state.user_query}"
    Number of results: {len(data)}
    Data sample: {data_summary}

    Instructions:
    1. Start with a natural summary (e.g., "I found X customers from Germany")
    2. Highlight key insights or interesting data points
    3. Use friendly, conversational language
    4. If there are many results, mention the total and highlight top items
    5. End with an offer to provide more details if needed
    6. Keep the response concise (3-8 sentences max)
    7. Do NOT show raw SQL or technical details unless specifically asked
    {instruction_override}

    Generate a conversational response:"""


        try:
            # Generate response using LLM
            conversational_text = self.llm.invoke(prompt).content
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            conversational_text = self._fallback_conversational_response(state, data)
        
        # --- NEW CODE STARTS HERE ---
        
        # Generate the Markdown Table
        table_view = self._format_as_markdown(data)
        
        # Combine: Conversational Text + Table
        response = f"{conversational_text}\n\n{table_view}"
        
        # --- NEW CODE ENDS HERE ---
        
        if state.show_sql:
            response += f"\n\n📝 **SQL Query:**\n```sql\n{state.cleaned_sql}\n```"
        
        if state.show_execution_details:
            response += f"\n\n⏱️ Execution time: {state.execution_time:.2f}s | Tables: {', '.join(state.selected_tables)}"
        
        return response
    
    def _format_detailed_response(self, state: AgentState, data: list) -> str:
        """Format detailed response with full table"""
        
        # Use SQL executor's formatting
        formatted_display = sql_executor.format_results_for_display(state.execution_results)
        
        # Add context
        context_info = f"""
📊 **Query Results**

{formatted_display}

**Query Details:**
- SQL: {state.cleaned_sql}
- Execution Time: {state.execution_time:.2f} seconds
- Tables Used: {', '.join(state.selected_tables)}
- Total Rows: {len(data)}
"""
        
        return context_info
    
    def _prepare_data_summary(self, data: list, user_query: str = "") -> str:
        """Prepare data summary with context awareness"""
        if not data:
            return "No data"
        
        # Determine how many rows to show based on context
        user_query_lower = user_query.lower()
        
        if any(phrase in user_query_lower for phrase in ["full", "complete", "all", "entire"]):
            # User wants everything
            sample_size = len(data)
            show_all = True
        elif len(data) <= 10:
            # Small result set - show everything
            sample_size = len(data)
            show_all = True
        else:
            # Large result set - show sample
            sample_size = min(5, len(data))
            show_all = False
        
        sample_data = data[:sample_size]
        
        # Get column names
        columns = list(sample_data[0].keys()) if sample_data else []
        
        # Format summary
        if show_all:
            summary = f"Columns: {', '.join(columns)}\n"
            summary += f"All {len(data)} rows:\n"
        else:
            summary = f"Columns: {', '.join(columns)}\n"
            summary += f"Showing {sample_size} of {len(data)} rows:\n"
        
        for i, row in enumerate(sample_data, 1):
            row_str = ", ".join([f"{k}: {v}" for k, v in list(row.items())[:4]])  # First 4 columns
            summary += f"  {i}. {row_str}\n"
        
        if not show_all and len(data) > sample_size:
            summary += f"  ... and {len(data) - sample_size} more rows\n"
        
        return summary
    
    def _fallback_conversational_response(self, state: AgentState, data: list) -> str:
        """Simple template-based response when LLM fails"""
        
        count = len(data)
        
        # Determine response based on query intent
        if state.business_intent == "aggregation":
            # Single value result
            if count == 1 and len(data[0]) == 1:
                value = list(data[0].values())[0]
                return f"The result is: **{value}**"
        
        # Multiple results
        if count <= 5:
            return f"I found {count} result{'s' if count != 1 else ''} for your query. Here they are:\n\n{self._format_simple_list(data)}"
        else:
            top_items = self._format_simple_list(data[:5])
            return f"I found {count} results. Here are the top 5:\n\n{top_items}\n\n_Type '/detailed' to see the complete list._"
    
    def _format_simple_list(self, data: list) -> str:
        """Format data as a simple bullet list"""
        if not data:
            return ""
        
        items = []
        for i, row in enumerate(data, 1):
            # Get first 3 columns
            values = list(row.values())[:3]
            item_str = " | ".join([str(v) for v in values])
            items.append(f"{i}. {item_str}")
        
        return "\n".join(items)
    
    def _format_error_response(self, state: AgentState) -> str:
        """Format error response with enhanced schema inspection error handling"""
        
        # NEW: Check for schema inspection failure first
        if not state.selected_tables:
            # Use the error message already formatted by schema inspector
            if state.errors:
                return state.errors[-1]
            
            # Fallback: generic schema error
            from database.northwind_context import BUSINESS_CONTEXT
            available_tables = list(BUSINESS_CONTEXT.keys())
            
            return f"""❌ I couldn't identify which database tables to use for your query: "{state.user_query}"

    📋 **Available tables:**
    {chr(10).join(f"  • {table}" for table in sorted(available_tables))}

    💡 **Try asking:**
    • Show me data from customers
    • List all orders
    • Show me products
    """
        
        # Original error formatting for other errors
        error_msg = "❌ I encountered an issue processing your query:\n\n"
        
        # Add specific error details
        if state.errors:
            error_msg += f"**Error:** {state.errors[-1]}\n\n"
        
        # Add the attempted SQL for reference
        if state.cleaned_sql and state.show_sql:
            error_msg += f"**Attempted SQL:**\n```sql\n{state.cleaned_sql}\n```\n\n"
        
        # Add suggestions if available
        if state.validation_results and "recommendations" in state.validation_results:
            recommendations = state.validation_results["recommendations"][:3]
            if recommendations:
                error_msg += "**Suggestions:**\n"
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
            final_state = graph_state_to_agent_state(final_state)
            logger.info(f"Query processing completed. Success: {final_state.processing_complete}")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            initial_state.add_error(f"Workflow execution failed: {e}")
            return initial_state
    
    def process_query_sync(self, user_query: str, session_id: str = None, existing_state: AgentState = None) -> AgentState:
        """Synchronous version of process_query with context preservation"""
        logger.info(f"Processing query (sync): {user_query}")
        
        # Use existing state if provided, otherwise create new
        if existing_state:
            # CRITICAL: Preserve context but reset query-specific fields
            initial_agent_state = existing_state
            
            # Update query
            initial_agent_state.user_query = user_query
            initial_agent_state.original_query = user_query
            initial_agent_state.add_user_message(user_query)
            
            # Reset query-specific fields (but keep context!)
            initial_agent_state.selected_tables = []  # Will be re-selected
            initial_agent_state.schema_context = {}
            initial_agent_state.generated_sql = ""
            initial_agent_state.cleaned_sql = ""
            initial_agent_state.validation_passed = False
            initial_agent_state.validation_results = {}
            initial_agent_state.execution_successful = False
            initial_agent_state.execution_results = {}
            initial_agent_state.result_count = 0
            initial_agent_state.errors = []
            initial_agent_state.warnings = []
            initial_agent_state.needs_correction = False
            initial_agent_state.correction_attempts = 0
            initial_agent_state.processing_complete = False
            initial_agent_state.execution_time = None
            
            # KEEP THESE (conversation context):
            # - last_query_topic
            # - last_tables_used
            # - last_result_summary
            # - last_user_intent
            # - active_entities
            # - context_window
            # - messages (conversation history)
            
            logger.info(f"📚 Reusing state with context: last_topic={initial_agent_state.last_query_topic}, last_tables={initial_agent_state.last_tables_used}")
        else:
            # Create new state
            initial_agent_state = StateManager.create_initial_state(user_query)
            
            if session_id:
                initial_agent_state.session_id = session_id
            
            logger.info("📝 Created new state (no previous context)")
        
        # Convert to GraphState
        initial_state = agent_state_to_graph_state(initial_agent_state)
        
        # Configure for graph execution
        config = {
            "configurable": {
                "thread_id": session_id or initial_agent_state.session_id
            }
        }
        
        try:
            # Execute the workflow synchronously
            result = self.graph.invoke(initial_state, config)
            
            # Convert result back to AgentState
            final_state = graph_state_to_agent_state(result)
            
            logger.info(f"Query processing completed. Success: {final_state.processing_complete}")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            initial_agent_state.add_error(f"Workflow execution failed: {e}")
            return initial_agent_state

    
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