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
from client.mcp_client import mcp_client as sql_executor
#from tools.sql_tools import sql_executor
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
            agent_state = graph_state_to_agent_state(state)
            
            validation_result = validate_complete_query(
                agent_state.cleaned_sql,
                schema_context=agent_state.schema_context,
                intent=agent_state.business_intent,
                database_check=True
            )
            
            # Use repaired SQL (CTE fix may have changed it)
            repaired_sql = validation_result.get("syntax_validation", {}).get("query")
            if repaired_sql and repaired_sql != agent_state.cleaned_sql:
                logger.info("CTE auto-repair applied; updating cleaned_sql")
                agent_state.cleaned_sql = repaired_sql

            agent_state.validation_results = validation_result
            agent_state.validation_passed = validation_result["success"]

            if not validation_result["success"]:
                # error_details are plain dicts (not ErrorDetail objects)
                syntax_errors = validation_result.get("syntax_validation", {}).get("error_details", [])
                for err in syntax_errors:
                    msg = err.get("user_message") or err.get("message") or "Validation failed"
                    agent_state.add_error(msg)
                    logger.error(f"Validation error: {err}")
            
            if not validation_result["success"]:
                agent_state.needs_correction = True
                agent_state.correction_attempts += 1
            
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
            
            start_time = datetime.now()
            execution_result = sql_executor.execute_query(agent_state.cleaned_sql)
            end_time = datetime.now()
            
            agent_state.execution_time = (end_time - start_time).total_seconds()
            agent_state.execution_results = execution_result
            agent_state.execution_successful = execution_result["success"]
            
            if execution_result["success"]:
                agent_state.result_count = execution_result.get("row_count", 0)
                logger.info(f"Query executed successfully. Returned {agent_state.result_count} rows")
                agent_state.update_conversation_context(agent_state.execution_results)
            else:
                # error_detail is now a plain dict
                error_detail = execution_result.get("error_detail")
                user_message = execution_result.get("user_message") or execution_result.get("error", "Unknown error")
                agent_state.add_error(user_message)
                logger.error(f"Execution error: {error_detail}")

                retryable = (error_detail or {}).get("retryable", True)
                if retryable:
                    agent_state.needs_correction = True
                    agent_state.correction_attempts += 1
            
            return agent_state_to_graph_state(agent_state)
            
        except Exception as e:
            error_msg = f"Query execution error: {str(e)}"
            logger.error(error_msg)
            agent_state = graph_state_to_agent_state(state)
            agent_state.add_error(error_msg)
            return agent_state_to_graph_state(agent_state)


class OutputFormatterNode:
    """Enhanced node for conversational responses"""
    
    def __init__(self):
        self.node_name = "output_formatter"
    
    def __call__(self, state: GraphState) -> GraphState:
        logger.info("🎨 Formatting output")
        
        try:
            agent_state = graph_state_to_agent_state(state)
            
            if agent_state.execution_successful:
                formatted_output = self._format_successful_results(agent_state)
            else:
                formatted_output = self._format_error_response(agent_state)
            
            agent_state.add_ai_message(formatted_output)
            agent_state.processing_complete = True
            
            return agent_state_to_graph_state(agent_state)
            
        except Exception as e:
            error_msg = f"Output formatting error: {str(e)}"
            logger.error(error_msg)
            agent_state = graph_state_to_agent_state(state)
            agent_state.add_error(error_msg)
            return agent_state_to_graph_state(agent_state)
    
    def _format_successful_results(self, state: AgentState) -> str:
        """Format successful results"""
        results = state.execution_results
        
        if not results.get("data"):
            return "I didn't find any results matching your query."
        
        data = results["data"]
        row_count = len(data)
        
        intro = self._generate_intro(state.user_query, row_count, data)
        summary = self._generate_summary(state.user_query, data)
        
        response = f"{intro}\n\n{summary}"
        
        if state.show_execution_details:
            response += f"\n\n⏱️ Executed in {state.execution_time:.2f}s"
        
        return response
    
    def _generate_intro(self, query: str, row_count: int, data: list) -> str:
        """Generate context-aware introduction"""
        query_lower = query.lower()
        
        # Complex quantity queries
        if 'quantity' in query_lower and 'country' in query_lower:
            if 'top customer' in query_lower or 'best customer' in query_lower:
                return "Here's the product quantity ordered by top customers in each country:"
            return "Here's the quantity breakdown by country:"
        
        # Top queries
        if 'top' in query_lower:
            if 'customer' in query_lower:
                return f"Here are the top {min(row_count, 10)} customers:"
            elif 'product' in query_lower:
                return f"Here are the top {min(row_count, 10)} products:"
        
        # Best/worst queries
        if 'best' in query_lower or 'highest' in query_lower:
            if 'product' in query_lower:
                return "Here's the best performing product:"
            elif 'customer' in query_lower:
                return "Here's the best customer:"
        
        if 'worst' in query_lower or 'lowest' in query_lower:
            if 'product' in query_lower:
                return "Here's the worst performing product:"
            elif 'customer' in query_lower:
                return "Here's the worst customer:"
        
        # Comparison queries
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'which']):
            return "Here's the comparison:"
        
        # Quantity/count queries
        if 'quantity' in query_lower or 'how many' in query_lower:
            return f"Here's the breakdown:"
        
        return f"I found {row_count} result(s):"
    
    def _generate_summary(self, query: str, data: list) -> str:
        """Generate smart summary"""
        if not data:
            return "No data found."
        
        query_lower = query.lower()
        first_row_keys = list(data[0].keys())
        
        # Comparison queries
        if any(word in query_lower for word in ['compare', 'vs', 'versus', 'which']):
            return self._summarize_comparison(data)
        
        # Quantity by country/customer
        if 'quantity' in query_lower:
            if 'country' in first_row_keys or any('country' in k.lower() for k in first_row_keys):
                return self._summarize_by_country(data)
            elif any('company' in k.lower() or 'customer' in k.lower() for k in first_row_keys):
                return self._summarize_by_customer(data)
        
        # Top/best queries
        if any(word in query_lower for word in ['top', 'best', 'highest']):
            return self._summarize_top_results(data)
        
        # Worst/bottom queries  
        if any(word in query_lower for word in ['worst', 'bottom', 'lowest']):
            return self._summarize_bottom_results(data)
        
        # Smart generic
        return self._summarize_smart_generic(data)
    
    def _summarize_comparison(self, data: list) -> str:
        """Summarize comparison queries"""
        if len(data) < 2:
            return self._summarize_smart_generic(data)
        
        metric_col = None
        for col in data[0].keys():
            if any(term in col.lower() for term in ['total', 'revenue', 'quantity', 'sales', 'amount']):
                metric_col = col
                break
        
        if not metric_col:
            return self._summarize_smart_generic(data)
        
        sorted_data = sorted(data, key=lambda x: float(x.get(metric_col, 0) or 0), reverse=True)
        
        winner = sorted_data[0]
        name_col = [c for c in winner.keys() if c != metric_col][0]
        
        winner_name = winner[name_col]
        winner_value = winner[metric_col]
        
        lines = [f"{winner_name} had the highest with {self._format_number(winner_value)}."]
        lines.append("\nBreakdown:")
        
        for i, row in enumerate(sorted_data[:10], 1):
            name = row[name_col]
            value = row[metric_col]
            lines.append(f"{i}. {name}: {self._format_number(value)}")
        
        return "\n".join(lines)
    
    def _summarize_by_country(self, data: list) -> str:
        """Summarize data grouped by country"""
        lines = []
        
        for i, row in enumerate(data[:20], 1):
            parts = []
            for key, value in row.items():
                key_lower = key.lower()
                if 'country' in key_lower:
                    parts.insert(0, f"{value}")
                elif 'quantity' in key_lower:
                    parts.append(f"{self._format_number(value)} units")
                elif 'company' in key_lower or 'customer' in key_lower:
                    parts.insert(1, f"({value})")
                elif any(term in key_lower for term in ['revenue', 'total', 'sales']):
                    parts.append(f"${self._format_number(value)}")
            
            lines.append(f"{i}. {' - '.join(parts)}")
        
        if len(data) > 20:
            lines.append(f"\n...and {len(data) - 20} more")
        
        return "\n".join(lines)
    
    def _summarize_by_customer(self, data: list) -> str:
        """Summarize data grouped by customer"""
        lines = []
        
        for i, row in enumerate(data[:20], 1):
            parts = []
            for key, value in row.items():
                key_lower = key.lower()
                if 'company' in key_lower or 'customer' in key_lower:
                    parts.insert(0, f"{value}")
                elif 'country' in key_lower:
                    parts.append(f"({value})")
                elif 'quantity' in key_lower:
                    parts.append(f"{self._format_number(value)} units")
                elif any(term in key_lower for term in ['revenue', 'total', 'sales']):
                    parts.append(f"${self._format_number(value)}")
            
            lines.append(f"{i}. {' - '.join(parts)}")
        
        if len(data) > 20:
            lines.append(f"\n...and {len(data) - 20} more customers")
        
        return "\n".join(lines)
    
    def _summarize_top_results(self, data: list) -> str:
        """Summarize top N results"""
        lines = []
        
        for i, row in enumerate(data[:10], 1):
            parts = []
            for key, value in row.items():
                key_lower = key.lower()
                if 'name' in key_lower:
                    parts.insert(0, str(value))
                elif any(term in key_lower for term in ['revenue', 'total', 'sales']):
                    parts.append(f"${self._format_number(value)}")
                elif 'quantity' in key_lower:
                    parts.append(f"{self._format_number(value)} units")
                elif 'country' in key_lower:
                    parts.append(f"({value})")
            
            lines.append(f"{i}. {' - '.join(parts)}")
        
        return "\n".join(lines)
    
    def _summarize_bottom_results(self, data: list) -> str:
        """Summarize worst/bottom results"""
        lines = []
        
        for i, row in enumerate(data[:10], 1):
            parts = []
            for key, value in row.items():
                key_lower = key.lower()
                if 'name' in key_lower:
                    parts.insert(0, str(value))
                elif any(term in key_lower for term in ['revenue', 'total', 'sales']):
                    parts.append(f"${self._format_number(value)}")
                elif 'quantity' in key_lower:
                    parts.append(f"{self._format_number(value)} units")
            
            lines.append(f"{i}. {' - '.join(parts)}")
        
        return "\n".join(lines)
    
    def _summarize_smart_generic(self, data: list) -> str:
        """Smart generic summary"""
        lines = []
        columns = list(data[0].keys())
        
        for i, row in enumerate(data[:20], 1):
            parts = []
            
            for key, value in row.items():
                key_lower = key.lower()
                
                if 'name' in key_lower:
                    parts.insert(0, str(value))
                elif 'country' in key_lower:
                    parts.append(f"({value})")
                elif 'quantity' in key_lower:
                    parts.append(f"{self._format_number(value)} units")
                elif any(term in key_lower for term in ['revenue', 'total', 'sales', 'amount']):
                    parts.append(f"${self._format_number(value)}")
                else:
                    parts.append(f"{value}")
            
            lines.append(f"{i}. {' - '.join(parts[:4])}")
        
        if len(data) > 20:
            lines.append(f"\n...and {len(data) - 20} more rows")
        
        return "\n".join(lines)
    
    def _format_number(self, value) -> str:
        """Format numbers nicely"""
        try:
            num = float(value)
            if num >= 1000000:
                return f"{num/1000000:.1f}M"
            elif num >= 1000:
                return f"{num:,.0f}"
            else:
                return f"{num:.2f}"
        except:
            return str(value)
    
    def _format_error_response(self, state: AgentState) -> str:
        """Format error response"""
        if state.errors:
            return f"❌ {state.errors[-1]}"
        return "❌ I encountered an issue processing your query."


class PowerBISQLAgent:
    """Main agent class"""
    
    def __init__(self):
        self.graph = None
        self.memory = MemorySaver()
        self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        logger.info("Building LangGraph workflow")
        
        query_validator_node = QueryValidatorNode()
        sql_executor_node = SQLExecutorNode()
        output_formatter_node = OutputFormatterNode()
        
        workflow = StateGraph(AgentState)
        
        workflow.add_node("schema_inspector", self._wrap_node(schema_inspector_node.execute))
        workflow.add_node("planner", self._wrap_node(planner_node.execute))
        workflow.add_node("query_validator", query_validator_node)
        workflow.add_node("sql_executor", sql_executor_node)
        workflow.add_node("output_formatter", output_formatter_node)
        
        from flow.edge import should_continue_to_planner, should_retry_query, should_execute_query
        
        workflow.set_entry_point("schema_inspector")
        
        workflow.add_conditional_edges(
            "schema_inspector",
            should_continue_to_planner,
            {"continue": "planner", "error": END}
        )
        
        workflow.add_conditional_edges(
            "planner",
            should_continue_to_planner,
            {"continue": "query_validator", "error": END}
        )
        
        workflow.add_conditional_edges(
            "query_validator",
            should_execute_query,
            {"execute": "sql_executor", "retry": "planner", "error": END}
        )
        
        workflow.add_conditional_edges(
            "sql_executor",
            should_retry_query,
            {"success": "output_formatter", "retry": "planner", "error": END}
        )
        
        workflow.add_edge("output_formatter", END)
        
        self.graph = workflow.compile(checkpointer=self.memory)
        logger.info("LangGraph workflow built successfully")
    
    def _wrap_node(self, node_func):
        def wrapper(state: AgentState) -> AgentState:
            return node_func(state)
        return wrapper
    
    async def process_query(self, user_query: str, session_id: str = None) -> AgentState:
        logger.info(f"Processing query: {user_query}")
        initial_state = StateManager.create_initial_state(user_query)
        if session_id:
            initial_state.session_id = session_id
        plan = PlanManager.create_basic_sql_plan(user_query, [])
        initial_state._current_plan = plan
        config = {"configurable": {"thread_id": session_id or initial_state.session_id}}
        
        try:
            final_state = await self.graph.ainvoke(initial_state, config)
            final_state = graph_state_to_agent_state(final_state)
            logger.info(f"Query processing completed. Success: {final_state.processing_complete}")
            return final_state
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            initial_state.add_error(f"Workflow execution failed: {e}")
            return initial_state
    
    def process_query_sync(self, user_query: str, session_id: str = None, existing_state: AgentState = None) -> AgentState:
        logger.info(f"Processing query (sync): {user_query}")
        
        if existing_state:
            initial_agent_state = existing_state
            initial_agent_state.user_query = user_query
            initial_agent_state.original_query = user_query
            initial_agent_state.add_user_message(user_query)
            
            initial_agent_state.selected_tables = []
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
        else:
            initial_agent_state = StateManager.create_initial_state(user_query)
            if session_id:
                initial_agent_state.session_id = session_id
        
        initial_state = agent_state_to_graph_state(initial_agent_state)
        config = {"configurable": {"thread_id": session_id or initial_agent_state.session_id}}
        
        try:
            result = self.graph.invoke(initial_state, config)
            final_state = graph_state_to_agent_state(result)
            logger.info(f"Query processing completed. Success: {final_state.processing_complete}")
            return final_state
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            initial_agent_state.add_error(f"Workflow execution failed: {e}")
            return initial_agent_state
    
    def get_conversation_history(self, session_id: str) -> list:
        try:
            return []
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            return []

agent = PowerBISQLAgent()

__all__ = ["PowerBISQLAgent", "agent"]