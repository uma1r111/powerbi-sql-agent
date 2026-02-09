# state/agent_state.py

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from datetime import datetime
import uuid
import re
from asyncssh import logger

class AgentState(BaseModel):
    """
    Comprehensive state management for the PowerBI SQL Agent
    Combines proven memory patterns with advanced planning and context tracking
    """
    
    # === SESSION MANAGEMENT ===
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # === USER INPUT & CONVERSATION ===
    user_query: str = Field(default="", description="Current user question/request")
    original_query: str = Field(default="", description="Original user query before any processing")
    
    # === MEMORY (From YouTube - Proven Pattern) ===
    messages: List[BaseMessage] = Field(default_factory=list, description="Chat history for context")
    conversation_summary: Optional[str] = Field(default=None, description="Summary of long conversations")
    
    # === PLANNING & EXECUTION ===
    current_plan: str = Field(default="", description="Current execution plan")
    plan_steps: List[str] = Field(default_factory=list, description="Breakdown of plan steps")
    completed_steps: List[str] = Field(default_factory=list, description="Steps already completed")
    
    # === SCHEMA & CONTEXT AWARENESS ===
    selected_tables: List[str] = Field(default_factory=list, description="Tables relevant to current query")
    schema_context: Dict[str, Any] = Field(default_factory=dict, description="Schema information for selected tables")
    table_relationships: Dict[str, Any] = Field(default_factory=dict, description="Relationships between selected tables")
    
    # === DYNAMIC FEW-SHOT (From YouTube - Smart Pattern) ===
    few_shot_examples: List[Dict[str, str]] = Field(default_factory=list, description="Dynamically selected examples")
    example_relevance_scores: Dict[str, float] = Field(default_factory=dict, description="Relevance scores for examples")
    
    # === QUERY PROCESSING ===
    generated_sql: str = Field(default="", description="Generated SQL query")
    cleaned_sql: str = Field(default="", description="Cleaned and validated SQL query")
    
    # === VALIDATION & CORRECTION ===
    validation_results: Dict[str, Any] = Field(default_factory=dict, description="Query validation results")
    validation_passed: bool = Field(default=False, description="Whether query passed validation")
    correction_attempts: int = Field(default=0, description="Number of correction attempts made")
    max_correction_attempts: int = Field(default=3, description="Maximum correction attempts allowed")
    
    # === EXECUTION & RESULTS ===
    execution_results: Dict[str, Any] = Field(default_factory=dict, description="SQL execution results")
    execution_successful: bool = Field(default=False, description="Whether query executed successfully")
    result_count: int = Field(default=0, description="Number of results returned")
    
    # === ERROR HANDLING ===
    errors: List[str] = Field(default_factory=list, description="List of errors encountered")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    last_error: Optional[str] = Field(default=None, description="Most recent error message")
    
    # === BUSINESS CONTEXT ===
    business_intent: str = Field(default="", description="Detected business intent of the query")
    suggested_visualizations: List[str] = Field(default_factory=list, description="Recommended chart types")
    
    # === PERFORMANCE & METRICS ===
    query_complexity: str = Field(default="low", description="Estimated query complexity: low/medium/high")
    execution_time: Optional[float] = Field(default=None, description="Query execution time in seconds")
    token_usage: Dict[str, int] = Field(default_factory=dict, description="Token usage tracking")
    
    # === CONTROL FLAGS ===
    needs_correction: bool = Field(default=False, description="Whether query needs correction")
    requires_user_input: bool = Field(default=False, description="Whether user input is needed")
    is_follow_up_query: bool = Field(default=False, description="Whether this is a follow-up question")
    processing_complete: bool = Field(default=False, description="Whether processing is complete")

    # === RESPONSE FORMATTING (NEW) ===
    response_format: str = Field(default="conversational", description="Response format: conversational or detailed")
    show_sql: bool = Field(default=False, description="Whether to show SQL in response")
    show_execution_details: bool = Field(default=False, description="Show execution time and metadata")

        # === CONVERSATION CONTEXT (NEW) ===
    conversation_context: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Context from previous turns in conversation"
    )
    last_query_topic: str = Field(
        default="", 
        description="Topic of the last query (e.g., 'out_of_stock_products')"
    )
    last_tables_used: List[str] = Field(
        default_factory=list, 
        description="Tables used in the last query"
    )
    last_result_summary: str = Field(
        default="", 
        description="Brief summary of last query results"
    )
    last_user_intent: str = Field(
        default="", 
        description="What the user was trying to accomplish in last query"
    )
    active_entities: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Currently discussed entities (customers, products, etc.)"
    )
    context_window: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Last 5 query contexts for deep history"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    # === HELPER METHODS ===
    
    def add_user_message(self, message: str) -> None:
        """Add user message to conversation history"""
        self.messages.append(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        """Add AI message to conversation history"""
        self.messages.append(AIMessage(content=message))
    
    def update_plan(self, new_plan: str, steps: List[str] = None) -> None:
        """Update the current plan and steps"""
        self.current_plan = new_plan
        if steps:
            self.plan_steps = steps
    
    def complete_step(self, step: str) -> None:
        """Mark a plan step as completed"""
        if step in self.plan_steps and step not in self.completed_steps:
            self.completed_steps.append(step)
    
    def add_error(self, error: str) -> None:
        """Add error to the error list"""
        self.errors.append(error)
        self.last_error = error
    
    def add_warning(self, warning: str) -> None:
        """Add warning to the warning list"""
        self.warnings.append(warning)
    
    def reset_for_new_query(self, keep_conversation: bool = True) -> None:
        """Reset state for a new query while optionally keeping conversation history"""
        if not keep_conversation:
            self.messages = []
        
        # Reset query-specific fields
        self.user_query = ""
        self.original_query = ""
        self.current_plan = ""
        self.plan_steps = []
        self.completed_steps = []
        self.selected_tables = []
        self.schema_context = {}
        self.few_shot_examples = []
        self.generated_sql = ""
        self.cleaned_sql = ""
        self.validation_results = {}
        self.validation_passed = False
        self.execution_results = {}
        self.execution_successful = False
        self.result_count = 0
        self.correction_attempts = 0
        self.errors = []
        self.warnings = []
        self.last_error = None
        self.needs_correction = False
        self.requires_user_input = False
        self.processing_complete = False
        self.execution_time = None
    
    def get_conversation_context(self, max_messages: int = 10) -> List[BaseMessage]:
        """Get recent conversation context for prompt injection"""
        return self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
    
    def update_conversation_context(self, query_result: Dict[str, Any]) -> None:
        """Update conversation context after each successful query"""
        
        # Store current query context
        current_context = {
            "query": self.user_query,
            "tables": self.selected_tables.copy(),
            "intent": self.business_intent,
            "result_count": self.result_count,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update context window (keep last 5)
        self.context_window.append(current_context)
        if len(self.context_window) > 5:
            self.context_window.pop(0)
        
        # Update immediate context
        self.last_query_topic = self._extract_topic(self.user_query)
        self.last_tables_used = self.selected_tables.copy()
        self.last_result_summary = self._create_result_summary(query_result)
        self.last_user_intent = self.business_intent
        
        # Update active entities
        self._update_active_entities(query_result)
    
    def _extract_topic(self, query: str) -> str:
        """Extract main topic from query"""
        query_lower = query.lower()
        
        # Topic patterns
        topics = {
            "customers": ["customer", "client", "company"],
            "orders": ["order", "purchase", "sale"],
            "products": ["product", "item", "inventory"],
            "revenue": ["revenue", "sales", "income"],
            "employees": ["employee", "staff", "worker"],
            "out_of_stock": ["out of stock", "unavailable", "zero inventory"],
            "shipping": ["ship", "delivery", "freight"]
        }
        
        for topic, keywords in topics.items():
            if any(keyword in query_lower for keyword in keywords):
                return topic
        
        return "general"
    
    def _create_result_summary(self, query_result: Dict[str, Any]) -> str:
        """Create a brief summary of query results"""
        if not query_result or not query_result.get("success"):
            return "No results"
        
        count = query_result.get("row_count", 0)
        return f"{count} results found"
    
    def _update_active_entities(self, query_result: Dict[str, Any]) -> None:
        """Track entities being discussed"""
        if self.last_query_topic:
            self.active_entities[self.last_query_topic] = {
                "tables": self.last_tables_used,
                "count": self.result_count,
                "last_accessed": datetime.now().isoformat()
            }
        
        # Keep only last 3 active entities
        if len(self.active_entities) > 3:
            # Remove oldest
            oldest_key = min(
                self.active_entities.keys(),
                key=lambda k: self.active_entities[k]["last_accessed"]
            )
            del self.active_entities[oldest_key]
    
    def get_context_for_follow_up(self) -> Dict[str, Any]:
        """Get relevant context for follow-up query processing"""
        return {
            "last_topic": self.last_query_topic,
            "last_tables": self.last_tables_used,
            "last_summary": self.last_result_summary,
            "last_intent": self.last_user_intent,
            "active_entities": self.active_entities,
            "recent_queries": self.context_window[-3:] if self.context_window else []
        }

    
    def is_ready_for_execution(self) -> bool:
        """Check if state is ready for SQL execution"""
        return (
            bool(self.cleaned_sql) and 
            self.validation_passed and 
            not self.needs_correction and
            self.correction_attempts < self.max_correction_attempts
        )
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get summary of current progress"""
        total_steps = len(self.plan_steps)
        completed = len(self.completed_steps)
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed,
            "progress_percentage": (completed / total_steps * 100) if total_steps > 0 else 0,
            "current_step": self.plan_steps[completed] if completed < total_steps else "Complete",
            "has_errors": len(self.errors) > 0,
            "needs_attention": self.requires_user_input or self.needs_correction
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            "session_id": self.session_id,
            "user_query": self.user_query,
            "current_plan": self.current_plan,
            "selected_tables": self.selected_tables,
            "generated_sql": self.generated_sql,
            "validation_passed": self.validation_passed,
            "execution_successful": self.execution_successful,
            "result_count": self.result_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "progress": self.get_progress_summary()
        }

class StateManager:
    """
    Utility class for managing agent state across nodes
    """
    
    @staticmethod
    def create_initial_state(user_query: str) -> AgentState:
        """Create initial state for a new query"""
        state = AgentState()
        state.user_query = user_query
        state.original_query = user_query
        state.add_user_message(user_query)
        return state
    
    @staticmethod
    def detect_follow_up_query(state: AgentState) -> bool:
        """Enhanced follow-up detection using conversation context"""
        
        if len(state.messages) < 2:
            logger.info("❌ Not enough messages for follow-up (need at least 2)")
            return False
        
        query_lower = state.user_query.lower()
        logger.info(f"🔎 Analyzing query for follow-up: '{query_lower}'")
        
        # Explicit follow-up indicators
        explicit_indicators = [
            "their", "them", "those", "these", "it", "that",
            "also", "and", "what about", "how about", "can you also",
            "list them", "show them", "tell me more", "more details",
            "complete list", "full list", "all of them", "the rest"
        ]
        
        # Check explicit indicators
        has_explicit = any(indicator in query_lower for indicator in explicit_indicators)
        if has_explicit:
            logger.info(f"✅ Found explicit follow-up indicator in query")
        
        # Implicit patterns
        implicit_patterns = [
            r"^(show|list|display|give|tell)\s+(me\s+)?(the\s+)?(complete|full|entire|all)",
            r"^(more|other|additional)\s+",
            r"^(what|how)\s+(about|many|much)",
        ]
        
        has_implicit = any(re.match(pattern, query_lower) for pattern in implicit_patterns)
        if has_implicit:
            logger.info(f"✅ Matched implicit follow-up pattern")
        
        # Check context availability
        has_active_context = bool(state.last_query_topic or state.active_entities)
        logger.info(f"📚 Context available: {has_active_context} (topic={state.last_query_topic}, entities={list(state.active_entities.keys())})")
        
        # Short query without entity mentions
        has_no_entity = not any(
            entity in query_lower 
            for entity in ["customer", "order", "product", "employee", "supplier", "category"]
        )
        is_short_query = len(query_lower.split()) <= 5
        
        if has_no_entity and is_short_query and has_active_context:
            logger.info(f"✅ Short ambiguous query with available context")
        
        # Final decision
        is_follow_up = (
            has_explicit or 
            has_implicit or
            (is_short_query and has_no_entity and has_active_context)
        )
        
        logger.info(f"{'✅' if is_follow_up else '❌'} Final decision: is_follow_up={is_follow_up}")
        return is_follow_up

    
    @staticmethod
    def should_continue_processing(state: AgentState) -> bool:
        """Determine if processing should continue based on state"""
        if state.processing_complete:
            return False
        if state.correction_attempts >= state.max_correction_attempts:
            return False
        if state.requires_user_input:
            return False
        return True
    
    @staticmethod
    def prepare_for_next_node(state: AgentState, node_name: str) -> Dict[str, Any]:
        """Prepare state data for the next node"""
        return {
            "state": state,
            "node_context": {
                "previous_node": getattr(state, '_current_node', 'unknown'),
                "next_node": node_name,
                "timestamp": datetime.now()
            }
        }

# Export the main classes
__all__ = ["AgentState", "StateManager"]