# nodes/schema_inspector.py

import logging
from typing import Dict, Any, List
from datetime import datetime
from difflib import SequenceMatcher
import re

# Import our state management
from state.agent_state import AgentState, StateManager
from state.plan_state import ExecutionPlan, StepStatus

# Import our tools
from tools.schema_tools import schema_inspector
from database.northwind_context import BUSINESS_CONTEXT
from database.relationships import get_related_tables, get_join_path

# NEW: Import error manager
from tools.error_manager import error_manager

from utils.query_preprocessor import preprocessor


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SchemaInspectorNode:
    """
    Node responsible for inspecting database schema and providing rich business context
    This is the first node after user input - it sets up all the context for planning
    """
    
    def __init__(self):
        self.node_name = "schema_inspector"
        self.description = "Inspects database schema and provides business context for query planning"
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Main execution method for the schema inspector node
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state with schema context
        
        Inspect schema and select relevant tables"""
        
        logger.info(f"🔍 Schema Inspector analyzing query: {state.user_query}")
        
        # ✨ NEW: Preprocess the query
        preprocessed_query, corrections = preprocessor.preprocess(state.user_query)
        
        # Log corrections if any were made
        if corrections:
            logger.info(f"📝 Query preprocessing corrections: {corrections}")
            # Optionally store corrections in state
            state.add_warning(f"Auto-corrected: {', '.join(corrections)}")
        
        # Use preprocessed query for the rest of the logic
        original_query = state.user_query
        state.user_query = preprocessed_query  # Update with corrected version
        state.original_query = original_query  # Keep original for reference
        
        # Continue with existing schema inspection logic...
        try:

            # Update current plan step
            current_plan = getattr(state, '_current_plan', None)
            if current_plan:
                current_step = current_plan.get_current_step()
                if current_step:
                    current_step.start_execution()
            
            # Step 1: Detect if this is a follow-up query
            state.is_follow_up_query = StateManager.detect_follow_up_query(state)
            logger.info(f"Follow-up query detected: {state.is_follow_up_query}")

            # Log conversation context if follow-up
            if state.is_follow_up_query:
                context = state.get_context_for_follow_up()
                logger.info(f"📚 Using conversation context:")
                logger.info(f"   Last topic: {context['last_topic']}")
                logger.info(f"   Last tables: {context['last_tables']}")
                logger.info(f"   Last summary: {context['last_summary']}")
            
            # Step 2: Suggest relevant tables based on query
            table_suggestions = self._suggest_tables_for_query(state)
            
            # NEW: Check if table suggestion failed
            if not table_suggestions.get("success") or not state.selected_tables:
                logger.error("❌ Table suggestion failed or returned empty")
                
                # Error was already added to state in _suggest_tables_for_query
                if current_plan and current_step:
                    current_step.fail_execution("No tables identified for query")
                
                return state
            
            # Step 3: Get comprehensive schema context
            schema_context = self._get_comprehensive_schema_context(state, table_suggestions)
            
            # Step 4: Analyze relationships between selected tables
            relationship_context = self._analyze_table_relationships(state.selected_tables)
            
            # Step 5: Get business context and intent
            business_context = self._extract_business_context(state)
            
            # Step 6: Update state with all gathered context
            state.schema_context = schema_context
            state.table_relationships = relationship_context
            state.business_intent = business_context.get("intent", "")
            
            # Step 7: Complete the current plan step
            if current_plan and current_step:
                current_step.complete_execution({
                    "selected_tables": state.selected_tables,
                    "schema_context_keys": list(schema_context.keys()),
                    "business_intent": state.business_intent
                })
            
            logger.info(f"✅ Schema inspection complete. Selected tables: {state.selected_tables}")
            return state
            
        except Exception as e:
            error_msg = f"Schema inspection failed: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            
            # Mark current step as failed
            if current_plan and current_step:
                current_step.fail_execution(error_msg)
            
            return state
    
    def _suggest_tables_for_query(self, state: AgentState) -> Dict[str, Any]:
        """
        Enhanced table suggestion with fuzzy matching and error handling
        """
        logger.info("🎯 Suggesting tables for query")
        
        try:
            # Check if this is a follow-up query
            if state.is_follow_up_query and state.last_tables_used:
                logger.info(f"📌 Follow-up detected! Reusing tables from context: {state.last_tables_used}")
                
                state.selected_tables = state.last_tables_used.copy()
                
                return {
                    "success": True,
                    "suggested_tables": {
                        table: {"relevance_score": 1.0, "source": "conversation_context"}
                        for table in state.last_tables_used
                    },
                    "context_used": True,
                    "previous_topic": state.last_query_topic,
                    "method": "context_reuse"
                }
            
            # Not a follow-up - use enhanced table suggestion
            query_lower = state.user_query.lower()
            
            # 1. EXACT KEYWORD MATCHING
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
            
            table_scores = {}
            for table_name, keywords in table_keywords.items():
                score = sum(1 for keyword in keywords if keyword in query_lower)
                if score > 0:
                    table_scores[table_name] = score
            
            # 2. FUZZY MATCHING if no exact matches
            if not table_scores:
                logger.info("🔄 No exact keyword matches, trying fuzzy matching...")
                valid_tables = list(BUSINESS_CONTEXT.keys())
                table_scores = self._fuzzy_match_tables(query_lower, valid_tables)
            
            # 3. EXPLICIT TABLE NAME MENTION
            valid_tables = list(BUSINESS_CONTEXT.keys())
            for table in valid_tables:
                if table.lower() in query_lower:
                    table_scores[table] = table_scores.get(table, 0) + 2.0
            
            # 4. STILL NO MATCHES? Handle error
            if not table_scores:
                logger.warning("⚠️ No tables matched user query")
                
                # Classify as schema error
                error_detail = error_manager.classify_error(
                    error_message=f"Could not identify relevant tables for query: '{state.user_query}'",
                    error_context={
                        "query": state.user_query,
                        "stage": "schema_inspection",
                        "available_tables": valid_tables
                    }
                )
                
                # Get suggestions using error manager
                suggestions = error_manager.suggest_alternatives(
                    error_detail,
                    valid_tables
                )
                
                # Format user-friendly message
                user_message = self._format_no_tables_error(
                    state.user_query,
                    suggestions if suggestions else valid_tables
                )
                
                state.add_error(user_message)
                
                return {
                    "success": False,
                    "error": "no_tables_identified",
                    "error_detail": error_detail,
                    "suggestions": suggestions,
                    "method": "error"
                }
            
            # Sort by relevance
            suggested_tables = sorted(table_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Get top suggestions
            top_suggestions = [table for table, score in suggested_tables[:3]]
            state.selected_tables = top_suggestions
            
            # Build context
            table_contexts = {}
            for table in top_suggestions:
                if table in BUSINESS_CONTEXT:
                    table_contexts[table] = {
                        "description": BUSINESS_CONTEXT[table]["description"],
                        "key_fields": BUSINESS_CONTEXT[table]["key_fields"],
                        "relevance_score": dict(suggested_tables)[table]
                    }
            
            logger.info(f"✅ Suggested tables: {top_suggestions} (scores: {[f'{t}={s:.2f}' for t, s in suggested_tables[:3]]})")
            
            return {
                "success": True,
                "suggested_tables": table_contexts,
                "all_scores": dict(suggested_tables),
                "method": "keyword_and_fuzzy"
            }
            
        except Exception as e:
            logger.error(f"Table suggestion failed: {e}")
            return self._fallback_table_suggestion(state)
    
    def _fuzzy_match_tables(self, query_lower: str, valid_tables: List[str], threshold: float = 0.6) -> Dict[str, float]:
        """
        Fuzzy match table names from query using similarity scoring
        """
        matches = {}
        query_words = re.findall(r'\b\w+\b', query_lower)
        
        for table in valid_tables:
            table_lower = table.lower()
            
            for word in query_words:
                if len(word) >= 4:  # Only check words of reasonable length
                    # Calculate similarity ratio
                    similarity = SequenceMatcher(None, word, table_lower).ratio()
                    
                    if similarity >= threshold:
                        if table not in matches or similarity > matches[table]:
                            matches[table] = similarity
                    
                    # Also check substring matching
                    if word in table_lower or table_lower in word:
                        containment_score = min(len(word), len(table_lower)) / max(len(word), len(table_lower))
                        if containment_score >= threshold:
                            if table not in matches or containment_score > matches[table]:
                                matches[table] = containment_score
        
        logger.info(f"🔍 Fuzzy matching found: {list(matches.keys())}")
        return matches
    
    def _format_no_tables_error(self, query: str, suggestions: List[str]) -> str:
        """
        Format a helpful error message when no tables are found
        """
        message = f"🤔 I couldn't identify which database tables to use for your query: \"{query}\"\n\n"
        
        if suggestions and len(suggestions) <= 5:
            message += "**Did you mean one of these tables?**\n"
            for suggestion in suggestions:
                message += f"  • {suggestion}\n"
        else:
            message += "**Available tables:**\n"
            for table in sorted(suggestions[:10]):  # Show max 10
                message += f"  • {table}\n"
        
        message += "\n💡 **Try asking:**\n"
        message += "  • Show me data from customers\n"
        message += "  • List all orders\n"
        message += "  • Show me products\n"
        
        return message
    
    def _fallback_table_suggestion(self, state: AgentState) -> Dict[str, Any]:
        """
        Fallback table suggestion based on keywords and conversation history
        """
        logger.info("🔄 Using fallback table suggestion")
        
        # Default to core business tables
        core_tables = ["customers", "orders", "order_details", "products"]
        
        # Check conversation history
        if state.is_follow_up_query and len(state.messages) > 2:
            previous_messages = [msg.content for msg in state.messages[:-1]]
            mentioned_tables = []
            
            for table_name in BUSINESS_CONTEXT.keys():
                if any(table_name in msg.lower() for msg in previous_messages):
                    mentioned_tables.append(table_name)
            
            if mentioned_tables:
                state.selected_tables = mentioned_tables
                logger.info(f"Follow-up query - using previously mentioned tables: {mentioned_tables}")
            else:
                state.selected_tables = core_tables[:2]
        else:
            # New query - suggest based on simple keywords
            query_lower = state.user_query.lower()
            suggested = []
            
            if any(word in query_lower for word in ["customer", "client", "company"]):
                suggested.append("customers")
            if any(word in query_lower for word in ["order", "purchase", "sale"]):
                suggested.extend(["orders", "order_details"])
            if any(word in query_lower for word in ["product", "item", "inventory"]):
                suggested.append("products")
            if any(word in query_lower for word in ["category", "type"]):
                suggested.append("categories")
            
            state.selected_tables = suggested if suggested else core_tables[:2]
        
        return {
            "success": True,
            "suggested_tables": {table: {"relevance_score": 1.0} for table in state.selected_tables},
            "method": "fallback"
        }
    
    def _get_comprehensive_schema_context(self, state: AgentState, table_suggestions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive schema context for selected tables
        """
        logger.info("📊 Gathering comprehensive schema context")
        
        schema_context = {
            "database_overview": {},
            "table_contexts": {},
            "business_scenario": {},
            "suggested_examples": []
        }
        
        try:
            # Get database overview
            overview = schema_inspector.get_database_overview()
            if overview["success"]:
                schema_context["database_overview"] = overview
            
            # Get context for each selected table
            if state.selected_tables:
                multi_table_context = schema_inspector.get_multiple_tables_context(state.selected_tables)
                if multi_table_context["success"]:
                    schema_context["table_contexts"] = multi_table_context["tables"]
                    schema_context["inter_table_relationships"] = multi_table_context.get("inter_table_relationships", {})
                    schema_context["suggested_joins"] = multi_table_context.get("suggested_joins", [])
            
            # Get relevant sample queries for learning
            schema_context["suggested_examples"] = self._get_relevant_examples(state)
            
            return schema_context
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive schema context: {e}")
            state.add_warning(f"Could not gather full schema context: {e}")
            return schema_context
    
    def _analyze_table_relationships(self, selected_tables: List[str]) -> Dict[str, Any]:
        """
        Analyze relationships between selected tables
        """
        if not selected_tables or len(selected_tables) < 2:
            return {"relationships": [], "join_patterns": []}
        
        logger.info("🔗 Analyzing table relationships")
        
        relationships = {}
        join_patterns = []
        
        try:
            # Analyze pairwise relationships
            for i, table1 in enumerate(selected_tables):
                for table2 in selected_tables[i+1:]:
                    join_path = get_join_path(table1, table2)
                    if "No known relationship" not in join_path:
                        relationships[f"{table1}_to_{table2}"] = join_path
                        
                        join_patterns.append({
                            "tables": [table1, table2],
                            "relationship": join_path,
                            "complexity": "simple" if "Direct" in join_path else "complex"
                        })
            
            return {
                "relationships": relationships,
                "join_patterns": join_patterns,
                "total_tables": len(selected_tables),
                "can_join": len(join_patterns) > 0
            }
            
        except Exception as e:
            logger.error(f"Relationship analysis failed: {e}")
            return {"relationships": {}, "join_patterns": [], "error": str(e)}
    
    def _extract_business_context(self, state: AgentState) -> Dict[str, Any]:
        """
        Extract business intent and context from the query
        """
        logger.info("💼 Extracting business context")
        
        query_lower = state.user_query.lower()
        
        # Detect business intent patterns
        intent_patterns = {
            "customer_analysis": ["customer", "client", "who", "which customer"],
            "sales_analysis": ["sales", "revenue", "total", "sum", "order", "purchase"],
            "product_analysis": ["product", "item", "inventory", "stock", "category"],
            "reporting": ["report", "show", "list", "display", "get"],
            "aggregation": ["how many", "count", "total", "average", "sum", "max", "min"],
            "comparison": ["compare", "versus", "vs", "difference", "between"],
            "trend_analysis": ["trend", "over time", "monthly", "yearly", "growth"],
            "filtering": ["where", "filter", "specific", "only", "just"]
        }
        
        detected_intents = []
        for intent, keywords in intent_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        
        # Determine primary intent
        primary_intent = detected_intents[0] if detected_intents else "general_query"
        
        # Get complexity estimate
        complexity_indicators = len(detected_intents) + (1 if len(state.selected_tables) > 2 else 0)
        if "join" in query_lower or len(state.selected_tables) > 1:
            complexity_indicators += 1
        
        complexity = "low"
        if complexity_indicators > 2:
            complexity = "medium"
        if complexity_indicators > 4 or any(word in query_lower for word in ["complex", "detailed", "comprehensive"]):
            complexity = "high"
        
        state.query_complexity = complexity
        
        return {
            "intent": primary_intent,
            "detected_intents": detected_intents,
            "complexity": complexity,
            "business_domain": "food_beverage_trading",
            "analysis_type": self._determine_analysis_type(detected_intents)
        }
    
    def _determine_analysis_type(self, intents: List[str]) -> str:
        """Determine the type of analysis needed"""
        if "aggregation" in intents:
            return "aggregation_analysis"
        elif "comparison" in intents:
            return "comparative_analysis"
        elif "trend_analysis" in intents:
            return "temporal_analysis"
        elif any(intent in intents for intent in ["customer_analysis", "sales_analysis", "product_analysis"]):
            return "business_intelligence"
        else:
            return "descriptive_analysis"
    
    def _get_relevant_examples(self, state: AgentState) -> List[Dict[str, Any]]:
        """
        Get relevant few-shot examples based on selected tables and intent
        """
        examples = []
        
        # Add table-specific examples
        for table in state.selected_tables:
            if table == "customers":
                examples.append({
                    "type": "table_specific",
                    "table": table,
                    "natural_language": f"Show me customers from Germany",
                    "complexity": "beginner"
                })
            elif table == "orders":
                examples.append({
                    "type": "table_specific",
                    "table": table,
                    "natural_language": f"How many orders were placed in 1996?",
                    "complexity": "intermediate"
                })
        
        # Add intent-specific examples
        if state.business_intent == "sales_analysis":
            examples.append({
                "type": "intent_specific",
                "intent": "sales_analysis",
                "natural_language": "Calculate total revenue by product",
                "complexity": "advanced"
            })
        
        return examples[:3]
    
    def get_node_info(self) -> Dict[str, Any]:
        """Get information about this node"""
        return {
            "node_name": self.node_name,
            "description": self.description,
            "capabilities": [
                "Table suggestion with fuzzy matching",
                "Error handling for no table matches",
                "Comprehensive schema context gathering",
                "Business intent detection",
                "Relationship analysis",
                "Few-shot example preparation"
            ],
            "outputs": [
                "selected_tables",
                "schema_context",
                "table_relationships",
                "business_intent",
                "query_complexity"
            ]
        }

# Create node instance
schema_inspector_node = SchemaInspectorNode()

__all__ = ["SchemaInspectorNode", "schema_inspector_node"]