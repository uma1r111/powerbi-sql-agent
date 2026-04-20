import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

# LangChain imports for few-shot learning (from YouTube approach)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings

# Import our state management
from state.agent_state import AgentState
from state.plan_state import ExecutionPlan, PlanManager, StepStatus

# Import our database knowledge
from database.sample_queries import SAMPLE_QUERIES, get_queries_with_tables
from database.relationships import COMMON_JOIN_PATTERNS

from tools.error_manager import error_manager

# Google Gemini API imports
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlannerNode:
    """
    Node responsible for generating execution plans and SQL queries
    Uses schema context from inspector + few-shot learning approach from YouTube
    """
    
    def __init__(self):
        self.node_name = "planner"
        self.description = "Generates execution plans and SQL queries using schema context and few-shot learning"
        
        # Initialize LLM (you'll need to set your Gemini API key)
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        
        # Initialize few-shot example selector
        self._initialize_example_selector()
        
        # Initialize prompts
        self._initialize_prompts()

    def _fetch_database_rules(self) -> str:
        """
        NEW METHOD: Fetch Northwind-specific query rules from MCP.
        Includes fallback to local file if MCP unavailable.
        
        Returns:
            Database-specific rules as formatted string
        """
        try:
            from client.mcp_client import mcp_client
            
            # Fetch rules from MCP server (will use cache after first call)
            rules = mcp_client.get_prompt("northwind_query_rules")
            
            if rules:
                logger.info("✅ Loaded Northwind query rules from MCP")
                return rules
            else:
                logger.warning("⚠️ MCP returned empty rules, using fallback")
                return self._load_fallback_rules()
                
        except Exception as e:
            logger.error(f"❌ Failed to fetch MCP rules: {e}")
            return self._load_fallback_rules()

    def _load_fallback_rules(self) -> str:
        """
        Fallback: Load rules from local file if MCP unavailable.
        """
        from pathlib import Path
        fallback_path = Path(__file__).parent.parent / "config" / "northwind_query_rules_fallback.txt"
        
        if fallback_path.exists():
            with open(fallback_path, "r", encoding="utf-8") as f:
                logger.info("📄 Loaded fallback rules from local file")
                return f.read()
        
        logger.error("❌ No fallback rules found!")
        return ""    
    
    def _initialize_example_selector(self):
        """Initialize the few-shot example selector with our sample queries"""
        logger.info("🎯 Initializing few-shot example selector")
        
        try:
            # Convert our sample queries to few-shot examples
            examples = []
            for category, queries in SAMPLE_QUERIES.items():
                for query_name, query_info in queries.items():
                    examples.append({
                        "input": query_info["natural_language"],
                        "query": query_info["sql"],
                        "complexity": query_info["complexity"],
                        "tables": ", ".join(query_info["tables_involved"])
                    })
            
            # Create vector store for semantic similarity
            vectorstore = Chroma()
            try:
                vectorstore.delete_collection()
            except:
                pass  # Collection might not exist
            
            # Create example selector
            self.example_selector = SemanticSimilarityExampleSelector.from_examples(
                examples,
                FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
                vectorstore,
                k=3,  # Select top 3 most similar examples
                input_keys=["input"],
            )
            
            logger.info(f"✅ Example selector initialized with {len(examples)} examples")
            
        except Exception as e:
            logger.error(f"Failed to initialize example selector: {e}")
            self.example_selector = None
    
    def _initialize_prompts(self):
        """Initialize the prompt templates for SQL generation"""
        
        # Example prompt template (from YouTube approach)
        self.example_prompt = ChatPromptTemplate.from_messages([
            ("human", "{input}\nSQLQuery:"),
            ("ai", "{query}"),
        ])
        
        # Few-shot prompt will be created dynamically based on selected examples
        
        # Main system prompt that combines everything
        self.main_prompt_template = """You are a PostgreSQL expert for a Northwind food & beverage trading company database. 

Given an input question, create a syntactically correct PostgreSQL query. 

CURRENT EXECUTION PLAN:
{current_plan}

DATABASE CONTEXT:
{table_info}

BUSINESS CONTEXT:
{business_context}

TABLE RELATIONSHIPS:
{relationships_info}

Below are similar examples for reference:
{few_shot_examples}

IMPORTANT RULES:
1. Only use tables that exist in the schema
2. Use proper PostgreSQL syntax
3. Include appropriate JOINs when querying multiple tables
4. Consider business logic (e.g., revenue = quantity * unit_price * (1 - discount))
5. Add LIMIT clauses for large result sets
6. Use meaningful aliases for tables
7. Follow the execution plan steps

Generate ONLY the SQL query, no explanations."""
    
    def execute(self, state: AgentState) -> AgentState:
        """
        Main execution method for the planner node
        
        Args:
            state: Current agent state with schema context
            
        Returns:
            Updated agent state with generated plan and SQL query
        """
        logger.info(f"🧠 Planner Node: Generating plan and SQL for '{state.user_query}'")
        
        try:
            # Update current plan step
            current_plan = getattr(state, '_current_plan', None)
            if current_plan:
                current_step = current_plan.get_current_step()
                if current_step:
                    current_step.start_execution()
            
            # Step 1: Select relevant few-shot examples
            selected_examples = self._select_few_shot_examples(state)
            state.few_shot_examples = selected_examples
            
            # Step 2: Generate or update execution plan
            if not hasattr(state, '_current_plan') or state._current_plan is None:
                execution_plan = self._generate_execution_plan(state)
                state._current_plan = execution_plan
            
            # Step 3: Generate SQL query using schema context + few-shot learning
            sql_query = self._generate_sql_query(state)
            state.generated_sql = sql_query
            
            # Step 4: Clean the generated SQL
            cleaned_sql = self._clean_sql_query(sql_query)
            state.cleaned_sql = cleaned_sql
            
            # Step 5: Update plan with generated query
            plan_context = self._create_plan_context(state)
            state.current_plan = plan_context
            
            # Step 6: Complete the current plan step
            if current_plan and current_step:
                current_step.complete_execution({
                    "generated_sql": sql_query,
                    "cleaned_sql": cleaned_sql,
                    "selected_examples_count": len(selected_examples),
                    "plan_updated": True
                })
            
            logger.info(f"✅ Planning complete. Generated SQL: {cleaned_sql[:100]}...")
            return state
            
        except Exception as e:
            error_msg = f"Planning failed: {str(e)}"
            logger.error(error_msg)
            state.add_error(error_msg)
            
            # Mark current step as failed
            if current_plan and current_step:
                current_step.fail_execution(error_msg)
            
            return state
    
    def _select_few_shot_examples(self, state: AgentState) -> List[Dict[str, Any]]:
        """
        Select relevant few-shot examples using semantic similarity
        """
        logger.info("🎲 Selecting few-shot examples")
        
        try:
            if not self.example_selector:
                return self._get_fallback_examples(state)
            
            # Select examples based on user query
            selected = self.example_selector.select_examples({"input": state.user_query})
            
            # Filter examples to prefer those using selected tables
            table_relevant_examples = []
            other_examples = []
            
            for example in selected:
                if any(table in example.get("tables", []) for table in state.selected_tables):
                    table_relevant_examples.append(example)
                else:
                    other_examples.append(example)
            
            # Combine table-relevant first, then others
            final_examples = table_relevant_examples + other_examples
            
            # Limit to top 3 examples
            final_examples = final_examples[:3]
            
            logger.info(f"Selected {len(final_examples)} few-shot examples")
            return final_examples
            
        except Exception as e:
            logger.error(f"Few-shot selection failed: {e}")
            return self._get_fallback_examples(state)
    
    def _get_fallback_examples(self, state: AgentState) -> List[Dict[str, Any]]:
        """Fallback examples when semantic selector fails"""
        
        # Get examples that use our selected tables
        relevant_queries = {}
        for table in state.selected_tables:
            table_queries = get_queries_with_tables([table])
            relevant_queries.update(table_queries)
        
        # Convert to example format
        examples = []
        for query_id, query_info in list(relevant_queries.items())[:3]:
            examples.append({
                "input": query_info["natural_language"],
                "query": query_info["sql"], 
                "complexity": query_info["complexity"],
                "tables": query_info["tables_involved"]
            })
        
        return examples
    
    def _generate_execution_plan(self, state: AgentState) -> ExecutionPlan:
        """Generate detailed execution plan"""
        logger.info("📋 Generating execution plan")
        
        # Determine complexity and create appropriate plan
        if state.query_complexity == "high" or len(state.selected_tables) > 2:
            plan = PlanManager.create_complex_analysis_plan(
                state.user_query, 
                state.selected_tables,
                requires_joins=len(state.selected_tables) > 1
            )
        else:
            plan = PlanManager.create_basic_sql_plan(state.user_query, state.selected_tables)
        
        # Customize plan based on business intent
        if state.business_intent == "sales_analysis":
            plan.add_step("Calculate revenue metrics with discount consideration", "result_analyzer")
        elif state.business_intent == "customer_analysis":
            plan.add_step("Analyze customer segmentation patterns", "result_analyzer")
        
        plan.start_execution()
        return plan
    
    def _generate_sql_query(self, state: AgentState) -> str:
        """
        Generate SQL query with error context if retrying
        """
        logger.info("⚡ Generating SQL query")

        # --- THE "AMNESIA" FIX GOES HERE ---
        # Clear out the old errors so the graph doesn't short-circuit on retry
        if state.needs_correction:
            logger.info("🧹 Clearing previous errors for retry attempt")
            if hasattr(state, 'errors'):
                state.errors = []
            if hasattr(state, 'error_dict'):
                state.error_dict = {}
        # -----------------------------------
        
        try:
            # Prepare context for the prompt
            context = self._prepare_prompt_context(state)
            
            # NEW: Fetch database-specific rules from MCP
            db_rules = self._fetch_database_rules()
            
            # Existing error recovery guidance
            error_guidance = ""
            if state.needs_correction and state.correction_attempts > 0:
                error_guidance = self._get_error_recovery_guidance(state)
                logger.info(f"Including error recovery guidance (attempt {state.correction_attempts})")
            
            # Create few-shot prompt dynamically
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=self.example_prompt,
                examples=state.few_shot_examples,
                input_variables=["input"]
            )
            
            # NEW: Build enhanced prompt with MCP rules at the START
            enhanced_prompt_template = ""
            
            # 1. Database rules go FIRST (sets mental model)
            if db_rules:
                enhanced_prompt_template = db_rules + "\n\n"
            
            # 2. Then the standard prompt
            enhanced_prompt_template += self.main_prompt_template
            
            # 3. Error recovery goes LAST (most recent context)
            if error_guidance:
                enhanced_prompt_template += f"\n\n{error_guidance}"
            
            # Create the main prompt
            final_prompt = ChatPromptTemplate.from_messages([
                ("system", enhanced_prompt_template),
                few_shot_prompt,
                MessagesPlaceholder(variable_name="messages"),
                ("human", "{input}"),
            ])
            
            # Create the chain
            chain = final_prompt | self.llm | StrOutputParser()
            
            # Generate the query
            result = chain.invoke({
                "input": state.user_query,
                "current_plan": context["current_plan"],
                "table_info": context["table_info"],
                "business_context": context["business_context"],
                "relationships_info": context["relationships_info"],
                "few_shot_examples": "",
                "messages": state.get_conversation_context()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            return self._generate_fallback_query(state)

    def _get_error_recovery_guidance(self, state: AgentState) -> str:
        """
        Get error recovery guidance for the LLM based on previous errors.
        
        ENHANCED: Now includes historical error context from MCP to prevent retry loops.
        
        Args:
            state: Current agent state
        
        Returns:
            Formatted error recovery guidance for LLM prompt
        """
        if not state.errors and not hasattr(state, 'error_history'):
            return ""
        
        # Get the most recent error details
        validation_results = state.validation_results
        execution_results = state.execution_results
        
        guidance_parts = ["=" * 70]
        guidance_parts.append("PREVIOUS ATTEMPT FAILED - ERROR RECOVERY")
        guidance_parts.append("=" * 70)
        
        # ========================================================================
        # SECTION 1: IMMEDIATE ERROR (What just happened)
        # ========================================================================
        guidance_parts.append("\n📍 IMMEDIATE ERROR (Current Attempt):")
        
        # Check for validation errors with details
        if validation_results and "error_details" in validation_results:
            for error_detail in validation_results["error_details"]:
                recovery_guidance = error_manager.get_recovery_guidance_for_llm(error_detail)
                guidance_parts.append(recovery_guidance)
        
        # Check for execution errors with details
        elif execution_results and "error_detail" in execution_results:
            error_detail = execution_results["error_detail"]
            recovery_guidance = error_manager.get_recovery_guidance_for_llm(error_detail)
            guidance_parts.append(recovery_guidance)
        
        # Fallback to generic error guidance
        else:
            last_error = state.errors[-1] if state.errors else "Unknown error"
            guidance_parts.append(f"""
    Error: {last_error}

    Please analyze what went wrong and generate a corrected SQL query.
    Pay special attention to:
    - Table names (check they exist in schema)
    - Column names (verify they exist in the tables)
    - SQL syntax (ensure proper structure)
    - Business logic (use correct calculations)
    """)
        
        # ========================================================================
        # SECTION 2: HISTORICAL PATTERN (What keeps failing)
        # ========================================================================
        # NEW: Add MCP error history if available
        if hasattr(state, 'error_history') and state.error_history:
            guidance_parts.append("\n" + "=" * 70)
            guidance_parts.append("📊 HISTORICAL ERROR PATTERN (Recent Failures):")
            guidance_parts.append("=" * 70)
            guidance_parts.append(state.error_history)
            
            # Pattern detection hint
            error_count = state.error_history.count("Error #")
            if error_count >= 3:
                guidance_parts.append("\n⚠️ WARNING: You are stuck in a retry loop!")
                guidance_parts.append("The same type of error is repeating. You MUST:")
                guidance_parts.append("1. Read the FULL error history above")
                guidance_parts.append("2. Identify the PATTERN (not just the last error)")
                guidance_parts.append("3. Try a COMPLETELY DIFFERENT approach")
                guidance_parts.append("4. If uncertain about schema, use get_database_schema tool")
        
        # ========================================================================
        # SECTION 3: RETRY STRATEGY (What to do now)
        # ========================================================================
        guidance_parts.append("\n" + "=" * 70)
        guidance_parts.append("🔧 REQUIRED ACTIONS:")
        guidance_parts.append("=" * 70)
        
        # Determine retry strategy based on error type
        retry_strategy = self._determine_retry_strategy(state)
        guidance_parts.append(retry_strategy)
        
        return "\n".join(guidance_parts)

    def _determine_retry_strategy(self, state: AgentState) -> str:
        """
        Determine the best retry strategy based on error patterns.
        
        Args:
            state: Current agent state
        
        Returns:
            Specific retry instructions for the LLM
        """
        # Check if we have historical errors
        has_history = hasattr(state, 'error_history') and state.error_history
        attempt_num = state.correction_attempts + 1
        
        # Build strategy based on attempt number and error type
        strategy_parts = []
        
        # First retry - simple fix
        if attempt_num == 1:
            strategy_parts.append("1. Review the immediate error above")
            strategy_parts.append("2. Apply the suggested fix")
            strategy_parts.append("3. Verify table/column names against schema")
        
        # Second retry - deeper analysis
        elif attempt_num == 2:
            strategy_parts.append("1. The simple fix didn't work - review your approach")
            strategy_parts.append("2. Compare your query structure to the few-shot examples")
            strategy_parts.append("3. Verify you're using correct JOINs and relationships")
            
            if has_history:
                strategy_parts.append("4. Check the error history - are you making the same mistake twice?")
        
        # Third retry - complete rethink
        else:
            strategy_parts.append("⚠️ THIS IS YOUR LAST ATTEMPT!")
            strategy_parts.append("1. STOP and read the FULL error history")
            strategy_parts.append("2. Your approach is fundamentally wrong - start from scratch")
            strategy_parts.append("3. Break down the user's question into simpler parts")
            strategy_parts.append("4. Use get_database_schema tool to verify exact schema")
            strategy_parts.append("5. Generate a MINIMAL query first, then build up")
            
            if has_history:
                strategy_parts.append("6. The pattern in your errors shows what NOT to do - avoid it!")
        
        # Add context-specific hints
        if state.execution_results and "error" in state.execution_results:
            error_msg = state.execution_results["error"].lower()
            
            # Column errors
            if "column" in error_msg and "does not exist" in error_msg:
                strategy_parts.append("\n💡 HINT: Column name error detected")
                strategy_parts.append("   - Use EXACT column names from schema")
                strategy_parts.append("   - Common mistake: 'country' vs 'ship_country'")
                strategy_parts.append("   - When in doubt: call get_database_schema tool")
            
            # Table errors
            elif "relation" in error_msg or "table" in error_msg:
                strategy_parts.append("\n💡 HINT: Table name error detected")
                strategy_parts.append("   - Valid tables: customers, orders, order_details, products, etc.")
                strategy_parts.append("   - Check for typos in table names")
            
            # Syntax errors
            elif "syntax" in error_msg:
                strategy_parts.append("\n💡 HINT: SQL syntax error detected")
                strategy_parts.append("   - Check parentheses balance")
                strategy_parts.append("   - Verify comma placement")
                strategy_parts.append("   - Use PostgreSQL syntax (not MySQL/SQLite)")
        
        return "\n".join(strategy_parts)

    
    def _prepare_prompt_context(self, state: AgentState) -> Dict[str, str]:
        """Prepare context for SQL generation prompt"""
        
        # Current plan context
        current_plan = getattr(state, '_current_plan', None)
        plan_context = current_plan.to_system_prompt_format() if current_plan else "No execution plan available"
        
        # Table information
        table_info = self._format_table_info(state)
        
        # Business context
        business_context = f"""
Business Intent: {state.business_intent}
Query Complexity: {state.query_complexity}
Selected Tables: {', '.join(state.selected_tables)}
Analysis Type: {state.schema_context.get('business_scenario', {}).get('business_model', 'Unknown')}
"""
        
        # Relationships information
        relationships_info = self._format_relationships_info(state)
        
        return {
            "current_plan": plan_context,
            "table_info": table_info,
            "business_context": business_context,
            "relationships_info": relationships_info
        }
    
    def _format_table_info(self, state: AgentState) -> str:
        """Format table information for the prompt"""
        info_parts = []
        
        for table in state.selected_tables:
            if table in state.schema_context.get("table_contexts", {}):
                table_context = state.schema_context["table_contexts"][table]
                info_parts.append(f"""
Table: {table}
Description: {table_context.get('description', 'No description')}
Key Fields: {', '.join(table_context.get('key_fields', []))}
Row Count: {table_context.get('row_count', 'Unknown')}
""")
        
        return "\n".join(info_parts)
    
    def _format_relationships_info(self, state: AgentState) -> str:
        """Format relationship information for the prompt"""
        if not state.table_relationships.get("join_patterns"):
            return "No relationships identified between selected tables."
        
        info_parts = []
        for pattern in state.table_relationships["join_patterns"]:
            info_parts.append(f"- {' and '.join(pattern['tables'])}: {pattern['relationship']}")
        
        return "Table Relationships:\n" + "\n".join(info_parts)
    
    def _clean_sql_query(self, query: str) -> str:
        """
        Clean SQL query using the regex approach from YouTube code
        """
        if not query:
            return ""
        
        # Remove code block syntax and SQL tags
        block_pattern = r"```(?:sql|SQL|SQLQuery|postgresql)?\s*(.*?)\s*```"
        query = re.sub(block_pattern, r"\1", query, flags=re.DOTALL)
        
        # Handle "SQLQuery:" prefix
        prefix_pattern = r"^(?:SQL\s*Query|SQLQuery|PostgreSQL|SQL)\s*:\s*"
        query = re.sub(prefix_pattern, "", query, flags=re.IGNORECASE)
        
        # Extract the first SQL statement
        sql_statement_pattern = r"(SELECT.*?;)"
        sql_match = re.search(sql_statement_pattern, query, flags=re.IGNORECASE | re.DOTALL)
        if sql_match:
            query = sql_match.group(1)
        
        # Remove backticks around identifiers  
        query = re.sub(r'`([^`]*)`', r'\1', query)
        
        # Normalize whitespace
        query = re.sub(r'\s+', ' ', query)
        
        # Format SQL keywords for readability
        keywords = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'HAVING', 'ORDER BY',
                   'LIMIT', 'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN']
        
        pattern = '|'.join(r'\b{}\b'.format(k) for k in keywords)
        query = re.sub(f'({pattern})', r'\n\1', query, flags=re.IGNORECASE)
        
        # Final cleanup
        query = query.strip()
        query = re.sub(r'\n\s*\n', '\n', query)
        
        return query
    
    def _generate_fallback_query(self, state: AgentState) -> str:
        """Generate a simple fallback query when main generation fails"""
        if not state.selected_tables:
            return "SELECT 1 as fallback_query;"
        
        # Simple SELECT from first table
        main_table = state.selected_tables[0]
        return f"SELECT * FROM {main_table} LIMIT 10;"
    
    def _create_plan_context(self, state: AgentState) -> str:
        """Create plan context string for injection into future prompts"""
        current_plan = getattr(state, '_current_plan', None)
        if not current_plan:
            return f"Simple query plan: Process '{state.user_query}' using tables: {', '.join(state.selected_tables)}"
        
        return current_plan.to_system_prompt_format()
    
    def get_node_info(self) -> Dict[str, Any]:
        """Get information about this node"""
        return {
            "node_name": self.node_name,
            "description": self.description,
            "capabilities": [
                "Dynamic few-shot example selection using semantic similarity",
                "SQL query generation with business context",
                "Execution plan creation and management",
                "Schema-aware query planning",
                "Conversation history integration",
                "Query complexity assessment"
            ],
            "inputs": [
                "user_query",
                "schema_context",
                "selected_tables", 
                "business_intent",
                "conversation_history"
            ],
            "outputs": [
                "generated_sql",
                "cleaned_sql",
                "execution_plan",
                "few_shot_examples",
                "current_plan"
            ],
            "dependencies": [
                "OpenAI API for LLM",
                "ChromaDB for vector storage",
                "Schema context from inspector node"
            ]
        }

# Create node instance for easy import
planner_node = PlannerNode()

# Export the node
__all__ = ["PlannerNode", "planner_node"]