"""
Query Classifier with Context Awareness
Detects greetings, casual conversation, and handles follow-up queries using an LLM Supervisor Node.
"""

import os
import re
import random
from typing import Tuple, Optional, Dict, Any
from langchain_groq import ChatGroq

class QueryClassifier:
    """Classify queries and generate contextual responses"""
    
    def __init__(self):
        # Specific greetings with custom responses
        self.greeting_responses = {
            'hi': [
                "Hi there! I'm IntelliQuery, ready to help you explore your data. What would you like to know?",
                "Hello! How can I help you with your data today?",
                "Hey! What data insights can I help you discover?"
            ],
            'hello': [
                "Hello! I'm IntelliQuery, your AI data assistant. What can I analyze for you?",
                "Hi! Ready to dive into your data. What questions do you have?",
                "Hello there! Let's explore your business data together. What would you like to see?"
            ],
            'hey': [
                "Hey! What data would you like to explore today?",
                "Hi! I'm here to help with your data questions.",
                "Hey there! Ready when you are. What can I look up for you?"
            ],
            'good morning': [
                "Good morning! ☀️ Let's start the day with some data insights. What would you like to know?",
                "Morning! Ready to help you make data-driven decisions today."
            ],
            'good afternoon': [
                "Good afternoon! How can I help you analyze your data?",
                "Afternoon! What business questions can I answer for you?"
            ],
            'good evening': [
                "Good evening! What data insights can I help you with?",
                "Evening! Let's look at your data together."
            ],
            'how are you': [
                "I'm doing great, thanks for asking! 😊 I'm ready to help you explore your data. What would you like to know?",
                "I'm doing well! Excited to help you find insights in your data. What can I look up for you?",
                "Doing fantastic! All systems running smoothly and ready to answer your data questions!"
            ],
            'whats up': [
                "Not much, just ready to analyze some data! What are you curious about?",
                "Ready to help you with data insights! What's on your mind?",
                "All set to help! What data questions do you have?"
            ]
        }
        
        # Thank you responses
        self.thanks_responses = [
            "You're welcome! Happy to help! 😊",
            "Anytime! Let me know if you need anything else.",
            "Glad I could help! Feel free to ask more questions.",
            "My pleasure! What else can I assist you with?"
        ]
        
        # Help responses
        self.help_text = """I'm IntelliQuery, your AI business intelligence assistant! Here's what I can do:

📊 **Answer data questions naturally:**
   • "Show me top 10 customers"
   • "What are monthly sales trends?"
   • "Which products are low in stock?"
   • "Compare revenue between Germany and USA"

💡 **Smart features:**
   • I understand abbreviations (qty, rev, cust, etc.)
   • I remember context from our conversation
   • I can handle complex multi-step queries

🔍 **Just ask naturally - I'll:**
   • Generate the SQL automatically
   • Show you the results in a clean table
   • Provide insights and summaries

Try asking me anything about your customers, orders, products, or sales!"""

        # Initialize the LLM Router Supervisor
        # Using llama-3-8b-8192 because it is lightning fast for binary classification
        self.router_llm = ChatGroq(
            temperature=0, 
            model_name="llama3-8b-8192", 
            api_key=os.getenv("GROQ_API_KEY")
        )

    def is_sql_query(self, query: str, last_query: str = None, last_topic: str = None) -> Tuple[bool, str]:
        """
        Determine if query needs SQL execution using LLM Supervisor with Regex Fallback
        """
        query_lower = query.lower().strip()
        
        # 1. Quick catch for obvious greetings (saves API calls and latency)
        query_clean = ''.join(e for e in query_lower if e.isalnum() or e.isspace())
        for greeting in self.greeting_responses.keys():
            if query_clean == greeting or query_clean.startswith(greeting + ' '):
                return False, f'greeting:{greeting}'
                
        if any(word in query_lower for word in ['thanks', 'thank you', 'thx']):
            return False, 'thanks'
        if any(phrase in query_lower for phrase in ['help', 'what can you do', 'how do i use']):
            return False, 'help'
        if any(word in query_lower for word in ['bye', 'goodbye', 'see you', 'later']):
            return False, 'goodbye'

        # 2. LLM SUPERVISOR ROUTER
        prompt = f"""You are a routing supervisor for an AI Database Assistant querying a Northwind sales database.

AVAILABLE TABLES: customers, orders, order_details, products, suppliers, employees, categories

USER QUERY: "{query}"
CONTEXT: Previous query was about: "{last_topic or 'None'}"

TASK: Route this query.
- SQL: Query requires looking up business data (customers, orders, products, sales, revenue, inventory, etc.)
- CASUAL: Small talk, greetings, or meta-questions ("What can you do?", "Thanks!")
- Note: Even conversational phrasing like "are there companies from Japan?" is SQL if it asks for data.

Reply with ONLY ONE WORD: SQL or CASUAL
"""
        try:
            print("🧠 Supervisor Node: Deciding route...")
            decision = self.router_llm.invoke(prompt).content.strip().upper()
            
            if "SQL" in decision:
                print(f"✅ Supervisor Route: SQL (Decision: {decision})")
                return True, 'follow_up'
            else:
                print(f"💬 Supervisor Route: CASUAL (Decision: {decision})")
                return False, 'casual'
                
        except Exception as e:
            print(f"⚠️ LLM Router failed, using fallback: {e}")
            # 3. IMPROVED FALLBACK (If internet drops or API limits hit)
            return self._fallback_is_sql_query(query_lower, last_query, last_topic)

    def _fallback_is_sql_query(self, query_lower: str, last_query: str, last_topic: str) -> Tuple[bool, str]:
        """Improved fallback with interrogative patterns + keyword matching"""
        
        # INTERROGATIVE patterns (common in natural queries)
        interrogative_patterns = [
            r'^(what|which|who|where|when|how many|how much|is there|are there|can you|could you|show|list|get|find)',
            r'(from\s+\w+|in\s+\w+|by\s+\w+)',  # Location/grouping hints
            r'(company|product|order|customer|employee|supplier)',  # Entity mentions
        ]
        
        # Check if ANY interrogative pattern + entity keyword present
        is_interrogative = any(re.search(pattern, query_lower) for pattern in interrogative_patterns)
        
        if is_interrogative:
            # Double-check: does it mention a data entity?
            entities = ['company', 'companies', 'product', 'customer', 'order', 'employee', 'supplier', 'japan', 'germany', 'usa']
            has_entity = any(entity in query_lower for entity in entities)
            if has_entity:
                return True, 'follow_up'  # ← Treat as SQL
        
        # Original keyword logic (as backup)
        sql_keywords = [
            'show', 'list', 'get', 'find', 'select', 'count', 'sum',
            'total', 'average', 'top', 'bottom', 'worst', 'best',
            'how many', 'what is', 'which', 'where', 'who', 'compare',
            'revenue', 'sales', 'customer', 'product', 'order', 'employee',
            'company', 'companies', 'country', 'supplier', 'japan', 'origin'
        ]
        
        has_sql_keyword = any(keyword in query_lower for keyword in sql_keywords)
        
        if has_sql_keyword or self._is_follow_up_query(query_lower, last_query, last_topic):
            return True, 'sql'
        
        return False, 'casual'

    def _is_follow_up_query(self, query: str, last_query: str, last_topic: str) -> bool:
        """Detect if this is a follow-up query that needs context"""
        if not last_query:
            return False
        
        follow_up_indicators = [
            'best', 'worst', 'top', 'bottom', 'highest', 'lowest',
            'most', 'least', 'better', 'worse', 'same', 'similar',
            'other', 'another', 'more', 'different'
        ]
        
        context_pronouns = ['it', 'them', 'those', 'these', 'that', 'this']
        words = query.split()
        
        if len(words) <= 4:
            if any(indicator in query for indicator in follow_up_indicators):
                return True
            if any(pronoun in words for pronoun in context_pronouns):
                return True
        
        return False
    
    def generate_response(self, query: str, response_type: str) -> str:
        """Generate appropriate response for non-SQL queries"""
        if response_type.startswith('greeting:'):
            greeting_key = response_type.split(':')[1]
            responses = self.greeting_responses.get(greeting_key, self.greeting_responses['hi'])
            return random.choice(responses)
        
        if response_type == 'thanks':
            return random.choice(self.thanks_responses)
        
        if response_type == 'help':
            return self.help_text
        
        if response_type == 'goodbye':
            return "Goodbye! Come back anytime you need data insights. Have a great day! 👋"
        
        return "I'm here to help with your data! Try asking me about customers, products, orders, or sales."
    
    def expand_follow_up_query(self, query: str, last_query: str, last_topic: str) -> str:
        """Expand a follow-up query with context from previous query"""
        query_lower = query.lower()
        
        if 'worst' in last_query.lower() and 'best' in query_lower:
            expanded = last_query.lower().replace('worst', 'best')
            print(f"🔄 Follow-up detected: '{query}' → '{expanded}'")
            return expanded
        
        if 'best' in last_query.lower() and 'worst' in query_lower:
            expanded = last_query.lower().replace('best', 'worst')
            print(f"🔄 Follow-up detected: '{query}' → '{expanded}'")
            return expanded
        
        if 'bottom' in last_query.lower() and 'top' in query_lower:
            expanded = last_query.lower().replace('bottom', 'top')
            print(f"🔄 Follow-up detected: '{query}' → '{expanded}'")
            return expanded
        
        if 'top' in last_query.lower() and 'bottom' in query_lower:
            expanded = last_query.lower().replace('top', 'bottom')
            print(f"🔄 Follow-up detected: '{query}' → '{expanded}'")
            return expanded
        
        countries = ['germany', 'france', 'usa', 'uk', 'denmark', 'spain', 'italy', 'japan']
        for country in countries:
            if country in query_lower:
                for old_country in countries:
                    if old_country in last_query.lower():
                        expanded = last_query.lower().replace(old_country, country)
                        print(f"🔄 Follow-up detected: '{query}' → '{expanded}'")
                        return expanded
        
        return query

# Global instance
classifier = QueryClassifier()