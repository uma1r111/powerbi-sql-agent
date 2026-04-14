"""
Query Classifier with Context Awareness
Detects greetings, casual conversation, and handles follow-up queries
"""

import re
from typing import Tuple, Optional, Dict, Any
import random

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

    def is_sql_query(self, query: str, last_query: str = None, last_topic: str = None) -> Tuple[bool, str]:
        """
        Determine if query needs SQL execution
        
        Args:
            query: Current user query
            last_query: Previous query for context
            last_topic: Topic of last query (product, customer, etc.)
        
        Returns:
            (needs_sql, response_type)
        """
        query_lower = query.lower().strip()
        query_clean = re.sub(r'[^\w\s]', '', query_lower)
        
        # Check for specific greetings
        for greeting in self.greeting_responses.keys():
            if query_clean == greeting or query_clean.startswith(greeting + ' '):
                return False, f'greeting:{greeting}'
        
        # Check for thanks
        if any(word in query_lower for word in ['thanks', 'thank you', 'thx']):
            return False, 'thanks'
        
        # Check for help
        if any(phrase in query_lower for phrase in ['help', 'what can you do', 'how do i use']):
            return False, 'help'
        
        # Check for goodbye
        if any(word in query_lower for word in ['bye', 'goodbye', 'see you', 'later']):
            return False, 'goodbye'
        
        # Check for follow-up questions (context-dependent)
        if self._is_follow_up_query(query_lower, last_query, last_topic):
            return True, 'follow_up'
        
        # Check for SQL keywords
        sql_keywords = [
            'show', 'list', 'get', 'find', 'select', 'count', 'sum',
            'total', 'average', 'top', 'bottom', 'worst', 'best',
            'how many', 'what is', 'which', 'where', 'who', 'compare',
            'revenue', 'sales', 'customer', 'product', 'order', 'employee'
        ]
        
        has_sql_keyword = any(keyword in query_lower for keyword in sql_keywords)
        
        if has_sql_keyword:
            return True, 'sql'
        
        # Default: treat as casual
        return False, 'casual'
    
    def _is_follow_up_query(self, query: str, last_query: str, last_topic: str) -> bool:
        """Detect if this is a follow-up query that needs context"""
        if not last_query:
            return False
        
        # Relative terms that suggest follow-up
        follow_up_indicators = [
            'best', 'worst', 'top', 'bottom', 'highest', 'lowest',
            'most', 'least', 'better', 'worse', 'same', 'similar',
            'other', 'another', 'more', 'different'
        ]
        
        # Pronouns that need context
        context_pronouns = ['it', 'them', 'those', 'these', 'that', 'this']
        
        # Check if query is very short and uses contextual words
        words = query.split()
        
        # Very short queries (1-4 words) with contextual indicators
        if len(words) <= 4:
            if any(indicator in query for indicator in follow_up_indicators):
                return True
            if any(pronoun in words for pronoun in context_pronouns):
                return True
        
        return False
    
    def generate_response(self, query: str, response_type: str) -> str:
        """Generate appropriate response for non-SQL queries"""
        
        # Handle specific greetings
        if response_type.startswith('greeting:'):
            greeting_key = response_type.split(':')[1]
            responses = self.greeting_responses.get(greeting_key, self.greeting_responses['hi'])
            return random.choice(responses)
        
        # Handle thanks
        if response_type == 'thanks':
            return random.choice(self.thanks_responses)
        
        # Handle help
        if response_type == 'help':
            return self.help_text
        
        # Handle goodbye
        if response_type == 'goodbye':
            return "Goodbye! Come back anytime you need data insights. Have a great day! 👋"
        
        # Default casual response
        return "I'm here to help with your data! Try asking me about customers, products, orders, or sales."
    
    def expand_follow_up_query(self, query: str, last_query: str, last_topic: str) -> str:
        """
        Expand a follow-up query with context from previous query
        
        Examples:
            Last: "worst performing product"
            Current: "best one" → "best performing product"
            
            Last: "customers from Germany"
            Current: "what about France" → "customers from France"
        """
        query_lower = query.lower()
        
        # If query contains "best" but last had "worst", swap it
        if 'worst' in last_query.lower() and 'best' in query_lower:
            # Replace "worst" with "best" in the last query
            expanded = last_query.lower().replace('worst', 'best')
            print(f"🔄 Follow-up detected: '{query}' → '{expanded}'")
            return expanded
        
        # If query contains "best" but last had "worst", swap it
        if 'best' in last_query.lower() and 'worst' in query_lower:
            expanded = last_query.lower().replace('best', 'worst')
            print(f"🔄 Follow-up detected: '{query}' → '{expanded}'")
            return expanded
        
        # Handle "top" vs "bottom"
        if 'bottom' in last_query.lower() and 'top' in query_lower:
            expanded = last_query.lower().replace('bottom', 'top')
            print(f"🔄 Follow-up detected: '{query}' → '{expanded}'")
            return expanded
        
        if 'top' in last_query.lower() and 'bottom' in query_lower:
            expanded = last_query.lower().replace('top', 'bottom')
            print(f"🔄 Follow-up detected: '{query}' → '{expanded}'")
            return expanded
        
        # Handle country changes: "what about France" when last was about Germany
        countries = ['germany', 'france', 'usa', 'uk', 'denmark', 'spain', 'italy']
        for country in countries:
            if country in query_lower:
                # Find what country was mentioned last
                for old_country in countries:
                    if old_country in last_query.lower():
                        expanded = last_query.lower().replace(old_country, country)
                        print(f"🔄 Follow-up detected: '{query}' → '{expanded}'")
                        return expanded
        
        # If we can't expand it, return original
        return query

# Global instance
classifier = QueryClassifier()