# query_preprocessor.py
"""
Query Preprocessor for IntelliQuery
Handles abbreviations, typos, and short forms to improve query understanding
"""

import logging
import re
from typing import Dict, List, Tuple
from difflib import get_close_matches

logger = logging.getLogger(__name__)


class QueryPreprocessor:
    """
    Preprocesses user queries to handle:
    - Common abbreviations (qty → quantity)
    - Typos and misspellings
    - Domain-specific shorthand
    - Case variations
    """
    
    def __init__(self):
        # Common business abbreviations
        self.abbreviations = {
            # Quantity/Numbers
            "qty": "quantity",
            "qnty": "quantity",
            "quant": "quantity",
            "amt": "amount",
            "no": "number",
            "num": "number",
            "#": "number",
            
            # Financial
            "rev": "revenue",
            "revs": "revenues",
            "sales": "sales",
            "sal": "sales",
            "tot": "total",
            "totl": "total",
            "avg": "average",
            "min": "minimum",
            "max": "maximum",
            "sum": "total",
            
            # Customer/People
            "cust": "customer",
            "custs": "customers",
            "custmr": "customer",
            "comp": "company",
            "emp": "employee",
            "emps": "employees",
            "mgr": "manager",
            "dept": "department",
            
            # Products
            "prod": "product",
            "prods": "products",
            "inv": "inventory",
            "sku": "product",
            "cat": "category",
            "ctg": "category",
            "ctgry": "category",
            
            # Orders
            "ord": "order",
            "ords": "orders",
            "ordr": "order",
            "ordrs": "orders",
            "shp": "shipment",
            "shpmnt": "shipment",
            "shpng": "shipping",
            "dlvry": "delivery",
            
            # Time
            "yr": "year",
            "yrs": "years",
            "mo": "month",
            "mos": "months",
            "mth": "month",
            "mths": "months",
            "wk": "week",
            "wks": "weeks",
            "dy": "day",
            "dys": "days",
            "qtr": "quarter",
            "qtrs": "quarters",
            
            # Status
            "stat": "status",
            "sts": "status",
            "actv": "active",
            "inactv": "inactive",
            "pndng": "pending",
            "cmpltd": "completed",
            "shppd": "shipped",
            
            # Location
            "addr": "address",
            "st": "street",
            "cty": "city",
            "cntry": "country",
            "rgn": "region",
            "zip": "postal code",
            "ph": "phone",
            
            # Miscellaneous
            "desc": "description",
            "info": "information",
            "req": "required",
            "opt": "optional",
            "avail": "available",
            "unavail": "unavailable",
            "tbd": "to be determined",
            "n/a": "not applicable",
            "asap": "as soon as possible",
        }
        
        # Common typos and variations
        self.typo_corrections = {
            # Common misspellings
            "costumer": "customer",
            "costumers": "customers",
            "recieve": "receive",
            "recieved": "received",
            "revnue": "revenue",
            "reveune": "revenue",
            "quantitiy": "quantity",
            "quantiy": "quantity",
            "seperate": "separate",
            "calender": "calendar",
            "orderd": "ordered",
            "shiped": "shipped",
            "shippd": "shipped",
            
            # Database field variations
            "companyname": "company name",
            "customername": "customer name",
            "productname": "product name",
            "ordernumber": "order number",
            "orderid": "order id",
            "productid": "product id",
            "customerid": "customer id",
            
            # Common query variations
            "top5": "top 5",
            "top10": "top 10",
            "top20": "top 20",
            "1st": "first",
            "2nd": "second",
            "3rd": "third",
        }
        
        # Domain-specific vocabulary (for fuzzy matching)
        self.domain_vocabulary = [
            # Tables
            "customers", "orders", "products", "employees", "suppliers",
            "categories", "shippers", "order_details", "territories",
            
            # Common columns
            "company", "name", "contact", "title", "address", "city",
            "region", "postal", "code", "country", "phone", "fax",
            "quantity", "price", "discount", "total", "revenue",
            "date", "shipped", "required", "freight", "ship",
            
            # Common terms
            "customer", "order", "product", "employee", "supplier",
            "category", "shipper", "territory", "sales", "purchase",
            "inventory", "stock", "unit", "amount", "number",
        ]
        
        # Pattern replacements
        self.pattern_replacements = [
            # Handle "top X" variations
            (r'\btop(\d+)\b', r'top \1'),
            (r'\btop\s+(\d+)\b', r'top \1'),
            
            # Handle date patterns
            (r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b', r'\1-\2-\3'),
            
            # Handle number ranges
            (r'(\d+)-(\d+)', r'\1 to \2'),
            
            # Handle common SQL-like syntax users might use
            (r'\bselect\s+', ''),
            (r'\bfrom\s+', ''),
            (r'\bwhere\s+', 'with '),
        ]
    
    def preprocess(self, query: str) -> Tuple[str, List[str]]:
        """
        Preprocess a user query
        
        Returns:
            Tuple[str, List[str]]: (processed_query, corrections_made)
        """
        original_query = query
        corrections = []
        
        logger.info(f"🔍 Preprocessing query: '{query}'")
        
        # Step 1: Basic cleanup
        query = query.strip()
        
        # Step 2: Expand abbreviations
        query, abbr_corrections = self._expand_abbreviations(query)
        corrections.extend(abbr_corrections)
        
        # Step 3: Fix typos
        query, typo_corrections = self._fix_typos(query)
        corrections.extend(typo_corrections)
        
        # Step 4: Apply pattern replacements
        query = self._apply_patterns(query)
        
        # Step 5: Fuzzy match domain terms
        query, fuzzy_corrections = self._fuzzy_match_terms(query)
        corrections.extend(fuzzy_corrections)
        
        # Step 6: Normalize spacing
        query = self._normalize_spacing(query)
        
        # Log results
        if query != original_query:
            logger.info(f"✅ Preprocessed: '{original_query}' → '{query}'")
            if corrections:
                logger.info(f"📝 Corrections: {corrections}")
        
        return query, corrections
    
    def _expand_abbreviations(self, query: str) -> Tuple[str, List[str]]:
        """Expand common abbreviations"""
        corrections = []
        words = query.split()
        expanded_words = []
        
        for word in words:
            # Check lowercase version for abbreviation
            word_lower = word.lower()
            # Remove punctuation from end
            word_clean = word_lower.rstrip('.,!?;:')
            
            if word_clean in self.abbreviations:
                expansion = self.abbreviations[word_clean]
                # Preserve original case style
                if word.isupper():
                    expansion = expansion.upper()
                elif word[0].isupper():
                    expansion = expansion.capitalize()
                
                expanded_words.append(expansion)
                corrections.append(f"'{word}' → '{expansion}'")
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words), corrections
    
    def _fix_typos(self, query: str) -> Tuple[str, List[str]]:
        """Fix common typos"""
        corrections = []
        original_query = query
        
        for typo, correction in self.typo_corrections.items():
            # Case-insensitive replacement
            pattern = re.compile(re.escape(typo), re.IGNORECASE)
            if pattern.search(query):
                query = pattern.sub(correction, query)
                corrections.append(f"'{typo}' → '{correction}'")
        
        return query, corrections
    
    def _apply_patterns(self, query: str) -> str:
        """Apply regex pattern replacements"""
        for pattern, replacement in self.pattern_replacements:
            query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
        
        return query
    
    def _fuzzy_match_terms(self, query: str) -> Tuple[str, List[str]]:
        """Use fuzzy matching for domain-specific terms"""
        corrections = []
        words = query.split()
        matched_words = []
        
        for word in words:
            word_clean = word.lower().rstrip('.,!?;:')
            
            # Skip very short words or numbers
            if len(word_clean) <= 2 or word_clean.isdigit():
                matched_words.append(word)
                continue
            
            # Check if word is already in vocabulary
            if word_clean in self.domain_vocabulary:
                matched_words.append(word)
                continue
            
            # Try fuzzy matching
            matches = get_close_matches(
                word_clean, 
                self.domain_vocabulary, 
                n=1, 
                cutoff=0.8  # 80% similarity threshold
            )
            
            if matches:
                best_match = matches[0]
                # Preserve original case
                if word[0].isupper():
                    best_match = best_match.capitalize()
                
                matched_words.append(best_match)
                corrections.append(f"'{word}' → '{best_match}' (fuzzy)")
            else:
                matched_words.append(word)
        
        return ' '.join(matched_words), corrections
    
    def _normalize_spacing(self, query: str) -> str:
        """Normalize whitespace and spacing"""
        # Replace multiple spaces with single space
        query = re.sub(r'\s+', ' ', query)
        
        # Fix spacing around punctuation
        query = re.sub(r'\s+([.,!?;:])', r'\1', query)
        query = re.sub(r'([.,!?;:])\s*', r'\1 ', query)
        
        # Remove trailing/leading spaces
        query = query.strip()
        
        return query
    
    def add_custom_abbreviation(self, abbr: str, expansion: str):
        """Add a custom abbreviation at runtime"""
        self.abbreviations[abbr.lower()] = expansion.lower()
        logger.info(f"Added custom abbreviation: '{abbr}' → '{expansion}'")
    
    def add_custom_vocabulary(self, terms: List[str]):
        """Add custom domain vocabulary"""
        self.domain_vocabulary.extend([t.lower() for t in terms])
        logger.info(f"Added {len(terms)} custom vocabulary terms")


class ContextAwarePreprocessor(QueryPreprocessor):
    """
    Extended preprocessor that learns from user's query history
    """
    
    def __init__(self):
        super().__init__()
        self.user_patterns = {}  # Track user-specific patterns
        self.query_history = []  # Recent queries
    
    def learn_from_query(self, original_query: str, corrected_query: str):
        """Learn from successful query corrections"""
        if original_query != corrected_query:
            # Extract patterns
            orig_words = set(original_query.lower().split())
            corr_words = set(corrected_query.lower().split())
            
            # Find new words in correction
            new_words = corr_words - orig_words
            
            for word in new_words:
                if word not in self.domain_vocabulary:
                    self.domain_vocabulary.append(word)
            
            logger.info(f"📚 Learned from query: added {len(new_words)} terms")
    
    def add_to_history(self, query: str):
        """Track query history for context"""
        self.query_history.append(query.lower())
        # Keep last 20 queries
        if len(self.query_history) > 20:
            self.query_history.pop(0)
    
    def get_frequent_terms(self, top_n: int = 10) -> List[str]:
        """Get most frequently used terms from history"""
        from collections import Counter
        
        all_words = []
        for query in self.query_history:
            all_words.extend(query.split())
        
        counter = Counter(all_words)
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        frequent = [(word, count) for word, count in counter.most_common(top_n * 2) 
                   if word not in stop_words]
        
        return [word for word, _ in frequent[:top_n]]


# Initialize global preprocessor
preprocessor = ContextAwarePreprocessor()

# Export
__all__ = ["QueryPreprocessor", "ContextAwarePreprocessor", "preprocessor"]