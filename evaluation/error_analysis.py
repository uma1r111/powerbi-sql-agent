# evaluation/error_analysis.py
"""
Error Analysis Module for IntelliQuery
Provides research-grade error categorization and analysis
"""

import logging
from typing import Dict, List, Any
from collections import defaultdict, Counter
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class ErrorPattern:
    """Represents a categorized error pattern"""
    category: str
    subcategory: str
    count: int
    examples: List[Dict[str, str]]


class ErrorAnalyzer:
    """
    Analyze failures to identify systematic errors
    
    Error Categories (Research Standard):
    1. Schema Errors
       - Wrong table selection
       - Wrong column selection
       - Missing join
    2. Logic Errors
       - Wrong aggregation function
       - Wrong filter condition
       - Missing GROUP BY
    3. Syntax Errors
       - Malformed SQL
       - Missing keywords
    4. Semantic Errors
       - Ambiguous query interpretation
       - Missing implicit constraints
    """
    
    def __init__(self):
        self.error_categories = {
            'SCHEMA_ERROR': {
                'wrong_table': [],
                'wrong_column': [],
                'missing_join': [],
                'extra_join': []
            },
            'LOGIC_ERROR': {
                'wrong_aggregation': [],
                'wrong_filter': [],
                'missing_group_by': [],
                'wrong_order_by': []
            },
            'SYNTAX_ERROR': {
                'malformed_sql': [],
                'missing_keyword': [],
                'invalid_syntax': []
            },
            'SEMANTIC_ERROR': {
                'ambiguous_interpretation': [],
                'missing_constraint': [],
                'over_specification': []
            }
        }
    
    def analyze_failures(self, evaluation_results: List) -> Dict[str, Any]:
        """
        Analyze all failures and categorize them
        
        Args:
            evaluation_results: List of EvaluationResult objects
        
        Returns:
            Dict with error analysis
        """
        logger.info("🔍 Analyzing errors...")
        
        failures = [r for r in evaluation_results if not r.execution_match]
        
        logger.info(f"Found {len(failures)} failures to analyze")
        
        for failure in failures:
            self._categorize_error(failure)
        
        # Compute statistics
        analysis = self._compute_error_statistics()
        
        return analysis
    
    def _categorize_error(self, failure):
        """Categorize a single failure"""
        
        question = failure.question
        gold_sql = failure.gold_sql
        predicted_sql = failure.predicted_sql
        
        # Check for schema errors
        if self._is_wrong_table_error(predicted_sql, gold_sql):
            self.error_categories['SCHEMA_ERROR']['wrong_table'].append({
                'question': question,
                'gold': gold_sql,
                'predicted': predicted_sql
            })
        
        if self._is_wrong_column_error(predicted_sql, gold_sql):
            self.error_categories['SCHEMA_ERROR']['wrong_column'].append({
                'question': question,
                'gold': gold_sql,
                'predicted': predicted_sql
            })
        
        if self._is_missing_join_error(predicted_sql, gold_sql):
            self.error_categories['SCHEMA_ERROR']['missing_join'].append({
                'question': question,
                'gold': gold_sql,
                'predicted': predicted_sql
            })
        
        # Check for logic errors
        if self._is_wrong_aggregation(predicted_sql, gold_sql):
            self.error_categories['LOGIC_ERROR']['wrong_aggregation'].append({
                'question': question,
                'gold': gold_sql,
                'predicted': predicted_sql
            })
        
        if self._is_missing_group_by(predicted_sql, gold_sql):
            self.error_categories['LOGIC_ERROR']['missing_group_by'].append({
                'question': question,
                'gold': gold_sql,
                'predicted': predicted_sql
            })
        
        # Check for syntax errors
        if self._is_syntax_error(predicted_sql):
            self.error_categories['SYNTAX_ERROR']['malformed_sql'].append({
                'question': question,
                'gold': gold_sql,
                'predicted': predicted_sql
            })
    
    def _is_wrong_table_error(self, predicted: str, gold: str) -> bool:
        """Check if wrong table was selected"""
        pred_tables = self._extract_tables(predicted)
        gold_tables = self._extract_tables(gold)
        
        # If tables don't match, it's a wrong table error
        return pred_tables != gold_tables and len(pred_tables) > 0
    
    def _is_wrong_column_error(self, predicted: str, gold: str) -> bool:
        """Check if wrong columns were selected"""
        pred_columns = self._extract_select_columns(predicted)
        gold_columns = self._extract_select_columns(gold)
        
        return pred_columns != gold_columns
    
    def _is_missing_join_error(self, predicted: str, gold: str) -> bool:
        """Check if required join is missing"""
        pred_joins = predicted.lower().count('join')
        gold_joins = gold.lower().count('join')
        
        return gold_joins > pred_joins
    
    def _is_wrong_aggregation(self, predicted: str, gold: str) -> bool:
        """Check if wrong aggregation function used"""
        agg_functions = ['count', 'sum', 'avg', 'max', 'min']
        
        pred_aggs = [agg for agg in agg_functions if agg in predicted.lower()]
        gold_aggs = [agg for agg in agg_functions if agg in gold.lower()]
        
        return pred_aggs != gold_aggs and len(gold_aggs) > 0
    
    def _is_missing_group_by(self, predicted: str, gold: str) -> bool:
        """Check if GROUP BY is missing"""
        has_group_by_gold = 'group by' in gold.lower()
        has_group_by_pred = 'group by' in predicted.lower()
        
        return has_group_by_gold and not has_group_by_pred
    
    def _is_syntax_error(self, sql: str) -> bool:
        """Check for basic syntax errors"""
        if not sql or sql.strip() == "":
            return True
        
        # Check for basic SQL keywords
        sql_lower = sql.lower()
        if 'select' not in sql_lower:
            return True
        
        # Check for unmatched parentheses
        if sql.count('(') != sql.count(')'):
            return True
        
        return False
    
    def _extract_tables(self, sql: str) -> set:
        """Extract table names from SQL"""
        # Simple regex to find table names after FROM and JOIN
        from_pattern = r'from\s+(\w+)'
        join_pattern = r'join\s+(\w+)'
        
        tables = set()
        tables.update(re.findall(from_pattern, sql.lower()))
        tables.update(re.findall(join_pattern, sql.lower()))
        
        return tables
    
    def _extract_select_columns(self, sql: str) -> set:
        """Extract column names from SELECT clause"""
        # Simple extraction (can be improved)
        try:
            select_part = sql.lower().split('from')[0]
            select_part = select_part.replace('select', '').strip()
            
            # Split by comma and clean
            columns = {col.strip() for col in select_part.split(',')}
            return columns
        except:
            return set()
    
    def _compute_error_statistics(self) -> Dict[str, Any]:
        """Compute statistics from categorized errors"""
        
        stats = {
            'total_errors_categorized': 0,
            'categories': {}
        }
        
        for category, subcategories in self.error_categories.items():
            category_stats = {
                'total': 0,
                'subcategories': {}
            }
            
            for subcategory, errors in subcategories.items():
                count = len(errors)
                category_stats['total'] += count
                category_stats['subcategories'][subcategory] = {
                    'count': count,
                    'percentage': 0,  # Will be computed later
                    'examples': errors[:3]  # First 3 examples
                }
            
            stats['categories'][category] = category_stats
            stats['total_errors_categorized'] += category_stats['total']
        
        # Compute percentages
        total = stats['total_errors_categorized']
        if total > 0:
            for category in stats['categories'].values():
                category['percentage'] = (category['total'] / total) * 100
                for subcategory in category['subcategories'].values():
                    subcategory['percentage'] = (subcategory['count'] / total) * 100
        
        return stats
    
    def print_error_analysis(self, analysis: Dict[str, Any]):
        """Print human-readable error analysis"""
        
        print("\n" + "="*80)
        print("🔍 ERROR ANALYSIS")
        print("="*80)
        
        total = analysis['total_errors_categorized']
        print(f"\nTotal Errors Analyzed: {total}")
        
        print("\n📊 Error Distribution by Category:\n")
        
        for category_name, category_data in analysis['categories'].items():
            if category_data['total'] == 0:
                continue
            
            print(f"\n{category_name.replace('_', ' ').title()}: "
                  f"{category_data['total']} ({category_data['percentage']:.1f}%)")
            
            for subcategory_name, subcategory_data in category_data['subcategories'].items():
                if subcategory_data['count'] == 0:
                    continue
                
                print(f"  • {subcategory_name.replace('_', ' ').title()}: "
                      f"{subcategory_data['count']} ({subcategory_data['percentage']:.1f}%)")
                
                # Show example
                if subcategory_data['examples']:
                    example = subcategory_data['examples'][0]
                    print(f"    Example: {example['question'][:60]}...")
        
        print("\n" + "="*80)
        
        # Recommendations
        self._print_recommendations(analysis)
    
    def _print_recommendations(self, analysis: Dict[str, Any]):
        """Print improvement recommendations based on error analysis"""
        
        print("\n💡 IMPROVEMENT RECOMMENDATIONS:\n")
        
        categories = analysis['categories']
        
        # Find top error category
        top_category = max(categories.items(), 
                          key=lambda x: x[1]['total'])
        
        category_name = top_category[0]
        
        recommendations = {
            'SCHEMA_ERROR': [
                "Improve schema inspection prompts",
                "Add more context about table relationships",
                "Enhance few-shot examples with schema information"
            ],
            'LOGIC_ERROR': [
                "Review aggregation function selection logic",
                "Improve WHERE clause generation",
                "Add more examples with GROUP BY patterns"
            ],
            'SYNTAX_ERROR': [
                "Add SQL syntax validation before execution",
                "Improve prompt engineering for SQL generation",
                "Add more structured output formatting"
            ],
            'SEMANTIC_ERROR': [
                "Add clarification questions for ambiguous queries",
                "Improve business context understanding",
                "Add implicit constraint detection"
            ]
        }
        
        if category_name in recommendations:
            print(f"Top Issue: {category_name.replace('_', ' ').title()}\n")
            for i, rec in enumerate(recommendations[category_name], 1):
                print(f"{i}. {rec}")
        
        print("")


# Export
__all__ = ["ErrorAnalyzer", "ErrorPattern"]