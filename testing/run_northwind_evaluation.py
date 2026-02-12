# run_northwind_evaluation.py
"""
Custom Evaluation on Northwind Database
Uses Northwind-specific queries instead of Spider dataset

This is more appropriate for your FYP since Spider uses different database schemas.
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flow.graph import agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NorthwindEvaluator:
    """
    Custom evaluator for Northwind database
    Tests on realistic business queries
    """
    
    def __init__(self):
        self.results = []
        
        # Northwind test queries with expected results
        self.test_queries = [
            {
                "question": "How many customers are in the database?",
                "expected_sql": "SELECT COUNT(*) FROM customers",
                "category": "simple_count"
            },
            {
                "question": "Show all products that are out of stock",
                "expected_sql": "SELECT * FROM products WHERE units_in_stock = 0",
                "category": "filtering"
            },
            {
                "question": "What are the top 5 products by unit price?",
                "expected_sql": "SELECT product_name, unit_price FROM products ORDER BY unit_price DESC LIMIT 5",
                "category": "ordering_limiting"
            },
            {
                "question": "How many orders were placed in 1997?",
                "expected_sql": "SELECT COUNT(*) FROM orders WHERE EXTRACT(YEAR FROM order_date) = 1997",
                "category": "date_filtering"
            },
            {
                "question": "List all customers from Germany",
                "expected_sql": "SELECT * FROM customers WHERE country = 'Germany'",
                "category": "simple_filtering"
            },
            {
                "question": "What is the total revenue from all orders?",
                "expected_sql": "SELECT SUM(unit_price * quantity * (1 - discount)) FROM order_details",
                "category": "aggregation"
            },
            {
                "question": "Show the top 10 customers by total revenue",
                "expected_sql": """SELECT c.customer_id, c.company_name, 
                                  SUM(od.unit_price * od.quantity * (1 - od.discount)) as total_revenue
                                  FROM customers c
                                  JOIN orders o ON c.customer_id = o.customer_id
                                  JOIN order_details od ON o.order_id = od.order_id
                                  GROUP BY c.customer_id, c.company_name
                                  ORDER BY total_revenue DESC
                                  LIMIT 10""",
                "category": "complex_join_aggregation"
            },
            {
                "question": "How many products are in each category?",
                "expected_sql": """SELECT c.category_name, COUNT(p.product_id) as product_count
                                  FROM categories c
                                  LEFT JOIN products p ON c.category_id = p.category_id
                                  GROUP BY c.category_name""",
                "category": "join_group_by"
            },
            {
                "question": "Which employees have processed the most orders?",
                "expected_sql": """SELECT e.employee_id, e.first_name, e.last_name, COUNT(o.order_id) as order_count
                                  FROM employees e
                                  JOIN orders o ON e.employee_id = o.employee_id
                                  GROUP BY e.employee_id, e.first_name, e.last_name
                                  ORDER BY order_count DESC""",
                "category": "join_aggregation"
            },
            {
                "question": "What is the average order value?",
                "expected_sql": """SELECT AVG(order_total) FROM (
                                      SELECT o.order_id, SUM(od.unit_price * od.quantity * (1 - od.discount)) as order_total
                                      FROM orders o
                                      JOIN order_details od ON o.order_id = od.order_id
                                      GROUP BY o.order_id
                                  ) as order_totals""",
                "category": "subquery_aggregation"
            },
            {
                "question": "Show monthly sales trends for 1997",
                "expected_sql": """SELECT EXTRACT(MONTH FROM o.order_date) as month,
                                  SUM(od.unit_price * od.quantity * (1 - od.discount)) as monthly_sales
                                  FROM orders o
                                  JOIN order_details od ON o.order_id = od.order_id
                                  WHERE EXTRACT(YEAR FROM o.order_date) = 1997
                                  GROUP BY month
                                  ORDER BY month""",
                "category": "date_aggregation"
            },
            {
                "question": "List products that need reordering",
                "expected_sql": "SELECT * FROM products WHERE units_in_stock < reorder_level AND discontinued = 0",
                "category": "complex_filtering"
            },
            {
                "question": "What percentage of products are discontinued?",
                "expected_sql": """SELECT 
                                  (COUNT(CASE WHEN discontinued = 1 THEN 1 END)::FLOAT / COUNT(*)::FLOAT * 100) as percentage
                                  FROM products""",
                "category": "conditional_aggregation"
            },
            {
                "question": "Show all orders shipped to France",
                "expected_sql": "SELECT * FROM orders WHERE ship_country = 'France'",
                "category": "simple_filtering"
            },
            {
                "question": "Which supplier provides the most products?",
                "expected_sql": """SELECT s.supplier_id, s.company_name, COUNT(p.product_id) as product_count
                                  FROM suppliers s
                                  JOIN products p ON s.supplier_id = p.supplier_id
                                  GROUP BY s.supplier_id, s.company_name
                                  ORDER BY product_count DESC
                                  LIMIT 1""",
                "category": "join_aggregation_limit"
            }
        ]
    
    def evaluate(self):
        """Run evaluation on all test queries"""
        
        print("\n" + "="*80)
        print("🚀 NORTHWIND DATABASE EVALUATION")
        print("="*80)
        print(f"\nEvaluating on {len(self.test_queries)} business intelligence queries\n")
        
        total = len(self.test_queries)
        successful = 0
        failed = 0
        
        category_stats = {}
        
        for idx, test_case in enumerate(self.test_queries, 1):
            print(f"\n{'─'*80}")
            print(f"Query {idx}/{total}: {test_case['question']}")
            print(f"Category: {test_case['category']}")
            
            # Process query
            result = self._evaluate_single_query(test_case)
            self.results.append(result)
            
            # Update stats
            if result['execution_successful']:
                successful += 1
                status = "✅ SUCCESS"
            else:
                failed += 1
                status = "❌ FAILED"
            
            print(f"Status: {status}")
            print(f"Execution Time: {result['execution_time']:.2f}s")
            
            # Track by category
            category = test_case['category']
            if category not in category_stats:
                category_stats[category] = {'total': 0, 'successful': 0}
            category_stats[category]['total'] += 1
            if result['execution_successful']:
                category_stats[category]['successful'] += 1
        
        # Print summary
        self._print_summary(successful, failed, total, category_stats)
        
        # Save results
        self._save_results()
        
        return {
            'total': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'category_stats': category_stats
        }
    
    def _evaluate_single_query(self, test_case: Dict) -> Dict:
        """Evaluate a single query"""
        import time
        
        start_time = time.time()
        
        try:
            # Process query with agent
            result = agent.process_query_sync(test_case['question'])
            
            execution_time = time.time() - start_time
            
            return {
                'question': test_case['question'],
                'category': test_case['category'],
                'expected_sql': test_case['expected_sql'],
                'generated_sql': result.cleaned_sql or result.generated_sql,
                'execution_successful': result.execution_successful,
                'result_count': result.result_count,
                'execution_time': execution_time,
                'errors': result.errors,
                'warnings': result.warnings
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return {
                'question': test_case['question'],
                'category': test_case['category'],
                'expected_sql': test_case['expected_sql'],
                'generated_sql': "",
                'execution_successful': False,
                'result_count': 0,
                'execution_time': execution_time,
                'errors': [str(e)],
                'warnings': []
            }
    
    def _print_summary(self, successful: int, failed: int, total: int, category_stats: Dict):
        """Print evaluation summary"""
        
        print("\n" + "="*80)
        print("📊 EVALUATION RESULTS")
        print("="*80)
        
        success_rate = (successful / total * 100) if total > 0 else 0
        
        print(f"\n📈 Overall Performance:")
        print(f"  • Total Queries: {total}")
        print(f"  • Successful: {successful} ({success_rate:.1f}%)")
        print(f"  • Failed: {failed} ({100-success_rate:.1f}%)")
        
        # Average execution time
        valid_times = [r['execution_time'] for r in self.results if r['execution_time'] > 0]
        avg_time = sum(valid_times) / len(valid_times) if valid_times else 0
        print(f"  • Average Execution Time: {avg_time:.2f}s")
        
        # Category breakdown
        print(f"\n🔍 Performance by Query Category:")
        for category, stats in sorted(category_stats.items()):
            cat_success_rate = (stats['successful'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"  • {category.replace('_', ' ').title()}: "
                  f"{stats['successful']}/{stats['total']} ({cat_success_rate:.1f}%)")
        
        # Interpretation
        print(f"\n💡 Interpretation:")
        if success_rate >= 80:
            print("  🌟 EXCELLENT! Your system handles Northwind queries very well!")
        elif success_rate >= 70:
            print("  ✅ GOOD! Solid performance on business intelligence queries.")
        elif success_rate >= 60:
            print("  📈 MODERATE. Consider improving error handling.")
        else:
            print("  ⚠️  NEEDS IMPROVEMENT. Review failed query categories.")
        
        print("\n" + "="*80)
    
    def _save_results(self):
        """Save detailed results to JSON"""
        
        output_file = "northwind_evaluation_results.json"
        
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'database': 'Northwind',
            'total_queries': len(self.results),
            'results': self.results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {output_file}")


def main():
    """Main evaluation runner"""
    
    evaluator = NorthwindEvaluator()
    metrics = evaluator.evaluate()
    
    print("\n✅ Evaluation complete!")
    print(f"\n📝 For your thesis, report:")
    print(f"  • Success Rate: {metrics['success_rate']:.1%}")
    print(f"  • Total Queries Tested: {metrics['total']}")
    print(f"  • Database: Northwind (Business Intelligence domain)")
    
    return metrics


if __name__ == "__main__":
    main()