# evaluation/baseline_comparison.py
"""
Baseline Comparison for IntelliQuery
Compares your agent against standard baselines for research evaluation
"""

import logging
import time
from typing import Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Result from baseline comparison"""
    system_name: str
    exact_match_accuracy: float
    execution_accuracy: float
    average_latency: float
    total_examples: int


class BaselineComparator:
    """
    Compare IntelliQuery against baseline systems
    
    Baselines:
    1. Simple LLM (Gemini) - Zero-shot prompting without your pipeline
    2. Simple rule-based system (keyword matching)
    """
    
    def __init__(self, your_agent):
        self.your_agent = your_agent
        
    def compare_on_dataset(self, dataset, max_examples: int = 50) -> Dict[str, BaselineResult]:
        """
        Run all baselines and your system on the same dataset
        
        Args:
            dataset: Spider validation set
            max_examples: Number of examples to compare
        
        Returns:
            Dict mapping system name to results
        """
        logger.info(f"🏁 Running baseline comparison on {max_examples} examples...")
        
        results = {}
        
        # Your system
        logger.info("\n📊 Evaluating: Your IntelliQuery Agent")
        results['IntelliQuery'] = self._evaluate_system(
            dataset, 
            self.your_agent_wrapper,
            max_examples
        )
        
        # Simple LLM baseline (if available)
        try:
            logger.info("\n📊 Evaluating: Simple LLM Baseline (Gemini)")
            results['Simple-LLM-Baseline'] = self._evaluate_system(
                dataset,
                self.gpt4_baseline_wrapper,  # Still uses this method but with Gemini
                max_examples
            )
        except Exception as e:
            logger.warning(f"Simple LLM baseline skipped: {e}")
            # Continue without this baseline
        
        # Simple rule-based baseline
        logger.info("\n📊 Evaluating: Rule-Based Baseline")
        results['Rule-Based'] = self._evaluate_system(
            dataset,
            self.rule_based_wrapper,
            max_examples
        )
        
        return results
    
    def _evaluate_system(self, dataset, system_wrapper, max_examples: int) -> BaselineResult:
        """Evaluate a single system"""
        exact_matches = 0
        execution_matches = 0
        latencies = []
        
        for idx, example in enumerate(dataset):
            if idx >= max_examples:
                break
            
            question = example['question']
            gold_sql = example['query']
            
            try:
                # Generate SQL
                start = time.time()
                predicted_sql = system_wrapper(question, example)
                latency = time.time() - start
                
                latencies.append(latency)
                
                # Check exact match
                if self._normalize_sql(predicted_sql) == self._normalize_sql(gold_sql):
                    exact_matches += 1
                
                # Check execution match (simplified)
                try:
                    from tools.sql_tools import sql_executor
                    pred_result = sql_executor.execute_query(predicted_sql)
                    gold_result = sql_executor.execute_query(gold_sql)
                    
                    if pred_result.get('success') and gold_result.get('success'):
                        if pred_result.get('data') == gold_result.get('data'):
                            execution_matches += 1
                except:
                    pass
                    
            except Exception as e:
                logger.debug(f"Error in system evaluation: {e}")
                continue
        
        total = min(max_examples, len(dataset))
        
        return BaselineResult(
            system_name="Unknown",
            exact_match_accuracy=exact_matches / total if total > 0 else 0,
            execution_accuracy=execution_matches / total if total > 0 else 0,
            average_latency=sum(latencies) / len(latencies) if latencies else 0,
            total_examples=total
        )
    
    def your_agent_wrapper(self, question: str, example: Dict) -> str:
        """Wrapper for your IntelliQuery agent"""
        result = self.your_agent.process_query_sync(question)
        return result.cleaned_sql or result.generated_sql or ""
    
    def gpt4_baseline_wrapper(self, question: str, example: Dict) -> str:
        """
        LLM baseline using Google Gemini
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            # Initialize simple LLM (no sophisticated pipeline)
            simple_llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0
            )
            
            # Simple prompt without your sophisticated pipeline
            prompt = f"""Convert this natural language question to SQL.
            
Question: {question}

Return ONLY the SQL query, nothing else."""
            
            response = simple_llm.invoke(prompt)
            sql = response.content.strip()
            
            # Clean up response (remove markdown if present)
            if '```sql' in sql:
                sql = sql.split('```sql')[1].split('```')[0].strip()
            elif '```' in sql:
                sql = sql.split('```')[1].strip()
            
            return sql
            
        except Exception as e:
            logger.error(f"Simple LLM baseline failed: {e}")
            return ""
    
    def rule_based_wrapper(self, question: str, example: Dict) -> str:
        """
        Simple rule-based baseline
        Uses keyword matching to generate basic SQL
        """
        question_lower = question.lower()
        
        # Very simple rules
        sql = "SELECT * FROM "
        
        # Detect table
        if "customer" in question_lower:
            sql += "customers"
        elif "order" in question_lower:
            sql += "orders"
        elif "product" in question_lower:
            sql += "products"
        else:
            sql += "table"
        
        # Add WHERE if filtering words detected
        if any(word in question_lower for word in ["where", "with", "from"]):
            sql += " WHERE 1=1"
        
        # Add LIMIT if "top" detected
        if "top" in question_lower:
            import re
            match = re.search(r'top\s+(\d+)', question_lower)
            if match:
                sql += f" LIMIT {match.group(1)}"
        
        return sql
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison"""
        return " ".join(sql.lower().split()).rstrip(';').strip()
    
    def print_comparison_table(self, results: Dict[str, BaselineResult]):
        """Print comparison table for thesis"""
        
        print("\n" + "="*80)
        print("📊 BASELINE COMPARISON RESULTS")
        print("="*80)
        print("\n| Model | Exact Match | Execution Accuracy | Avg Latency (s) |")
        print("|-------|-------------|-------------------|-----------------|")
        
        for system_name, result in results.items():
            print(f"| {system_name:<20} | {result.exact_match_accuracy:>10.1%} | "
                  f"{result.execution_accuracy:>16.1%} | {result.average_latency:>14.3f} |")
        
        print("="*80)
        
        # Analysis
        if 'IntelliQuery' in results:
            your_result = results['IntelliQuery']
            
            print(f"\n💡 Analysis:")
            
            # Compare against baselines
            for baseline_name, baseline_result in results.items():
                if baseline_name == 'IntelliQuery':
                    continue
                
                improvement = (your_result.execution_accuracy - 
                             baseline_result.execution_accuracy) * 100
                
                if improvement > 0:
                    print(f"  ✅ IntelliQuery outperforms {baseline_name} by {improvement:.1f}%")
                else:
                    print(f"  📉 IntelliQuery underperforms {baseline_name} by {abs(improvement):.1f}%")
        
        print("")


# Export
__all__ = ["BaselineComparator", "BaselineResult"]