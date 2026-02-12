# evaluation/spider_evaluator.py
"""
Spider Benchmark Evaluation for IntelliQuery
Implements research-grade text-to-SQL evaluation following academic standards
"""

import logging
import json
import time
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Single query evaluation result"""
    question: str
    gold_sql: str
    predicted_sql: str
    exact_match: bool
    execution_match: bool
    execution_time: float
    error: str = None
    component_scores: Dict[str, bool] = None


class SpiderEvaluator:
    """
    Spider Benchmark Evaluator for Text-to-SQL Systems
    
    Implements standard metrics:
    - Exact Match Accuracy
    - Execution Accuracy
    - Component Matching
    - Latency
    """
    
    def __init__(self, agent, database_path: str = None):
        """
        Initialize evaluator
        
        Args:
            agent: Your SQL agent instance
            database_path: Path to test database
        """
        self.agent = agent
        self.database_path = database_path
        self.results = []
        
    def load_spider_dataset(self, split: str = "validation", sample_size: int = None):
        """
        Load Spider dataset from HuggingFace
        
        Args:
            split: 'train', 'validation', or 'test'
            sample_size: Number of samples to evaluate (None = all)
        """
        try:
            from datasets import load_dataset
            
            logger.info(f"📥 Loading Spider dataset ({split} split)...")
            dataset = load_dataset("spider", split=split)
            
            if sample_size:
                dataset = dataset.select(range(min(sample_size, len(dataset))))
            
            logger.info(f"✅ Loaded {len(dataset)} examples")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load Spider dataset: {e}")
            raise
    
    def evaluate_dataset(self, dataset, max_examples: int = None) -> Dict[str, Any]:
        """
        Evaluate agent on entire dataset
        
        Args:
            dataset: Spider dataset from HuggingFace
            max_examples: Maximum examples to evaluate (for testing)
        
        Returns:
            Dict with evaluation metrics
        """
        logger.info("🚀 Starting evaluation...")
        
        self.results = []
        total = min(len(dataset), max_examples) if max_examples else len(dataset)
        
        for idx, example in enumerate(dataset):
            if max_examples and idx >= max_examples:
                break
            
            logger.info(f"Evaluating {idx + 1}/{total}: {example['question'][:50]}...")
            
            result = self._evaluate_single_example(example)
            self.results.append(result)
            
            # Progress update
            if (idx + 1) % 10 == 0:
                current_metrics = self._compute_metrics(self.results)
                logger.info(f"Progress: {idx + 1}/{total} | Exec Acc: {current_metrics['execution_accuracy']:.2%}")
        
        # Compute final metrics
        metrics = self._compute_metrics(self.results)
        logger.info(f"\n✅ Evaluation complete!")
        logger.info(f"📊 Results: {metrics}")
        
        return metrics
    
    def _evaluate_single_example(self, example: Dict) -> EvaluationResult:
        """
        Evaluate single example
        
        Args:
            example: Spider dataset example with 'question' and 'query' (gold SQL)
        
        Returns:
            EvaluationResult with metrics
        """
        question = example['question']
        gold_sql = example['query']
        
        try:
            # Generate SQL using your agent
            start_time = time.time()
            agent_result = self.agent.process_query_sync(question)
            execution_time = time.time() - start_time
            
            # Get predicted SQL
            predicted_sql = agent_result.cleaned_sql or agent_result.generated_sql
            
            # Compute exact match
            exact_match = self._check_exact_match(predicted_sql, gold_sql)
            
            # Compute execution match
            execution_match = self._check_execution_match(
                predicted_sql, 
                gold_sql,
                example.get('db_id', 'default')
            )
            
            # Component matching (optional, advanced)
            component_scores = self._compute_component_scores(predicted_sql, gold_sql)
            
            return EvaluationResult(
                question=question,
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                exact_match=exact_match,
                execution_match=execution_match,
                execution_time=execution_time,
                component_scores=component_scores
            )
            
        except Exception as e:
            logger.error(f"Error evaluating example: {e}")
            return EvaluationResult(
                question=question,
                gold_sql=gold_sql,
                predicted_sql="",
                exact_match=False,
                execution_match=False,
                execution_time=0.0,
                error=str(e)
            )
    
    def _check_exact_match(self, predicted: str, gold: str) -> bool:
        """
        Check if predicted SQL exactly matches gold SQL
        (normalized for whitespace and case)
        """
        # Normalize SQL
        pred_normalized = self._normalize_sql(predicted)
        gold_normalized = self._normalize_sql(gold)
        
        return pred_normalized == gold_normalized
    
    def _normalize_sql(self, sql: str) -> str:
        """Normalize SQL for comparison"""
        # Remove extra whitespace
        sql = " ".join(sql.split())
        # Convert to lowercase
        sql = sql.lower()
        # Remove trailing semicolon
        sql = sql.rstrip(';')
        return sql.strip()
    
    def _check_execution_match(self, predicted: str, gold: str, db_id: str) -> bool:
        """
        Check if predicted SQL returns same results as gold SQL
        (This is the MOST IMPORTANT metric)
        """
        try:
            # Execute both queries
            pred_results = self._execute_sql(predicted, db_id)
            gold_results = self._execute_sql(gold, db_id)
            
            # Compare results
            if pred_results is None or gold_results is None:
                return False
            
            return self._results_equal(pred_results, gold_results)
            
        except Exception as e:
            logger.debug(f"Execution comparison failed: {e}")
            return False
    
    def _execute_sql(self, sql: str, db_id: str) -> List[Dict]:
        """Execute SQL query and return results"""
        try:
            # Use your existing SQL executor
            from tools.sql_tools import sql_executor
            
            result = sql_executor.execute_query(sql)
            
            if result.get("success"):
                return result.get("data", [])
            else:
                return None
                
        except Exception as e:
            logger.debug(f"SQL execution failed: {e}")
            return None
    
    def _results_equal(self, results1: List[Dict], results2: List[Dict]) -> bool:
        """Check if two result sets are equal"""
        if len(results1) != len(results2):
            return False
        
        # Convert to comparable format
        df1 = pd.DataFrame(results1) if results1 else pd.DataFrame()
        df2 = pd.DataFrame(results2) if results2 else pd.DataFrame()
        
        # Sort columns
        if not df1.empty and not df2.empty:
            df1 = df1.reindex(sorted(df1.columns), axis=1)
            df2 = df2.reindex(sorted(df2.columns), axis=1)
        
        # Compare
        try:
            return df1.equals(df2)
        except:
            return False
    
    def _compute_component_scores(self, predicted: str, gold: str) -> Dict[str, bool]:
        """
        Compute component-level matching
        Checks: SELECT, FROM, WHERE, JOIN, GROUP BY, ORDER BY
        """
        components = {
            'SELECT': False,
            'FROM': False,
            'WHERE': False,
            'JOIN': False,
            'GROUP_BY': False,
            'ORDER_BY': False
        }
        
        pred_lower = predicted.lower()
        gold_lower = gold.lower()
        
        # Simple component checking (can be made more sophisticated)
        for component in components.keys():
            keyword = component.replace('_', ' ')
            pred_has = keyword.lower() in pred_lower
            gold_has = keyword.lower() in gold_lower
            
            # If both have or both don't have the component, it matches
            components[component] = (pred_has == gold_has)
        
        return components
    
    def _compute_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Compute evaluation metrics from results
        
        Returns standard Spider metrics:
        - Exact Match Accuracy
        - Execution Accuracy
        - Average Latency
        - Component Scores
        """
        total = len(results)
        
        if total == 0:
            return {}
        
        # Count successes
        exact_matches = sum(1 for r in results if r.exact_match)
        execution_matches = sum(1 for r in results if r.execution_match)
        errors = sum(1 for r in results if r.error)
        
        # Compute accuracies
        exact_match_acc = exact_matches / total
        execution_acc = execution_matches / total
        error_rate = errors / total
        
        # Average latency
        valid_times = [r.execution_time for r in results if r.execution_time > 0]
        avg_latency = sum(valid_times) / len(valid_times) if valid_times else 0
        
        # Component accuracies
        component_acc = self._compute_component_accuracies(results)
        
        return {
            'total_examples': total,
            'exact_match_accuracy': exact_match_acc,
            'execution_accuracy': execution_acc,
            'error_rate': error_rate,
            'average_latency_seconds': avg_latency,
            'component_accuracies': component_acc
        }
    
    def _compute_component_accuracies(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """Compute accuracy for each SQL component"""
        component_counts = defaultdict(int)
        component_correct = defaultdict(int)
        
        for result in results:
            if result.component_scores:
                for component, correct in result.component_scores.items():
                    component_counts[component] += 1
                    if correct:
                        component_correct[component] += 1
        
        return {
            component: component_correct[component] / component_counts[component]
            for component in component_counts
            if component_counts[component] > 0
        }
    
    def generate_report(self, output_path: str = "evaluation_report.json"):
        """Generate detailed evaluation report"""
        
        metrics = self._compute_metrics(self.results)
        
        report = {
            'evaluation_date': datetime.now().isoformat(),
            'dataset': 'Spider',
            'total_examples': len(self.results),
            'metrics': metrics,
            'detailed_results': [
                {
                    'question': r.question,
                    'gold_sql': r.gold_sql,
                    'predicted_sql': r.predicted_sql,
                    'exact_match': r.exact_match,
                    'execution_match': r.execution_match,
                    'execution_time': r.execution_time,
                    'error': r.error
                }
                for r in self.results
            ]
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Report saved to {output_path}")
        
        return report
    
    def print_summary(self):
        """Print human-readable summary"""
        metrics = self._compute_metrics(self.results)
        
        print("\n" + "="*60)
        print("📊 SPIDER BENCHMARK EVALUATION RESULTS")
        print("="*60)
        print(f"\n📈 Overall Performance:")
        print(f"  • Total Examples: {metrics['total_examples']}")
        print(f"  • Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
        print(f"  • Execution Accuracy: {metrics['execution_accuracy']:.2%}")
        print(f"  • Error Rate: {metrics['error_rate']:.2%}")
        print(f"  • Average Latency: {metrics['average_latency_seconds']:.3f}s")
        
        print(f"\n🔍 Component Accuracies:")
        for component, acc in metrics['component_accuracies'].items():
            print(f"  • {component}: {acc:.2%}")
        
        print("\n" + "="*60)


# Export
__all__ = ["SpiderEvaluator", "EvaluationResult"]