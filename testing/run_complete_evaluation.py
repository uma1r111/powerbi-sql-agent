# run_complete_evaluation.py
"""
Complete Research-Grade Evaluation Pipeline for IntelliQuery
Runs all evaluations and generates thesis-ready reports

This implements the full evaluation strategy:
1. Spider Benchmark Evaluation
2. Baseline Comparison
3. Error Analysis
4. Report Generation
"""

import sys
import logging
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.spider_evaluator import SpiderEvaluator
from evaluation.baseline_comparison import BaselineComparator
from evaluation.error_analysis import ErrorAnalyzer
from flow.graph import agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveEvaluator:
    """
    Complete evaluation pipeline for research-grade assessment
    """
    
    def __init__(self, agent_instance, output_dir: str = "evaluation_results"):
        self.agent = agent_instance
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this evaluation run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_full_evaluation(self, sample_size: int = 100):
        """
        Run complete evaluation pipeline
        
        Args:
            sample_size: Number of Spider examples to evaluate
        """
        
        print("\n" + "="*80)
        print("🚀 INTELLIQUERY - COMPREHENSIVE EVALUATION PIPELINE")
        print("="*80)
        print(f"\nEvaluation Run: {self.timestamp}")
        print(f"Sample Size: {sample_size} examples")
        print("="*80 + "\n")
        
        results = {}
        
        # ═══════════════════════════════════════════════════════════
        # STEP 1: Spider Benchmark Evaluation
        # ═══════════════════════════════════════════════════════════
        
        print("\n📊 STEP 1: Spider Benchmark Evaluation")
        print("-" * 80)
        
        spider_evaluator = SpiderEvaluator(agent=self.agent)
        
        try:
            # Load dataset
            dataset = spider_evaluator.load_spider_dataset(
                split="validation",
                sample_size=sample_size
            )
            
            # Run evaluation
            spider_metrics = spider_evaluator.evaluate_dataset(
                dataset, 
                max_examples=sample_size
            )
            
            # Print summary
            spider_evaluator.print_summary()
            
            # Save results
            results['spider_evaluation'] = spider_metrics
            results['spider_detailed'] = spider_evaluator.results
            
        except Exception as e:
            logger.error(f"Spider evaluation failed: {e}")
            print(f"\n❌ Spider evaluation failed: {e}")
            print("💡 Make sure to install: pip install datasets")
            spider_metrics = None
        
        # ═══════════════════════════════════════════════════════════
        # STEP 2: Baseline Comparison
        # ═══════════════════════════════════════════════════════════
        
        print("\n📊 STEP 2: Baseline Comparison")
        print("-" * 80)
        
        try:
            comparator = BaselineComparator(your_agent=self.agent)
            
            # Run comparison (use smaller sample for baselines)
            baseline_sample = min(50, sample_size)
            comparison_results = comparator.compare_on_dataset(
                dataset,
                max_examples=baseline_sample
            )
            
            # Print comparison table
            comparator.print_comparison_table(comparison_results)
            
            # Save results
            results['baseline_comparison'] = {
                name: {
                    'exact_match_accuracy': r.exact_match_accuracy,
                    'execution_accuracy': r.execution_accuracy,
                    'average_latency': r.average_latency
                }
                for name, r in comparison_results.items()
            }
            
        except Exception as e:
            logger.error(f"Baseline comparison failed: {e}")
            print(f"\n⚠️  Baseline comparison skipped: {e}")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 3: Error Analysis
        # ═══════════════════════════════════════════════════════════
        
        print("\n📊 STEP 3: Error Analysis")
        print("-" * 80)
        
        if spider_metrics:
            try:
                error_analyzer = ErrorAnalyzer()
                
                # Analyze errors
                error_analysis = error_analyzer.analyze_failures(
                    spider_evaluator.results
                )
                
                # Print analysis
                error_analyzer.print_error_analysis(error_analysis)
                
                # Save results
                results['error_analysis'] = error_analysis
                
            except Exception as e:
                logger.error(f"Error analysis failed: {e}")
                print(f"\n⚠️  Error analysis skipped: {e}")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 4: Generate Reports
        # ═══════════════════════════════════════════════════════════
        
        print("\n📊 STEP 4: Generating Reports")
        print("-" * 80)
        
        # Save comprehensive JSON report
        json_report_path = self.output_dir / f"evaluation_report_{self.timestamp}.json"
        with open(json_report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ JSON report saved: {json_report_path}")
        
        # Generate markdown report for thesis
        md_report_path = self.output_dir / f"evaluation_report_{self.timestamp}.md"
        self._generate_markdown_report(results, md_report_path)
        
        print(f"✅ Markdown report saved: {md_report_path}")
        
        # ═══════════════════════════════════════════════════════════
        # STEP 5: Final Summary
        # ═══════════════════════════════════════════════════════════
        
        self._print_final_summary(results)
        
        return results
    
    def _generate_markdown_report(self, results: dict, output_path: Path):
        """Generate thesis-ready markdown report"""
        
        with open(output_path, 'w') as f:
            f.write("# IntelliQuery Evaluation Report\n\n")
            f.write(f"**Evaluation Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Spider Results
            if 'spider_evaluation' in results:
                spider = results['spider_evaluation']
                
                f.write("## Spider Benchmark Results\n\n")
                f.write(f"- **Total Examples:** {spider['total_examples']}\n")
                f.write(f"- **Exact Match Accuracy:** {spider['exact_match_accuracy']:.2%}\n")
                f.write(f"- **Execution Accuracy:** {spider['execution_accuracy']:.2%}\n")
                f.write(f"- **Average Latency:** {spider['average_latency_seconds']:.3f}s\n\n")
                
                # Component accuracies
                if 'component_accuracies' in spider:
                    f.write("### Component-Level Performance\n\n")
                    for component, acc in spider['component_accuracies'].items():
                        f.write(f"- **{component}:** {acc:.2%}\n")
                    f.write("\n")
            
            # Baseline Comparison
            if 'baseline_comparison' in results:
                f.write("## Baseline Comparison\n\n")
                f.write("| Model | Exact Match | Execution Accuracy | Avg Latency |\n")
                f.write("|-------|-------------|-------------------|-------------|\n")
                
                for name, metrics in results['baseline_comparison'].items():
                    f.write(f"| {name} | {metrics['exact_match_accuracy']:.2%} | "
                           f"{metrics['execution_accuracy']:.2%} | "
                           f"{metrics['average_latency']:.3f}s |\n")
                f.write("\n")
            
            # Error Analysis
            if 'error_analysis' in results:
                error_data = results['error_analysis']
                
                f.write("## Error Analysis\n\n")
                f.write(f"**Total Errors Analyzed:** {error_data['total_errors_categorized']}\n\n")
                
                for category_name, category_data in error_data['categories'].items():
                    if category_data['total'] > 0:
                        f.write(f"### {category_name.replace('_', ' ').title()}\n\n")
                        f.write(f"- **Total:** {category_data['total']} "
                               f"({category_data['percentage']:.1f}%)\n\n")
                        
                        for subcat_name, subcat_data in category_data['subcategories'].items():
                            if subcat_data['count'] > 0:
                                f.write(f"  - {subcat_name.replace('_', ' ').title()}: "
                                       f"{subcat_data['count']} ({subcat_data['percentage']:.1f}%)\n")
                        f.write("\n")
            
            f.write("---\n\n")
            f.write("*Generated by IntelliQuery Evaluation Pipeline*\n")
    
    def _print_final_summary(self, results: dict):
        """Print final summary for terminal"""
        
        print("\n" + "="*80)
        print("✅ EVALUATION COMPLETE - FINAL SUMMARY")
        print("="*80)
        
        if 'spider_evaluation' in results:
            spider = results['spider_evaluation']
            
            print(f"\n📊 Spider Benchmark Performance:")
            print(f"  • Execution Accuracy: {spider['execution_accuracy']:.2%}")
            print(f"  • Exact Match Accuracy: {spider['exact_match_accuracy']:.2%}")
            print(f"  • Average Latency: {spider['average_latency_seconds']:.3f}s")
            
            # Interpretation
            exec_acc = spider['execution_accuracy']
            print(f"\n💡 Performance Interpretation:")
            if exec_acc >= 0.80:
                print("  🌟 EXCELLENT - Research-level performance!")
                print("  Your system achieves >80% execution accuracy.")
                print("  This is publication-worthy for a student project.")
            elif exec_acc >= 0.70:
                print("  ✅ GOOD - Strong performance for FYP")
                print("  Your system achieves 70-80% execution accuracy.")
                print("  This demonstrates effective implementation.")
            elif exec_acc >= 0.60:
                print("  📈 MODERATE - Acceptable with room for improvement")
                print("  Consider improving prompt engineering and schema handling.")
            else:
                print("  ⚠️  NEEDS IMPROVEMENT")
                print("  Focus on improving core SQL generation logic.")
        
        print(f"\n📁 Reports saved to: {self.output_dir}/")
        print("="*80 + "\n")


def main():
    """Main evaluation runner"""
    
    # Configuration
    SAMPLE_SIZE = 100  # Number of Spider examples to evaluate
    
    # Initialize evaluator
    evaluator = ComprehensiveEvaluator(
        agent_instance=agent,
        output_dir="evaluation_results"
    )
    
    # Run full evaluation
    results = evaluator.run_full_evaluation(sample_size=SAMPLE_SIZE)
    
    print("\n✅ All evaluations complete!")
    print(f"📄 Check the 'evaluation_results/' directory for detailed reports")
    
    return results


if __name__ == "__main__":
    main()