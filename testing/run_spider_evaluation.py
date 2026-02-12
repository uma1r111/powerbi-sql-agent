# run_spider_evaluation.py
"""
Script to run Spider benchmark evaluation on IntelliQuery
Usage: python run_spider_evaluation.py
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.spider_evaluator import SpiderEvaluator
from flow.graph import agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run Spider benchmark evaluation"""
    
    print("\n" + "="*70)
    print("🚀 IntelliQuery - Spider Benchmark Evaluation")
    print("="*70 + "\n")
    
    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = SpiderEvaluator(agent=agent)
    
    # Load Spider dataset
    logger.info("Loading Spider dataset...")
    
    # Start with a small sample for testing (10 examples)
    # Change to None to evaluate on full dataset
    SAMPLE_SIZE = 10  # Set to None for full evaluation
    
    try:
        dataset = evaluator.load_spider_dataset(
            split="validation",
            sample_size=SAMPLE_SIZE
        )
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("\n💡 Installing required package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        
        # Try again
        dataset = evaluator.load_spider_dataset(
            split="validation",
            sample_size=SAMPLE_SIZE
        )
    
    # Run evaluation
    logger.info(f"\nEvaluating on {len(dataset)} examples...")
    metrics = evaluator.evaluate_dataset(dataset, max_examples=SAMPLE_SIZE)
    
    # Print summary
    evaluator.print_summary()
    
    # Generate detailed report
    logger.info("\nGenerating detailed report...")
    report = evaluator.generate_report("spider_evaluation_report.json")
    
    print("\n✅ Evaluation complete!")
    print(f"📄 Detailed report saved to: spider_evaluation_report.json")
    
    # Print key metrics
    print("\n📊 Key Metrics Summary:")
    print(f"  • Execution Accuracy: {metrics['execution_accuracy']:.2%}")
    print(f"  • Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}")
    print(f"  • Average Latency: {metrics['average_latency_seconds']:.3f}s")
    
    # Interpretation guide
    print("\n💡 Interpretation Guide:")
    exec_acc = metrics['execution_accuracy']
    if exec_acc >= 0.80:
        print("  🌟 EXCELLENT! Your system achieves research-level performance (>80%)")
    elif exec_acc >= 0.70:
        print("  ✅ GOOD! Solid performance for a student project (70-80%)")
    elif exec_acc >= 0.60:
        print("  📈 MODERATE. Room for improvement (60-70%)")
    else:
        print("  ⚠️  NEEDS WORK. Consider improving your prompt engineering (<60%)")
    
    return metrics


if __name__ == "__main__":
    main()