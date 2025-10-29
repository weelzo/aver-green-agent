#!/usr/bin/env python3
"""
Run Baseline Tests on All Models

Tests all 7 models on all 41 tasks with perfect data collection.
Generates all tables needed for research paper.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.aver.green_agent import AVERGreenAgent
from src.aver.results_analyzer import ResultsAnalyzer, create_results_summary
from src.aver.consistency_tester import run_consistency_suite


# Models to test (from user specification)
MODELS_TO_TEST = [
    "google/gemini-2.5-pro",
    "qwen/qwen3-coder",
    "openai/gpt-5",
    "x-ai/grok-code-fast-1",
    "anthropic/claude-sonnet-4.5",
    "z-ai/glm-4.6",
    "minimax/minimax-m2"
]

AGENT_URL = "http://127.0.0.1:8001"  # OpenRouter agent


async def test_single_model(model_name: str, num_tasks: int = 41):
    """
    Test a single model on all tasks

    Args:
        model_name: Model identifier for OpenRouter
        num_tasks: Number of tasks to run (default: all)
    """
    print("="*80)
    print(f"TESTING MODEL: {model_name}")
    print("="*80)
    print()

    # Initialize green agent
    green_agent = AVERGreenAgent(
        tasks_dir="tasks",
        results_dir="results"
    )

    # Run assessment
    print(f"Running {num_tasks} tasks...")
    print(f"Model: {model_name}")
    print(f"Agent URL: {AGENT_URL}")
    print()

    try:
        results = await green_agent.assess_agent(
            agent_url=AGENT_URL,
            agent_id=f"baseline_{model_name.replace('/', '_')}",
            num_tasks=num_tasks
        )

        print(f"\n✅ Completed {len(results)} tasks for {model_name}")
        print(f"Average Score: {mean([r.total_score for r in results]):.1f}/100")

        return results

    except Exception as e:
        print(f"\n❌ Error testing {model_name}: {e}")
        return []


async def test_all_models():
    """
    Test all 7 models sequentially

    NOTE: User must update .env AVER_MODEL and restart purple agent
    between each model test!
    """
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                                                              ║")
    print("║          AVER BASELINE TESTING - ALL MODELS                  ║")
    print("║                                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    print("MODELS TO TEST:")
    for i, model in enumerate(MODELS_TO_TEST, 1):
        print(f"  {i}. {model}")
    print()

    print("IMPORTANT: For each model, you must:")
    print("  1. Update .env: AVER_MODEL={model_name}")
    print("  2. Restart purple agent: python3 scenarios/aver/openrouter_agent.py")
    print("  3. Press Enter here to continue")
    print("  4. This script will run the tests")
    print()

    all_results = {}

    for i, model in enumerate(MODELS_TO_TEST, 1):
        print(f"\n{'='*80}")
        print(f"MODEL {i}/{len(MODELS_TO_TEST)}: {model}")
        print(f"{'='*80}")
        print()

        input(f"Press Enter when purple agent is running with {model}...")

        results = await test_single_model(model)
        all_results[model] = results

        print(f"\n✅ Model {i}/{len(MODELS_TO_TEST)} complete!")

    # Generate analysis
    print("\n" + "="*80)
    print("GENERATING ANALYSIS TABLES")
    print("="*80)
    print()

    analyzer = ResultsAnalyzer()
    analyzer.load_all_results()
    analyzer.generate_all_tables()

    print("\n✅ ALL BASELINE TESTING COMPLETE!")
    print(f"   Tested {len(MODELS_TO_TEST)} models on 41 tasks")
    print(f"   Results saved to: results/")
    print(f"   Tables saved to: paper/tables/")


async def test_with_consistency(model_name: str, sample_size: int = 5):
    """
    Test a model with consistency (pass@3) on sample tasks

    Args:
        model_name: Model to test
        sample_size: Number of tasks to test with k=3
    """
    print("="*80)
    print(f"CONSISTENCY TESTING: {model_name}")
    print("="*80)
    print()

    green_agent = AVERGreenAgent(tasks_dir="tasks", results_dir="results")

    # Select random sample
    from src.aver.models import ErrorCategory
    tasks = []
    for category in ErrorCategory:
        task = green_agent.task_suite.select_random(category=category)
        if task and len(tasks) < sample_size:
            tasks.append(task)

    print(f"Testing {len(tasks)} tasks with k=3 runs each")
    print()

    # Run consistency suite
    consistency_results = await run_consistency_suite(
        green_agent=green_agent,
        agent_url=AGENT_URL,
        agent_id=f"consistency_{model_name.replace('/', '_')}",
        tasks=tasks,
        k=3
    )

    # Save consistency results
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/consistency_{model_name.replace('/', '_')}_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(consistency_results, f, indent=2)

    print(f"\n✅ Consistency results saved to: {filename}")

    return consistency_results


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run AVER baseline tests",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--mode",
        choices=["single", "all", "consistency", "analyze"],
        default="all",
        help="Test mode"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name for single model test"
    )

    parser.add_argument(
        "--tasks",
        type=int,
        default=41,
        help="Number of tasks to run"
    )

    args = parser.parse_args()

    if args.mode == "single":
        if not args.model:
            print("Error: --model required for single mode")
            sys.exit(1)
        asyncio.run(test_single_model(args.model, args.tasks))

    elif args.mode == "all":
        asyncio.run(test_all_models())

    elif args.mode == "consistency":
        if not args.model:
            print("Error: --model required for consistency mode")
            sys.exit(1)
        asyncio.run(test_with_consistency(args.model, sample_size=5))

    elif args.mode == "analyze":
        analyzer = ResultsAnalyzer()
        analyzer.load_all_results()
        analyzer.generate_all_tables()


if __name__ == "__main__":
    main()
