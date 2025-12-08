"""
AVER CLI - Command Line Interface

Run AVER benchmark similar to AgentBeats tutorial:
  uv run aver-run scenarios/aver/scenario.toml
"""

import asyncio
import sys
import argparse
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older Python

from .green_agent import AVERGreenAgent
from .task_suite import TaskSuite
from .evaluator import generate_evaluation_report


def load_scenario_config(scenario_path: str) -> dict:
    """
    Load scenario configuration from TOML file

    Args:
        scenario_path: Path to scenario.toml

    Returns:
        Configuration dictionary
    """
    with open(scenario_path, 'rb') as f:
        config = tomllib.load(f)
    return config


async def run_scenario(scenario_path: str):
    """
    Run AVER scenario from TOML configuration

    Args:
        scenario_path: Path to scenario.toml file
    """
    print("="*80)
    print("AVER BENCHMARK")
    print("="*80)
    print()

    # Load configuration
    print(f"📋 Loading scenario: {scenario_path}")
    config = load_scenario_config(scenario_path)

    scenario_config = config.get('scenario', {})
    task_config = config.get('config', {})
    task_selection = config.get('task_selection', {})

    print(f"   Scenario: {scenario_config.get('name', 'N/A')}")
    print(f"   Version: {scenario_config.get('version', '0.1.0')}")
    print()

    # Initialize Green Agent
    print("🤖 Initializing AVER Green Agent...")
    green_agent = AVERGreenAgent(
        tasks_dir=task_config.get('tasks_dir', 'tasks'),
        results_dir=task_config.get('results_dir', 'results'),
        use_llm_judge=task_config.get('use_llm_judge', False)
    )
    print()

    # Show task statistics
    stats = green_agent.task_suite.get_statistics()
    print("📊 Task Suite Statistics:")
    print(f"   Total tasks: {stats['total_tasks']}")
    print(f"   Categories: {', '.join([f'{k}={v}' for k, v in stats['by_category'].items() if v > 0])}")
    print()

    # Get participants (purple agents to test)
    participants = config.get('participants', [])
    enabled_participants = [p for p in participants if p.get('enabled', True)]

    if not enabled_participants:
        print("⚠️  No enabled participants found in configuration")
        print("   Using mock agent for testing")
        enabled_participants = [{
            'role': 'mock_test_agent',
            'endpoint': 'mock',
            'model': 'mock',
            'description': 'Mock agent for testing'
        }]

    print(f"🎭 Testing {len(enabled_participants)} agent(s):")
    for participant in enabled_participants:
        print(f"   - {participant['role']}: {participant.get('model', 'N/A')}")
    print()

    # Task selection parameters
    task_id = task_selection.get('task_id', '')
    task_ids = task_selection.get('task_ids', [])  # List of specific task IDs for fair comparison
    category = task_selection.get('category', '')
    difficulty = task_selection.get('difficulty', 0)
    num_tasks = task_selection.get('num_tasks', 1)

    # Select tasks ONCE before agent loop (ensures all agents get same tasks)
    if task_ids:
        # Use specific task IDs for fair multi-model comparison
        selected_tasks = []
        for tid in task_ids:
            task = green_agent.task_suite.get_task_by_id(tid)
            if task:
                selected_tasks.append(task)
            else:
                print(f"   ⚠️ Task not found: {tid}")
        print(f"📋 Using {len(selected_tasks)} specific tasks for fair comparison")
    elif task_id:
        # Single specific task
        task = green_agent.task_suite.get_task_by_id(task_id)
        selected_tasks = [task] if task else []
    else:
        # Random selection - do it once so all agents get same tasks
        from .models import ErrorCategory, DifficultyLevel
        cat_filter = ErrorCategory(category) if category else None
        diff_filter = DifficultyLevel(difficulty) if difficulty else None

        selected_tasks = []
        for _ in range(num_tasks):
            task = green_agent.task_suite.select_random(
                category=cat_filter,
                difficulty=diff_filter
            )
            if task:
                selected_tasks.append(task)
        print(f"📋 Selected {len(selected_tasks)} random tasks (same for all agents)")

    # Run assessment for each participant
    all_results = {}

    for participant in enabled_participants:
        agent_id = participant['role']
        agent_url = participant['endpoint']

        print("-"*80)
        print(f"Testing: {agent_id}")
        print("-"*80)
        print()

        # Use mock agent if endpoint is 'mock'
        if agent_url == 'mock':
            print("   Using universal mock agent (simulated execution)")

            # Create universal mock agent with configurable behavior
            from .mock_agent import UniversalMockPurpleAgent, MockAgentConfig

            mock_config = MockAgentConfig(
                agent_id=agent_id,
                model_name=participant.get('model', 'mock-gpt-4'),
                deterministic=True,  # Reproducible results
                verbose=True
            )
            mock_agent = UniversalMockPurpleAgent(config=mock_config)

            # Use pre-selected tasks (same for all agents)
            tasks = selected_tasks

            # Run each task
            results = []
            for i, task in enumerate(tasks, 1):
                print(f"   Task {i}/{len(tasks)}: {task.task_id}")

                # Inject error
                from .error_injector import ErrorInjector
                injector = ErrorInjector()
                injected_task = injector.inject_error(task)

                # Execute mock agent with FULL TaskScenario (not dict)
                trace = await mock_agent.execute_task(task)

                # Evaluate
                from .evaluator import ReliabilityEvaluator
                evaluator = ReliabilityEvaluator()
                metrics = evaluator.evaluate(task, trace)
                results.append(metrics)

                print(f"      Score: {metrics.total_score:.1f}/100")

            all_results[agent_id] = results

        elif agent_url.startswith('llm:'):
            # Real LLM agent via OpenRouter API
            model = agent_url.split(':', 1)[1]  # Extract model after "llm:"
            print(f"   Using LLM agent with model: {model}")

            from .llm_purple_agent import LLMPurpleAgent, LLMAgentConfig

            llm_config = LLMAgentConfig(
                agent_id=agent_id,
                model=model,
                verbose=True
            )

            try:
                llm_agent = LLMPurpleAgent(config=llm_config)
            except ValueError as e:
                print(f"   ❌ Failed to initialize LLM agent: {e}")
                print("   Make sure OPENROUTER_API_KEY is set in .env")
                continue

            # Use pre-selected tasks (same for all agents)
            tasks = selected_tasks

            # Run each task with LLM agent
            results = []
            for i, task in enumerate(tasks, 1):
                print(f"   Task {i}/{len(tasks)}: {task.task_id}")

                # Inject error
                from .error_injector import ErrorInjector
                injector = ErrorInjector()
                injected_task = injector.inject_error(task)

                try:
                    # Execute LLM agent
                    trace = await llm_agent.execute_task(task)

                    # Evaluate
                    from .evaluator import ReliabilityEvaluator
                    evaluator = ReliabilityEvaluator()
                    metrics = evaluator.evaluate(task, trace)
                    results.append(metrics)

                    print(f"      Score: {metrics.total_score:.1f}/100")

                except Exception as e:
                    print(f"      ❌ Task failed: {e}")
                    continue

            all_results[agent_id] = results

        else:
            # Real agent via A2A protocol
            print(f"   Connecting to: {agent_url}")

            # Run assessment using green agent
            results = await green_agent.assess_agent(
                agent_url=agent_url,
                agent_id=agent_id,
                task_id=task_id if task_id else None,
                category=category if category else None,
                difficulty=difficulty if difficulty else None,
                num_tasks=num_tasks
            )

            all_results[agent_id] = results

        print()

    # Generate comparative report with FAIR scoring (competition-ready)
    print("="*80)
    print("AGGREGATE RESULTS (Competition-Fair Scoring)")
    print("="*80)
    print()

    for agent_id, results in all_results.items():
        if results:
            # Separate error tasks from negative controls for FAIR comparison
            error_tasks = [r for r in results if not r.is_negative_control]
            negative_controls = [r for r in results if r.is_negative_control]

            print(f"Agent: {agent_id}")
            print(f"  Total Tasks: {len(results)} ({len(error_tasks)} error + {len(negative_controls)} negative)")
            print()

            # Error Task Metrics (what judges care about)
            if error_tasks:
                avg_detection = sum(r.detection_score for r in error_tasks) / len(error_tasks)
                avg_diagnosis = sum(r.diagnosis_score for r in error_tasks) / len(error_tasks)
                avg_recovery = sum(r.recovery_score for r in error_tasks) / len(error_tasks)
                avg_error_score = sum(r.total_score for r in error_tasks) / len(error_tasks)

                print(f"  📊 ERROR TASK PERFORMANCE (n={len(error_tasks)}):")
                print(f"    Detection:  {avg_detection:.2f} ({avg_detection*100:.0f}%)")
                print(f"    Diagnosis:  {avg_diagnosis:.2f} ({avg_diagnosis*100:.0f}%)")
                print(f"    Recovery:   {avg_recovery:.2f} ({avg_recovery*100:.0f}%)")
                print(f"    Avg Score:  {avg_error_score:.1f}/100")
            else:
                print(f"  📊 ERROR TASK PERFORMANCE: No error tasks")
                avg_error_score = 0.0

            # Negative Control Metrics (false positive rate)
            if negative_controls:
                # False positive = agent detected error when there was none
                false_positives = sum(
                    1 for r in negative_controls
                    if r.detection_details.get('false_positive', False) or r.detection_score > 0.3
                )
                fp_rate = false_positives / len(negative_controls)
                avg_nc_recovery = sum(r.recovery_score for r in negative_controls) / len(negative_controls)

                print(f"  🎯 NEGATIVE CONTROL (n={len(negative_controls)}):")
                print(f"    False Positive Rate: {fp_rate:.0%} ({false_positives}/{len(negative_controls)})")
                print(f"    Task Completion:     {avg_nc_recovery:.2f} ({avg_nc_recovery*100:.0f}%)")
            else:
                print(f"  🎯 NEGATIVE CONTROL: No negative control tasks")
                fp_rate = 0.0

            # Overall score (weighted by task type count for fairness)
            # Competition judges want: Good error detection + low false positives
            if error_tasks and negative_controls:
                # Composite score: Error task score penalized by false positive rate
                overall_score = avg_error_score * (1 - fp_rate * 0.5)  # FP penalty
                print(f"  ⭐ OVERALL SCORE: {overall_score:.1f}/100 (FP-adjusted)")
            elif error_tasks:
                print(f"  ⭐ OVERALL SCORE: {avg_error_score:.1f}/100")
            print()

    # Save results to JSON files
    import json
    import os
    from datetime import datetime

    results_dir = task_config.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)

    saved_files = []
    for agent_id, results in all_results.items():
        if results:
            # Calculate aggregate scores
            error_tasks = [r for r in results if not r.is_negative_control]
            negative_controls = [r for r in results if r.is_negative_control]

            # Build aggregate scores
            aggregate = {}
            if error_tasks:
                aggregate["avg_detection"] = sum(r.detection_score for r in error_tasks) / len(error_tasks)
                aggregate["avg_diagnosis"] = sum(r.diagnosis_score for r in error_tasks) / len(error_tasks)
                aggregate["avg_recovery"] = sum(r.recovery_score for r in error_tasks) / len(error_tasks)
                aggregate["avg_error_score"] = sum(r.total_score for r in error_tasks) / len(error_tasks)
            if negative_controls:
                false_positives = sum(
                    1 for r in negative_controls
                    if r.detection_details.get('false_positive', False) or r.detection_score > 0.3
                )
                aggregate["false_positive_rate"] = false_positives / len(negative_controls)
                aggregate["negative_control_completion"] = sum(r.recovery_score for r in negative_controls) / len(negative_controls)

            # Create result document
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_doc = {
                "agent_id": agent_id,
                "timestamp": datetime.now().isoformat(),
                "num_tasks": len(results),
                "num_error_tasks": len(error_tasks),
                "num_negative_controls": len(negative_controls),
                "results": [r.to_dict() for r in results],
                "aggregate_scores": aggregate
            }

            # Save to file
            filename = f"{agent_id}_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(result_doc, f, indent=2, default=str)
            saved_files.append(filepath)

    print("="*80)
    print("✅ AVER Assessment Complete!")
    print()
    if saved_files:
        print(f"Results saved to {len(saved_files)} file(s):")
        for f in saved_files:
            print(f"  - {f}")
    else:
        print("No results to save.")
    print()


def main():
    """
    Main CLI entry point

    Usage:
        uv run aver-run scenarios/aver/scenario.toml

    Or with arguments:
        uv run aver-run scenarios/aver/scenario.toml --verbose
    """
    parser = argparse.ArgumentParser(
        description="AVER Benchmark - Agent Verification & Error Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run aver-run scenarios/aver/scenario.toml
  uv run aver-run scenarios/aver/scenario.toml --verbose
        """
    )

    parser.add_argument(
        'scenario',
        type=str,
        help='Path to scenario TOML file'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    # Check if scenario file exists
    if not Path(args.scenario).exists():
        print(f"❌ Error: Scenario file not found: {args.scenario}")
        print()
        print("Usage: uv run aver-run scenarios/aver/scenario.toml")
        sys.exit(1)

    # Run scenario
    try:
        asyncio.run(run_scenario(args.scenario))
    except KeyboardInterrupt:
        print("\n\n⚠️  Assessment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
