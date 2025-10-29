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

from .green_agent import AVERGreenAgent, MockPurpleAgent
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
    category = task_selection.get('category', '')
    difficulty = task_selection.get('difficulty', 0)
    num_tasks = task_selection.get('num_tasks', 1)

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
            print("   Using mock agent (simulated execution)")

            # Create mock agent
            mock_agent = MockPurpleAgent(agent_id=agent_id)

            # Select tasks
            if task_id:
                task = green_agent.task_suite.get_task_by_id(task_id)
                tasks = [task] if task else []
            else:
                from .models import ErrorCategory, DifficultyLevel
                cat_filter = ErrorCategory(category) if category else None
                diff_filter = DifficultyLevel(difficulty) if difficulty else None

                tasks = []
                for _ in range(num_tasks):
                    task = green_agent.task_suite.select_random(
                        category=cat_filter,
                        difficulty=diff_filter
                    )
                    if task:
                        tasks.append(task)

            # Run each task
            results = []
            for i, task in enumerate(tasks, 1):
                print(f"   Task {i}/{len(tasks)}: {task.task_id}")

                # Inject error
                from .error_injector import ErrorInjector
                injector = ErrorInjector()
                injected_task = injector.inject_error(task)

                # Execute mock agent
                trace = await mock_agent.execute_task({
                    "task_id": task.task_id,
                    "description": task.task_description,
                    "tools": [t.to_dict() for t in task.tools]
                })

                # Evaluate
                from .evaluator import ReliabilityEvaluator
                evaluator = ReliabilityEvaluator()
                metrics = evaluator.evaluate(task, trace)
                results.append(metrics)

                print(f"      Score: {metrics.total_score:.1f}/100")

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

    # Generate comparative report
    print("="*80)
    print("AGGREGATE RESULTS")
    print("="*80)
    print()

    for agent_id, results in all_results.items():
        if results:
            avg_score = sum(r.total_score for r in results) / len(results)
            avg_detection = sum(r.detection_score for r in results) / len(results)
            avg_diagnosis = sum(r.diagnosis_score for r in results) / len(results)
            avg_recovery = sum(r.recovery_score for r in results) / len(results)

            print(f"Agent: {agent_id}")
            print(f"  Tasks: {len(results)}")
            print(f"  Average Score: {avg_score:.1f}/100")
            print(f"    Detection: {avg_detection:.2f} ({avg_detection*100:.0f}%)")
            print(f"    Diagnosis: {avg_diagnosis:.2f} ({avg_diagnosis*100:.0f}%)")
            print(f"    Recovery:  {avg_recovery:.2f} ({avg_recovery*100:.0f}%)")
            print()

    print("="*80)
    print("✅ AVER Assessment Complete!")
    print()
    print(f"Results saved to: {task_config.get('results_dir', 'results')}/")
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
