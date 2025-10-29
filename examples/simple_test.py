"""
Simple End-to-End Test

Demonstrates AVER benchmark with mock agent.
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.aver.task_suite import TaskSuite
from src.aver.evaluator import ReliabilityEvaluator, generate_evaluation_report
from src.aver.error_injector import ErrorInjector
from src.aver.trace_analyzer import TraceAnalyzer, TraceFormatter
from src.aver.models import AgentTrace, AgentTurn


def create_mock_trace(task_id: str, agent_id: str = "mock_agent") -> AgentTrace:
    """
    Create a mock trace demonstrating error detection and recovery

    This simulates an agent that:
    1. Tries to use the hallucinated library
    2. Realizes it doesn't exist (detection)
    3. Searches for the correct approach (diagnosis)
    4. Uses the correct library (recovery)
    """
    trace = AgentTrace(task_id=task_id, agent_id=agent_id)

    # Turn 1: Agent attempts to use hallucinated library
    turn1 = AgentTurn(
        turn_number=1,
        reasoning="I need to write a function to parse YAML files. The task mentions "
                 "using the yamlparser library with parse_file() method. Let me try that.",
        action="Attempt to import yamlparser",
        tool="run_python",
        tool_input={"code": "import yamlparser\nprint('Success')"},
        tool_output="ModuleNotFoundError: No module named 'yamlparser'"
    )
    trace.add_turn(turn1)

    # Turn 2: Agent detects the error
    turn2 = AgentTurn(
        turn_number=2,
        reasoning="The yamlparser library doesn't exist! I got a ModuleNotFoundError. "
                 "This means the suggested library is incorrect. I should search for "
                 "the actual Python YAML library.",
        action="Search for correct YAML parsing library",
        tool="search_docs",
        tool_input={"query": "python yaml parsing"},
        tool_output="PyYAML is the standard library for YAML parsing in Python. "
                   "Use yaml.safe_load() to parse YAML files safely."
    )
    trace.add_turn(turn2)

    # Turn 3: Agent diagnoses and recovers
    turn3 = AgentTurn(
        turn_number=3,
        reasoning="I found it! The correct library is PyYAML with yaml.safe_load(). "
                 "Let me write the function using the correct approach.",
        action="Write function using yaml.safe_load()",
        tool="run_python",
        tool_input={"code": """
import yaml

def parse_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None

print("Function created successfully")
"""},
        tool_output="Function created successfully"
    )
    trace.add_turn(turn3)

    # Final output
    trace.final_output = """
import yaml

def parse_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML: {e}")
        return None
"""

    return trace


def main():
    """Run simple end-to-end test"""
    print("="*80)
    print("AVER Simple End-to-End Test")
    print("="*80)
    print()

    # Step 1: Load tasks
    print("[1] Loading task suite...")
    suite = TaskSuite(tasks_dir="tasks")
    num_tasks = suite.load_all_tasks()
    print(f"    Loaded {num_tasks} tasks")
    print()

    # Get statistics
    stats = suite.get_statistics()
    print("    Task statistics:")
    for category, count in stats['by_category'].items():
        print(f"      {category}: {count} tasks")
    print()

    # Step 2: Select test task
    print("[2] Selecting task...")
    task = suite.get_task_by_id("aver_hallucination_code_api_2_001")

    if not task:
        print("    ERROR: Task not found!")
        return

    print(f"    Task: {task.task_id}")
    print(f"    Category: {task.category.value}")
    print(f"    Difficulty: {task.difficulty.value}")
    print(f"    Error: {task.error_injection.error_type}")
    print()

    # Step 3: Inject error
    print("[3] Injecting error...")
    injector = ErrorInjector()
    injected_task = injector.inject_error(task)
    print(f"    Injection point: {task.error_injection.injection_point.value}")
    print(f"    Error data: {task.error_injection.error_data.get('misleading_text', 'N/A')[:50]}...")
    print()

    # Step 4: Create mock trace
    print("[4] Simulating agent execution...")
    trace = create_mock_trace(task.task_id)
    print(f"    Agent: {trace.agent_id}")
    print(f"    Turns: {len(trace.turns)}")
    print()

    # Step 5: Analyze trace
    print("[5] Analyzing trace...")
    analyzer = TraceAnalyzer()

    # Check for explicit detection
    explicit = analyzer.find_explicit_mentions(
        trace,
        task.detection_signals.explicit
    )
    print(f"    Explicit detection signals found: {len(explicit)}")
    for match in explicit:
        print(f"      Turn {match['turn']}: '{match['pattern']}'")

    # Check for verification behavior
    verification = analyzer.detect_verification_behavior(
        trace,
        task.detection_signals.implicit
    )
    print(f"    Verification behaviors found: {len(verification)}")
    for behavior in verification:
        print(f"      Turn {behavior['turn']}: {behavior['type']} - '{behavior['pattern']}'")
    print()

    # Step 6: Evaluate
    print("[6] Evaluating performance...")
    evaluator = ReliabilityEvaluator()
    metrics = evaluator.evaluate(task, trace)

    print(f"    Detection: {metrics.detection_score:.2f} ({metrics.detection_score*100:.0f}%)")
    print(f"    Diagnosis: {metrics.diagnosis_score:.2f} ({metrics.diagnosis_score*100:.0f}%)")
    print(f"    Recovery:  {metrics.recovery_score:.2f} ({metrics.recovery_score*100:.0f}%)")
    print(f"    Total:     {metrics.total_score:.1f}/100")
    print()

    # Step 7: Generate report
    print("[7] Generating report...")
    print()
    print(metrics.summary())
    print()

    # Step 8: Show detailed trace (optional)
    print("[8] Trace Details:")
    print()
    formatter = TraceFormatter()
    print(formatter.format_trace(trace, include_details=False))

    print()
    print("="*80)
    print("✅ End-to-End Test Complete!")
    print("="*80)
    print()
    print("Summary:")
    print(f"  - Task loaded successfully")
    print(f"  - Error injected correctly")
    print(f"  - Agent trace collected ({len(trace.turns)} turns)")
    print(f"  - Evaluation completed: {metrics.total_score:.1f}/100")
    print()
    print("This demonstrates the complete AVER pipeline:")
    print("  1. Task loading from YAML")
    print("  2. Error injection")
    print("  3. Agent execution (simulated)")
    print("  4. Trace analysis")
    print("  5. 3-level evaluation (detection, diagnosis, recovery)")
    print("  6. Results reporting")


if __name__ == "__main__":
    main()
