"""
AVER Benchmark - AgentBeats Integration Entry Point

This is the main entry point for AgentBeats controller integration.
The controller calls: python main.py run

Commands:
    run     - Start the AVER green agent server (A2A SDK)
    assess  - Run assessment against a purple agent
    info    - Show benchmark information
"""

import asyncio
import os
import sys
from pathlib import Path

try:
    import typer
except ImportError:
    print("Error: typer not installed. Run: pip install earthshaker")
    sys.exit(1)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.aver.green_agent import AVERGreenAgent
from src.aver.task_suite import TaskSuite


# Create Typer CLI app
app = typer.Typer(
    name="aver",
    help="AVER Benchmark - Agent Verification & Error Recovery",
    add_completion=False
)


@app.command()
def run(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(None, "--port", "-p", help="Port to bind to (default: $PORT or 9000)"),
    tasks_dir: str = typer.Option("tasks", "--tasks-dir", help="Tasks directory"),
    results_dir: str = typer.Option("results", "--results-dir", help="Results directory"),
):
    """
    Start the AVER green agent server using A2A SDK.

    This command is called by agentbeats run_ctrl to start the agent.
    The server uses the official A2A SDK which provides:
    - message/send endpoint for starting assessments
    - tasks/get endpoint for polling task status
    - Proper async task state management
    """
    # Get port from environment variable
    if port is None:
        port = int(os.environ.get("AGENT_PORT", os.environ.get("PORT", 9000)))

    # Determine port source for logging
    if "AGENT_PORT" in os.environ:
        port_source = "$AGENT_PORT (AgentBeats)"
    elif "PORT" in os.environ:
        port_source = "$PORT (Cloud Run)"
    else:
        port_source = "default"

    print(f"Port: {port} (from {port_source})")

    # Use the new A2A SDK server
    from src.aver.server import run_server
    run_server(
        host=host,
        port=port,
        tasks_dir=tasks_dir,
        results_dir=results_dir
    )


@app.command()
def assess(
    agent_url: str = typer.Argument(..., help="URL of the purple agent to test"),
    agent_id: str = typer.Option("test_agent", "--id", "-i", help="Agent identifier"),
    task_id: str = typer.Option(None, "--task", "-t", help="Specific task ID"),
    category: str = typer.Option(None, "--category", "-c", help="Task category filter"),
    difficulty: int = typer.Option(None, "--difficulty", "-d", help="Difficulty filter"),
    num_tasks: int = typer.Option(1, "--num", "-n", help="Number of tasks to run"),
    tasks_dir: str = typer.Option("tasks", "--tasks-dir", help="Tasks directory"),
    results_dir: str = typer.Option("results", "--results-dir", help="Results directory"),
):
    """
    Run AVER assessment against a purple agent.

    Example:
        python main.py assess http://localhost:8080 --id my_agent --num 5
    """
    print("=" * 60)
    print("AVER BENCHMARK - Assessment Mode")
    print("=" * 60)
    print()

    # Initialize green agent
    green_agent = AVERGreenAgent(
        tasks_dir=tasks_dir,
        results_dir=results_dir,
        use_llm_judge=False
    )

    # Run assessment
    async def run_assessment():
        return await green_agent.assess_agent(
            agent_url=agent_url,
            agent_id=agent_id,
            task_id=task_id,
            category=category,
            difficulty=difficulty,
            num_tasks=num_tasks
        )

    results = asyncio.run(run_assessment())

    if results:
        print()
        print("=" * 60)
        print("Assessment Complete!")
        print(f"  Tasks run: {len(results)}")
        print(f"  Avg Score: {sum(r.total_score for r in results) / len(results):.1f}/100")
        print("=" * 60)


@app.command()
def info():
    """
    Show AVER benchmark information.
    """
    print("=" * 60)
    print("AVER BENCHMARK")
    print("Agent Verification & Error Recovery")
    print("=" * 60)
    print()

    # Load tasks
    task_suite = TaskSuite("tasks")
    num_tasks = task_suite.load_all_tasks()
    stats = task_suite.get_statistics()

    print(f"Total Tasks: {num_tasks}")
    print()
    print("Categories:")
    for cat, count in stats['by_category'].items():
        if count > 0:
            print(f"  {cat}: {count} tasks")
    print()
    print("Difficulty Levels:")
    for diff, count in stats['by_difficulty'].items():
        if count > 0:
            print(f"  Level {diff}: {count} tasks")
    print()
    print("Scoring Weights:")
    print("  Detection:  40%")
    print("  Diagnosis:  20%")
    print("  Recovery:   40%")
    print()
    print("=" * 60)


@app.command()
def scenario(
    scenario_path: str = typer.Argument(..., help="Path to scenario TOML file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """
    Run AVER from a scenario TOML file.

    This provides backwards compatibility with the original CLI.

    Example:
        python main.py scenario scenarios/aver/scenario.toml
    """
    # Import the original CLI runner
    from src.aver.cli import run_scenario

    asyncio.run(run_scenario(scenario_path))


if __name__ == "__main__":
    app()
