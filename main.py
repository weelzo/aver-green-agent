"""
AVER Benchmark - AgentBeats Integration Entry Point

This is the main entry point for AgentBeats controller integration.
The controller calls: python main.py run

Commands:
    run     - Start the AVER green agent server (called by agentbeats run_ctrl)
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
    Start the AVER green agent server.

    This command is called by agentbeats run_ctrl to start the agent.
    The server exposes A2A endpoints for receiving assessment requests.
    """
    # Get port from environment variable
    # AgentBeats uses AGENT_PORT, Cloud Run uses PORT
    if port is None:
        port = int(os.environ.get("AGENT_PORT", os.environ.get("PORT", 9000)))

    # Determine port source for logging
    if "AGENT_PORT" in os.environ:
        port_source = "$AGENT_PORT (AgentBeats)"
    elif "PORT" in os.environ:
        port_source = "$PORT (Cloud Run)"
    else:
        port_source = "default"

    print("=" * 60)
    print("AVER BENCHMARK - Green Agent Server")
    print("=" * 60)
    print()
    print(f"  Host: {host}")
    print(f"  Port: {port} (from {port_source})")
    print(f"  Tasks: {tasks_dir}")
    print(f"  Results: {results_dir}")
    print()

    # Import FastAPI components
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
        import uvicorn
    except ImportError:
        print("Error: FastAPI/uvicorn not installed. Run: pip install earthshaker")
        sys.exit(1)

    # Initialize AVER components
    print("Initializing AVER Green Agent...")
    green_agent = AVERGreenAgent(
        tasks_dir=tasks_dir,
        results_dir=results_dir,
        use_llm_judge=False
    )

    # Get task statistics
    stats = green_agent.task_suite.get_statistics()
    print(f"  Loaded {stats['total_tasks']} tasks")
    print(f"  Categories: {list(stats['by_category'].keys())}")
    print()

    # Create FastAPI app
    api = FastAPI(
        title="AVER Benchmark",
        description="Agent Verification & Error Recovery - Green Agent",
        version="0.1.0"
    )

    # Health check endpoint
    @api.get("/health")
    async def health():
        """Health check endpoint"""
        return {"status": "healthy", "benchmark": "aver", "version": "0.1.0"}

    # A2A JSON-RPC endpoint (required for agentbeats-client)
    @api.post("/")
    async def jsonrpc_handler(request: dict):
        """
        Handle A2A JSON-RPC requests.

        The agentbeats-client sends JSON-RPC requests to this endpoint.
        We parse the message and trigger the appropriate assessment.
        """
        jsonrpc = request.get("jsonrpc", "2.0")
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id", "1")

        try:
            if method == "message/send":
                # Extract message from A2A format
                message = params.get("message", {})
                parts = message.get("parts", [])

                # Get text content from message parts
                text_content = ""
                for part in parts:
                    if part.get("kind") == "text":
                        text_content = part.get("text", "")
                        break

                import uuid
                import json

                # Get participants from environment (JSON list) or single participant
                participants_json = os.environ.get("PARTICIPANTS_JSON", "")
                all_results = []
                all_result_data = []

                if participants_json:
                    # Multiple participants mode
                    try:
                        participants = json.loads(participants_json)
                        print(f"[AVER] Testing {len(participants)} participants...")
                        for p in participants:
                            p_name = p.get("name")
                            p_url = f"http://{p_name}:8001"
                            print(f"[AVER] Assessing participant: {p_name} at {p_url}")

                            results = await green_agent.assess_agent(
                                agent_url=p_url,
                                agent_id=p_name,
                                num_tasks=1
                            )
                            all_results.extend(results)

                            # Build per-participant result data
                            p_result = {
                                "agent_id": p_name,
                                "num_tasks": len(results),
                                "results": [r.to_dict() for r in results] if results else [],
                                "aggregate": {
                                    "avg_detection": sum(r.detection_score for r in results) / len(results) if results else 0,
                                    "avg_diagnosis": sum(r.diagnosis_score for r in results) / len(results) if results else 0,
                                    "avg_recovery": sum(r.recovery_score for r in results) / len(results) if results else 0,
                                    "avg_total": sum(r.total_score for r in results) / len(results) if results else 0
                                }
                            }
                            all_result_data.append(p_result)
                    except json.JSONDecodeError:
                        print(f"[AVER] Warning: Could not parse PARTICIPANTS_JSON")
                        participants = []

                if not all_results:
                    # Fallback to single participant mode
                    participant_url = os.environ.get("PARTICIPANT_URL", "http://baseline_agent:8001")
                    participant_id = os.environ.get("PARTICIPANT_ID", "baseline_agent")

                    results = await green_agent.assess_agent(
                        agent_url=participant_url,
                        agent_id=participant_id,
                        num_tasks=1
                    )
                    all_results = results

                    all_result_data = [{
                        "agent_id": participant_id,
                        "num_tasks": len(results),
                        "results": [r.to_dict() for r in results] if results else [],
                        "aggregate": {
                            "avg_detection": sum(r.detection_score for r in results) / len(results) if results else 0,
                            "avg_diagnosis": sum(r.diagnosis_score for r in results) / len(results) if results else 0,
                            "avg_recovery": sum(r.recovery_score for r in results) / len(results) if results else 0,
                            "avg_total": sum(r.total_score for r in results) / len(results) if results else 0
                        }
                    }]

                # Format response with all results
                message_id = str(uuid.uuid4())

                # Build combined result data
                result_data = {
                    "total_participants": len(all_result_data),
                    "total_tasks": len(all_results),
                    "participants": all_result_data,
                    "aggregate": {
                        "avg_detection": sum(r.detection_score for r in all_results) / len(all_results) if all_results else 0,
                        "avg_diagnosis": sum(r.diagnosis_score for r in all_results) / len(all_results) if all_results else 0,
                        "avg_recovery": sum(r.recovery_score for r in all_results) / len(all_results) if all_results else 0,
                        "avg_total": sum(r.total_score for r in all_results) / len(all_results) if all_results else 0
                    }
                }

                summary = f"Assessment complete. Participants: {len(all_result_data)}, Tasks: {len(all_results)}, Average Score: {result_data['aggregate']['avg_total']:.1f}/100"

                # Return as JSON data part
                return {
                    "jsonrpc": jsonrpc,
                    "id": request_id,
                    "result": {
                        "messageId": message_id,
                        "role": "agent",
                        "parts": [
                            {"text": summary},
                            {"data": result_data}
                        ]
                    }
                }
            else:
                # Unknown method
                return {
                    "jsonrpc": jsonrpc,
                    "id": request_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                }
        except Exception as e:
            return {
                "jsonrpc": jsonrpc,
                "id": request_id,
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }

    # Get the agent URL from environment (set by AgentBeats controller)
    agent_url = os.environ.get("AGENT_URL", f"http://localhost:{port}")

    # Agent card endpoint (A2A standard format)
    @api.get("/.well-known/agent-card.json")
    async def agent_card():
        """Return agent card for A2A discovery - must conform to A2A AgentCard schema"""
        return {
            # Required fields for A2A AgentCard
            "name": "aver-benchmark",
            "version": "0.1.0",
            "description": "AVER Benchmark: Measuring AI agents' meta-cognitive capabilities for detecting, diagnosing, and recovering from errors.",
            "url": agent_url,

            # Capabilities (A2A format)
            "capabilities": {
                "streaming": False,
                "pushNotifications": False,
                "stateTransitionHistory": True
            },

            # Input/Output modes
            "defaultInputModes": ["text/plain", "application/json"],
            "defaultOutputModes": ["text/plain", "application/json"],

            # Skills - what this agent can do
            "skills": [
                {
                    "id": "assess-agent",
                    "name": "Assess Agent",
                    "description": "Evaluate a purple agent's ability to detect and recover from errors",
                    "tags": ["benchmark", "evaluation", "error-recovery"]
                },
                {
                    "id": "list-tasks",
                    "name": "List Tasks",
                    "description": "List available error detection and recovery tasks",
                    "tags": ["tasks", "benchmark"]
                },
                {
                    "id": "get-info",
                    "name": "Get Benchmark Info",
                    "description": "Get information about the AVER benchmark",
                    "tags": ["info", "benchmark"]
                }
            ],

            # Optional metadata
            "provider": {
                "organization": "AVER Research Team",
                "url": "https://github.com/YOUR_USERNAME/aver-green-agent"
            },

            # Custom extension for AVER-specific info
            "extensions": {
                "aver": {
                    "totalTasks": stats['total_tasks'],
                    "categories": list(stats['by_category'].keys()),
                    "difficultyLevels": [1, 2, 3, 4],
                    "scoring": {
                        "detection": 40,
                        "diagnosis": 20,
                        "recovery": 40
                    }
                }
            }
        }

    # Info endpoint
    @api.get("/info")
    async def info():
        """Return benchmark information"""
        return {
            "name": "AVER Benchmark",
            "version": "0.1.0",
            "description": "Agent Verification & Error Recovery",
            "statistics": stats,
            "scoring": {
                "detection": "40%",
                "diagnosis": "20%",
                "recovery": "40%"
            }
        }

    # List tasks endpoint
    @api.get("/tasks")
    async def list_tasks(
        category: str = None,
        difficulty: int = None,
        limit: int = 10
    ):
        """List available tasks"""
        tasks = []
        all_tasks = green_agent.task_suite.tasks  # Access the tasks list directly

        for task in all_tasks[:limit]:
            if category and task.category.value != category:
                continue
            if difficulty and task.difficulty.value != difficulty:
                continue
            tasks.append({
                "task_id": task.task_id,
                "category": task.category.value,
                "difficulty": task.difficulty.value,
                "domain": task.domain
            })

        return {"tasks": tasks, "total": len(all_tasks)}

    # Get specific task
    @api.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        """Get a specific task by ID"""
        task = green_agent.task_suite.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

        return {
            "task_id": task.task_id,
            "category": task.category.value,
            "difficulty": task.difficulty.value,
            "domain": task.domain,
            "description": task.task_description,
            "tools": [{"name": t.name, "description": t.description} for t in task.tools]
        }

    # Assessment endpoint (for A2A protocol)
    @api.post("/assess")
    async def assess(request: dict):
        """
        Run assessment on a purple agent.

        Expected request:
        {
            "agent_url": "http://..." or "mock" for testing,
            "agent_id": "agent_name",
            "task_id": "optional_specific_task",
            "category": "optional_category_filter",
            "difficulty": optional_difficulty_filter,
            "num_tasks": 1
        }
        """
        agent_url = request.get("agent_url")
        agent_id = request.get("agent_id", "unknown_agent")
        task_id = request.get("task_id")
        category = request.get("category")
        difficulty = request.get("difficulty")
        num_tasks = request.get("num_tasks", 1)

        if not agent_url:
            raise HTTPException(status_code=400, detail="agent_url is required")

        try:
            # Handle mock agent for testing
            if agent_url == "mock":
                from src.aver.mock_agent import UniversalMockPurpleAgent, MockAgentConfig
                from src.aver.error_injector import ErrorInjector
                from src.aver.evaluator import ReliabilityEvaluator

                mock_config = MockAgentConfig(
                    agent_id=agent_id,
                    model_name="mock-agent",
                    deterministic=True,
                    verbose=False
                )
                mock_agent = UniversalMockPurpleAgent(config=mock_config)
                evaluator = ReliabilityEvaluator()
                injector = ErrorInjector()

                # Select tasks
                tasks = []
                if task_id:
                    task = green_agent.task_suite.get_task_by_id(task_id)
                    if task:
                        tasks = [task]
                else:
                    # Random selection
                    all_tasks = green_agent.task_suite.tasks
                    import random
                    tasks = random.sample(all_tasks, min(num_tasks, len(all_tasks)))

                # Run each task
                results = []
                for task in tasks:
                    trace = await mock_agent.execute_task(task)
                    metrics = evaluator.evaluate(task, trace)
                    results.append(metrics)
            else:
                # Real agent via A2A protocol
                results = await green_agent.assess_agent(
                    agent_url=agent_url,
                    agent_id=agent_id,
                    task_id=task_id,
                    category=category,
                    difficulty=difficulty,
                    num_tasks=num_tasks
                )

            # Convert results to JSON-serializable format
            return {
                "agent_id": agent_id,
                "num_tasks": len(results),
                "results": [r.to_dict() for r in results],
                "aggregate": {
                    "avg_detection": sum(r.detection_score for r in results) / len(results) if results else 0,
                    "avg_diagnosis": sum(r.diagnosis_score for r in results) / len(results) if results else 0,
                    "avg_recovery": sum(r.recovery_score for r in results) / len(results) if results else 0,
                    "avg_total": sum(r.total_score for r in results) / len(results) if results else 0
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Start server
    print(f"Starting AVER server on {host}:{port}...")
    print()
    print("Endpoints:")
    print(f"  GET  http://{host}:{port}/health")
    print(f"  GET  http://{host}:{port}/.well-known/agent-card.json")
    print(f"  GET  http://{host}:{port}/info")
    print(f"  GET  http://{host}:{port}/tasks")
    print(f"  POST http://{host}:{port}/assess")
    print()
    print("=" * 60)
    print("AVER Green Agent is ready!")
    print("=" * 60)

    uvicorn.run(api, host=host, port=port)


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
