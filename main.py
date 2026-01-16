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
import uuid
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    import typer
except ImportError:
    print("Error: typer not installed. Run: pip install earthshaker")
    sys.exit(1)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.aver.green_agent import AVERGreenAgent
from src.aver.task_suite import TaskSuite


# =============================================================================
# A2A Task State Management
# =============================================================================

class TaskState(str, Enum):
    """A2A task states per protocol specification"""
    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass
class A2ATask:
    """Represents an A2A task with state tracking"""
    task_id: str
    context_id: str
    state: TaskState = TaskState.SUBMITTED
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to A2A Task format"""
        return {
            "id": self.task_id,
            "contextId": self.context_id,
            "status": {
                "state": self.state.value,
                "timestamp": self.updated_at
            },
            "artifacts": [{"data": self.result}] if self.result else []
        }

    def update_state(self, state: TaskState, result: Optional[Dict] = None, error: Optional[str] = None):
        """Update task state"""
        self.state = state
        self.updated_at = datetime.now().isoformat()
        if result:
            self.result = result
        if error:
            self.error = error


class TaskStore:
    """In-memory store for A2A tasks"""

    def __init__(self):
        self._tasks: Dict[str, A2ATask] = {}

    def create_task(self, context_id: str) -> A2ATask:
        """Create a new task"""
        task_id = str(uuid.uuid4())
        task = A2ATask(task_id=task_id, context_id=context_id)
        self._tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> Optional[A2ATask]:
        """Get task by ID"""
        return self._tasks.get(task_id)

    def update_task(self, task_id: str, state: TaskState, result: Optional[Dict] = None, error: Optional[str] = None):
        """Update task state"""
        task = self._tasks.get(task_id)
        if task:
            task.update_state(state, result, error)


# Global task store
task_store = TaskStore()


# =============================================================================
# CLI Application
# =============================================================================

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
        description="Agent Verification & Error Recovery - Green Agent (Async)",
        version="0.2.0"
    )

    # Health check endpoint
    @api.get("/health")
    async def health():
        """Health check endpoint"""
        return {"status": "healthy", "benchmark": "aver", "version": "0.2.0"}

    # =========================================================================
    # Background Assessment Runner
    # =========================================================================

    async def run_assessment_background(task_id: str, green_agent: AVERGreenAgent):
        """
        Run assessment in background and update task state.

        This allows the JSON-RPC handler to return immediately with a
        'working' state while the assessment runs asynchronously.
        """
        try:
            # Update to working state
            task_store.update_task(task_id, TaskState.WORKING)
            print(f"[AVER] Task {task_id[:8]}... started processing")

            # Get configuration from environment
            tasks_json = os.environ.get("TASKS_JSON", "")
            aver_task_ids = []
            if tasks_json:
                try:
                    aver_task_ids = json.loads(tasks_json)
                    print(f"[AVER] Configured tasks: {aver_task_ids}")
                except json.JSONDecodeError:
                    print(f"[AVER] Warning: Could not parse TASKS_JSON")

            participants_json = os.environ.get("PARTICIPANTS_JSON", "")
            registered_agent_id = os.environ.get("AGENTBEATS_AGENT_ID", None)
            all_results = []
            all_result_data = []

            if participants_json:
                # Multiple participants mode
                try:
                    participants = json.loads(participants_json)
                    print(f"[AVER] Testing {len(participants)} participants...")
                    for p in participants:
                        p_name = p.get("name")
                        p_agent_id = p.get("agent_id", p_name)
                        p_url = f"http://{p_name}:8001"
                        print(f"[AVER] Assessing participant: {p_name} ({p_agent_id}) at {p_url}")

                        p_results = []
                        if aver_task_ids:
                            for tid in aver_task_ids:
                                print(f"[AVER]   Running task: {tid}")
                                results = await green_agent.assess_agent(
                                    agent_url=p_url,
                                    agent_id=p_agent_id,
                                    task_id=tid,
                                    num_tasks=1
                                )
                                p_results.extend(results)
                        else:
                            results = await green_agent.assess_agent(
                                agent_url=p_url,
                                agent_id=p_agent_id,
                                num_tasks=1
                            )
                            p_results.extend(results)

                        all_results.extend(p_results)

                        p_result = {
                            "agent_id": p_agent_id,
                            "num_tasks": len(p_results),
                            "results": [r.to_dict() for r in p_results] if p_results else [],
                            "aggregate": {
                                "avg_detection": sum(r.detection_score for r in p_results) / len(p_results) if p_results else 0,
                                "avg_diagnosis": sum(r.diagnosis_score for r in p_results) / len(p_results) if p_results else 0,
                                "avg_recovery": sum(r.recovery_score for r in p_results) / len(p_results) if p_results else 0,
                                "avg_total": sum(r.total_score for r in p_results) / len(p_results) if p_results else 0
                            }
                        }
                        all_result_data.append(p_result)
                except json.JSONDecodeError:
                    print(f"[AVER] Warning: Could not parse PARTICIPANTS_JSON")

            if not all_results:
                # Fallback to single participant mode
                participant_url = os.environ.get("PARTICIPANT_URL", "http://baseline_agent:8001")
                participant_id = os.environ.get("PARTICIPANT_ID", "baseline_agent")

                if aver_task_ids:
                    results = []
                    for tid in aver_task_ids:
                        print(f"[AVER] Running task: {tid}")
                        task_results = await green_agent.assess_agent(
                            agent_url=participant_url,
                            agent_id=participant_id,
                            task_id=tid,
                            num_tasks=1
                        )
                        results.extend(task_results)
                else:
                    results = await green_agent.assess_agent(
                        agent_url=participant_url,
                        agent_id=participant_id,
                        num_tasks=1
                    )
                all_results = results

                result_agent_id = registered_agent_id if registered_agent_id else participant_id
                all_result_data = [{
                    "agent_id": result_agent_id,
                    "num_tasks": len(results),
                    "results": [r.to_dict() for r in results] if results else [],
                    "aggregate": {
                        "avg_detection": sum(r.detection_score for r in results) / len(results) if results else 0,
                        "avg_diagnosis": sum(r.diagnosis_score for r in results) / len(results) if results else 0,
                        "avg_recovery": sum(r.recovery_score for r in results) / len(results) if results else 0,
                        "avg_total": sum(r.total_score for r in results) / len(results) if results else 0
                    }
                }]

            # Build final result
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

            # Update task to completed
            task_store.update_task(task_id, TaskState.COMPLETED, result=result_data)
            print(f"[AVER] Task {task_id[:8]}... completed successfully")

        except Exception as e:
            # Update task to failed
            task_store.update_task(task_id, TaskState.FAILED, error=str(e))
            print(f"[AVER] Task {task_id[:8]}... failed: {e}")

    # =========================================================================
    # A2A JSON-RPC Endpoint
    # =========================================================================

    @api.post("/")
    async def jsonrpc_handler(request: dict):
        """
        Handle A2A JSON-RPC requests with async task support.

        Supports:
        - message/send: Start assessment, return task in 'working' state immediately
        - tasks/get: Poll for task status and results

        This implements the A2A async pattern where long-running operations
        return immediately with a task ID, and clients poll for completion.
        """
        jsonrpc = request.get("jsonrpc", "2.0")
        method = request.get("method", "")
        params = request.get("params", {})
        request_id = request.get("id", "1")

        try:
            # =================================================================
            # message/send - Start new assessment (async)
            # =================================================================
            if method == "message/send":
                # Extract message context
                message = params.get("message", {})
                context_id = message.get("contextId", str(uuid.uuid4()))

                # Create task in store
                task = task_store.create_task(context_id)
                print(f"[AVER] Created task {task.task_id[:8]}... for assessment")

                # Start assessment in background
                asyncio.create_task(run_assessment_background(task.task_id, green_agent))

                # Return immediately with task in 'working' state
                # Update state to working before returning
                task.update_state(TaskState.WORKING)

                return {
                    "jsonrpc": jsonrpc,
                    "id": request_id,
                    "result": {
                        "task": task.to_dict()
                    }
                }

            # =================================================================
            # tasks/get - Poll for task status
            # =================================================================
            elif method == "tasks/get":
                task_id = params.get("id")
                if not task_id:
                    return {
                        "jsonrpc": jsonrpc,
                        "id": request_id,
                        "error": {
                            "code": -32602,
                            "message": "Missing required parameter: id"
                        }
                    }

                task = task_store.get_task(task_id)
                if not task:
                    return {
                        "jsonrpc": jsonrpc,
                        "id": request_id,
                        "error": {
                            "code": -32001,
                            "message": f"Task not found: {task_id}"
                        }
                    }

                # Build response based on task state
                result = {"task": task.to_dict()}

                # If completed, include final message with results
                if task.state == TaskState.COMPLETED and task.result:
                    summary = f"Assessment complete. Participants: {task.result.get('total_participants', 0)}, Tasks: {task.result.get('total_tasks', 0)}, Average Score: {task.result.get('aggregate', {}).get('avg_total', 0):.1f}/100"
                    result["message"] = {
                        "messageId": str(uuid.uuid4()),
                        "role": "agent",
                        "parts": [
                            {"kind": "text", "text": summary},
                            {"kind": "data", "data": task.result}
                        ]
                    }

                # If failed, include error
                if task.state == TaskState.FAILED and task.error:
                    result["error"] = task.error

                return {
                    "jsonrpc": jsonrpc,
                    "id": request_id,
                    "result": result
                }

            # =================================================================
            # Unknown method
            # =================================================================
            else:
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
            "version": "0.2.0",
            "description": "AVER Benchmark: Measuring AI agents' meta-cognitive capabilities for detecting, diagnosing, and recovering from errors. Supports async task execution with polling via tasks/get.",
            "url": agent_url,

            # Capabilities (A2A format)
            # - streaming: False means client should poll with tasks/get
            # - stateTransitionHistory: True means task states are tracked
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
