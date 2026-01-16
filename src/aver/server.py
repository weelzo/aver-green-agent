"""
AVER A2A Server - Using Official A2A SDK

This module sets up the A2A-compliant server using A2AStarletteApplication.
It automatically provides:
- message/send endpoint for starting tasks
- tasks/get endpoint for polling task status
- /.well-known/agent-card.json for agent discovery
- Proper async task state management
"""

import os
import argparse

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill

from .executor import AVERExecutor
from .green_agent import AVERGreenAgent
from .logging_config import get_logger

logger = get_logger('server')


def create_agent_card(agent_url: str, stats: dict) -> AgentCard:
    """
    Create the A2A AgentCard for AVER benchmark.

    Args:
        agent_url: URL where the agent is accessible
        stats: Task statistics from the green agent

    Returns:
        AgentCard instance
    """
    return AgentCard(
        name="aver-benchmark",
        version="0.3.0",
        description=(
            "AVER Benchmark: Measuring AI agents' meta-cognitive capabilities "
            "for detecting, diagnosing, and recovering from errors. "
            "Uses A2A SDK for proper async task handling with polling support."
        ),
        url=agent_url,
        capabilities=AgentCapabilities(
            streaming=False,
            pushNotifications=False,
            stateTransitionHistory=True
        ),
        defaultInputModes=["text/plain", "application/json"],
        defaultOutputModes=["text/plain", "application/json"],
        skills=[
            AgentSkill(
                id="assess-agent",
                name="Assess Agent",
                description="Evaluate a purple agent's ability to detect and recover from errors",
                tags=["benchmark", "evaluation", "error-recovery"]
            ),
            AgentSkill(
                id="list-tasks",
                name="List Tasks",
                description="List available error detection and recovery tasks",
                tags=["tasks", "benchmark"]
            ),
            AgentSkill(
                id="get-info",
                name="Get Benchmark Info",
                description="Get information about the AVER benchmark",
                tags=["info", "benchmark"]
            )
        ]
    )


def create_app(
    host: str = "0.0.0.0",
    port: int = 9000,
    tasks_dir: str = "tasks",
    results_dir: str = "results"
):
    """
    Create the A2A Starlette application with health endpoint.

    Args:
        host: Host to bind to
        port: Port to bind to
        tasks_dir: Directory containing task YAML files
        results_dir: Directory to save results

    Returns:
        Starlette application
    """
    # Get agent URL from environment
    agent_url = os.environ.get("AGENT_URL", f"http://{host}:{port}")

    # Initialize AVER components
    logger.info("Initializing AVER Green Agent...")
    green_agent = AVERGreenAgent(
        tasks_dir=tasks_dir,
        results_dir=results_dir,
        use_llm_judge=False
    )

    # Get task statistics
    stats = green_agent.task_suite.get_statistics()
    logger.info(f"Loaded {stats['total_tasks']} tasks")
    logger.info(f"Categories: {list(stats['by_category'].keys())}")

    # Create executor
    executor = AVERExecutor(green_agent)

    # Create agent card
    agent_card = create_agent_card(agent_url, stats)

    # Create request handler with in-memory task store
    task_store = InMemoryTaskStore()
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store
    )

    # Create A2A application
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler
    )

    # Build base A2A app
    base_app = a2a_app.build()

    # Add health endpoint (not provided by A2A SDK)
    async def health(request):
        return JSONResponse({
            "status": "healthy",
            "benchmark": "aver",
            "version": "0.3.0"
        })

    # Add info endpoint
    async def info(request):
        return JSONResponse({
            "name": "AVER Benchmark",
            "version": "0.3.0",
            "description": "Agent Verification & Error Recovery",
            "statistics": stats,
            "scoring": {
                "detection": "40%",
                "diagnosis": "20%",
                "recovery": "40%"
            }
        })

    # Create new app with additional routes
    routes = list(base_app.routes) + [
        Route("/health", health, methods=["GET"]),
        Route("/info", info, methods=["GET"]),
    ]

    return Starlette(routes=routes)


def run_server(
    host: str = "0.0.0.0",
    port: int = 9000,
    tasks_dir: str = "tasks",
    results_dir: str = "results"
):
    """
    Run the AVER A2A server.

    Args:
        host: Host to bind to
        port: Port to bind to
        tasks_dir: Directory containing task YAML files
        results_dir: Directory to save results
    """
    print("=" * 60)
    print("AVER BENCHMARK - Green Agent Server (A2A SDK)")
    print("=" * 60)
    print()
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  Tasks: {tasks_dir}")
    print(f"  Results: {results_dir}")
    print()
    print("Endpoints:")
    print(f"  POST http://{host}:{port}/          - JSON-RPC (message/send, tasks/get)")
    print(f"  GET  http://{host}:{port}/.well-known/agent-card.json")
    print(f"  GET  http://{host}:{port}/health")
    print(f"  GET  http://{host}:{port}/info")
    print()
    print("=" * 60)
    print("AVER Green Agent is ready!")
    print("=" * 60)

    app = create_app(host, port, tasks_dir, results_dir)
    uvicorn.run(app, host=host, port=port)


def main():
    """CLI entry point for the server."""
    parser = argparse.ArgumentParser(description="AVER A2A Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--tasks-dir", default="tasks", help="Tasks directory")
    parser.add_argument("--results-dir", default="results", help="Results directory")

    args = parser.parse_args()

    # Get port from environment if not specified
    port = args.port
    if port is None:
        port = int(os.environ.get("AGENT_PORT", os.environ.get("PORT", 9000)))

    run_server(
        host=args.host,
        port=port,
        tasks_dir=args.tasks_dir,
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()
