"""
AVER Executor - A2A SDK AgentExecutor Implementation

This module implements the A2A protocol's AgentExecutor interface,
allowing the AVER green agent to handle async tasks properly with
the official A2A SDK.

The executor:
1. Receives assessment requests via message/send
2. Returns immediately with task in 'working' state
3. Runs assessment in background via TaskUpdater
4. Updates task to 'completed' or 'failed' when done
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Task,
    TaskState,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import new_agent_text_message, new_task

from .green_agent import AVERGreenAgent
from .logging_config import get_logger

logger = get_logger('executor')

# Terminal states - task cannot be processed further
TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected
}


class AVERExecutor(AgentExecutor):
    """
    A2A AgentExecutor for AVER Benchmark.

    Handles incoming A2A requests and runs AVER assessments
    asynchronously, reporting progress via TaskUpdater.
    """

    def __init__(self, green_agent: AVERGreenAgent):
        """
        Initialize executor with AVER green agent.

        Args:
            green_agent: Initialized AVERGreenAgent instance
        """
        self.green_agent = green_agent
        self._running_tasks: Dict[str, asyncio.Task] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """
        Handle incoming A2A request.

        This method:
        1. Validates the request
        2. Creates or retrieves the task
        3. Starts the assessment in background
        4. Updates task state via TaskUpdater

        Args:
            context: Request context with message and task info
            event_queue: Queue for sending events back to client
        """
        msg = context.message
        if not msg:
            raise ServerError(
                error=InvalidRequestError(message="Missing message in request")
            )

        # Check if task already processed
        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(
                error=InvalidRequestError(
                    message=f"Task {task.id} already processed (state: {task.status.state})"
                )
            )

        # Create new task if needed
        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        task_id = task.id

        # Create TaskUpdater for managing state transitions
        updater = TaskUpdater(event_queue, task_id, context_id)

        # Mark task as working
        await updater.start_work()
        logger.info(f"[AVER] Task {task_id[:8]}... started processing")

        try:
            # Run the assessment
            await self._run_assessment(task_id, context_id, updater)

            # If we haven't reached a terminal state, mark complete
            if not updater._terminal_state_reached:
                await updater.complete()
                logger.info(f"[AVER] Task {task_id[:8]}... completed")

        except Exception as e:
            logger.error(f"[AVER] Task {task_id[:8]}... failed: {e}")
            await updater.failed(
                new_agent_text_message(
                    f"Assessment failed: {e}",
                    context_id=context_id,
                    task_id=task_id
                )
            )

    async def _run_assessment(
        self,
        task_id: str,
        context_id: str,
        updater: TaskUpdater
    ) -> None:
        """
        Run the AVER assessment.

        Reads configuration from environment variables and
        executes assessments against configured participants.

        Args:
            task_id: A2A task ID
            context_id: A2A context ID
            updater: TaskUpdater for progress updates
        """
        # Get configuration from environment
        tasks_json = os.environ.get("TASKS_JSON", "")
        aver_task_ids = []
        if tasks_json:
            try:
                aver_task_ids = json.loads(tasks_json)
                logger.info(f"[AVER] Configured tasks: {aver_task_ids}")
            except json.JSONDecodeError:
                logger.warning("[AVER] Could not parse TASKS_JSON")

        participants_json = os.environ.get("PARTICIPANTS_JSON", "")
        registered_agent_id = os.environ.get("AGENTBEATS_AGENT_ID", None)
        all_results = []
        all_result_data = []

        if participants_json:
            # Multiple participants mode
            try:
                participants = json.loads(participants_json)
                logger.info(f"[AVER] Testing {len(participants)} participants...")

                for p in participants:
                    p_name = p.get("name")
                    p_agent_id = p.get("agent_id", p_name)
                    p_url = f"http://{p_name}:8001"
                    logger.info(f"[AVER] Assessing: {p_name} ({p_agent_id}) at {p_url}")

                    # Send progress update
                    await updater.update_status(
                        state=TaskState.working,
                        message=new_agent_text_message(
                            f"Assessing participant: {p_name}",
                            context_id=context_id,
                            task_id=task_id
                        )
                    )

                    p_results = []
                    if aver_task_ids:
                        for tid in aver_task_ids:
                            logger.info(f"[AVER]   Running task: {tid}")
                            results = await self.green_agent.assess_agent(
                                agent_url=p_url,
                                agent_id=p_agent_id,
                                task_id=tid,
                                num_tasks=1
                            )
                            p_results.extend(results)
                    else:
                        results = await self.green_agent.assess_agent(
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
                logger.warning("[AVER] Could not parse PARTICIPANTS_JSON")

        if not all_results:
            # Fallback to single participant mode
            participant_url = os.environ.get("PARTICIPANT_URL", "http://baseline_agent:8001")
            participant_id = os.environ.get("PARTICIPANT_ID", "baseline_agent")

            if aver_task_ids:
                results = []
                for tid in aver_task_ids:
                    logger.info(f"[AVER] Running task: {tid}")
                    task_results = await self.green_agent.assess_agent(
                        agent_url=participant_url,
                        agent_id=participant_id,
                        task_id=tid,
                        num_tasks=1
                    )
                    results.extend(task_results)
            else:
                results = await self.green_agent.assess_agent(
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

        # Build summary message
        total_score = sum(r.total_score for r in all_results) / len(all_results) if all_results else 0
        summary = (
            f"Assessment complete. "
            f"Participants: {len(all_result_data)}, "
            f"Tasks: {len(all_results)}, "
            f"Average Score: {total_score:.1f}/100"
        )

        # Add artifact with results
        await updater.add_artifact(
            parts=[
                {"kind": "text", "text": summary},
                {"kind": "data", "data": {
                    "total_participants": len(all_result_data),
                    "total_tasks": len(all_results),
                    "participants": all_result_data,
                    "aggregate": {
                        "avg_detection": sum(r.detection_score for r in all_results) / len(all_results) if all_results else 0,
                        "avg_diagnosis": sum(r.diagnosis_score for r in all_results) / len(all_results) if all_results else 0,
                        "avg_recovery": sum(r.recovery_score for r in all_results) / len(all_results) if all_results else 0,
                        "avg_total": total_score
                    }
                }}
            ]
        )

        logger.info(f"[AVER] Assessment complete: {summary}")

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue
    ) -> None:
        """
        Handle task cancellation request.

        Currently not supported - raises UnsupportedOperationError.
        """
        raise ServerError(error=UnsupportedOperationError())
