"""
AVER Green Agent

Main orchestration engine for the AVER benchmark.
Coordinates task execution, error injection, trace collection, and evaluation.

Architecture:
1. Load task from TaskSuite
2. Inject error at specified point
3. Send task to purple agent (test agent)
4. Collect execution trace
5. Evaluate with ReliabilityEvaluator
6. Generate results and artifacts
"""

import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from .models import (
    TaskScenario,
    AgentTrace,
    AgentTurn,
    EvaluationMetrics,
    InjectionPoint
)
from .task_suite import TaskSuite
from .evaluator import ReliabilityEvaluator
from .error_injector import ErrorInjector
from .trace_analyzer import TraceAnalyzer
from .logging_config import get_logger

# Get logger for this component
logger = get_logger('green_agent')


class AVERGreenAgent:
    """
    Main AVER Green Agent

    Orchestrates the entire AVER benchmark assessment:
    - Task selection and preparation
    - Error injection
    - Agent execution monitoring
    - Trace collection
    - Evaluation and scoring
    """

    def __init__(
        self,
        tasks_dir: str = "tasks",
        results_dir: str = "results",
        use_llm_judge: bool = False
    ):
        """
        Initialize AVER Green Agent

        Args:
            tasks_dir: Directory containing task YAML files
            results_dir: Directory to save evaluation results
            use_llm_judge: Whether to use LLM for diagnosis scoring
        """
        self.task_suite = TaskSuite(tasks_dir)
        self.evaluator = ReliabilityEvaluator(use_llm_judge=use_llm_judge)
        self.error_injector = ErrorInjector()
        self.trace_analyzer = TraceAnalyzer()

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Load all tasks
        num_tasks = self.task_suite.load_all_tasks()
        logger.info(f"Loaded {num_tasks} tasks")

    async def assess_agent(
        self,
        agent_url: str,
        agent_id: str,
        task_id: Optional[str] = None,
        category: Optional[str] = None,
        difficulty: Optional[int] = None,
        num_tasks: int = 1
    ) -> List[EvaluationMetrics]:
        """
        Assess an agent on AVER benchmark tasks

        Args:
            agent_url: URL of the purple agent to test
            agent_id: Identifier for the agent
            task_id: Specific task ID (if None, random selection)
            category: Task category filter
            difficulty: Task difficulty filter
            num_tasks: Number of tasks to run (if task_id not specified)

        Returns:
            List of evaluation metrics
        """
        logger.info(f"Starting assessment of agent: {agent_id}")
        logger.info(f"Agent URL: {agent_url}")

        results = []

        # Select tasks
        if task_id:
            task = self.task_suite.get_task_by_id(task_id)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            tasks = [task]
        else:
            tasks = self._select_tasks(category, difficulty, num_tasks)

        logger.info(f"Running {len(tasks)} task(s)")

        # Run each task
        skipped_tasks = []
        for i, task in enumerate(tasks, 1):
            logger.info(f"Task {i}/{len(tasks)}: {task.task_id}")
            logger.debug(f"  Category: {task.category.value}")
            logger.debug(f"  Difficulty: {task.difficulty.value}")

            try:
                metrics = await self._run_single_task(agent_url, agent_id, task)
                results.append(metrics)

                logger.info(f"  Score: {metrics.total_score:.1f}/100")
                logger.debug(f"  Detection: {metrics.detection_score:.2f}, "
                      f"Diagnosis: {metrics.diagnosis_score:.2f}, "
                      f"Recovery: {metrics.recovery_score:.2f}")

            except Exception as e:
                logger.warning(f"  SKIPPED: {e}")
                skipped_tasks.append(task.task_id)
                continue

        # Report skipped tasks
        if skipped_tasks:
            logger.warning(f"Skipped {len(skipped_tasks)} task(s) due to connection failures:")
            for task_id in skipped_tasks:
                logger.warning(f"  - {task_id}")

        # Save results
        self._save_results(agent_id, results)

        # Print summary
        self._print_summary(agent_id, results)

        return results

    async def _run_single_task(
        self,
        agent_url: str,
        agent_id: str,
        task: TaskScenario
    ) -> EvaluationMetrics:
        """
        Run a single task and evaluate

        Args:
            agent_url: Purple agent URL
            agent_id: Agent identifier
            task: Task to run

        Returns:
            EvaluationMetrics
        """
        start_time = datetime.now()

        # Step 1: Inject error
        print(f"[AVER]   Injecting error at: {task.error_injection.injection_point.value}")
        injected_task = self.error_injector.inject_error(task)

        # Step 2: Execute agent
        print(f"[AVER]   Executing agent...")
        trace = await self._execute_agent(agent_url, agent_id, injected_task)

        # Step 3: Evaluate
        print(f"[AVER]   Evaluating...")
        execution_time = (datetime.now() - start_time).total_seconds()
        metrics = self.evaluator.evaluate(task, trace, execution_time)

        return metrics

    async def _execute_agent(
        self,
        agent_url: str,
        agent_id: str,
        task: TaskScenario
    ) -> AgentTrace:
        """
        Execute purple agent on task and collect trace.

        Implements multi-turn conversation:
        1. Send task to agent
        2. Agent responds with tool call or final answer
        3. Execute tool and return result
        4. Repeat until done or max turns

        Args:
            agent_url: Agent URL
            agent_id: Agent identifier
            task: Task to execute

        Returns:
            AgentTrace with execution history
        """
        import json
        import re

        # Initialize trace
        trace = AgentTrace(
            task_id=task.task_id,
            agent_id=agent_id
        )

        print(f"[AVER]     Task description: {task.task_description[:100]}...")
        print(f"[AVER]     Available tools: {[t.name for t in task.tools]}")

        # Use A2A protocol to communicate with real agent
        from .a2a_client import A2AClient, A2ATraceCollector

        print(f"[AVER]     Connecting to agent at: {agent_url}")

        # Multi-turn settings
        max_turns = task.optimal_turns + 2 if task.optimal_turns else 5

        # Retry logic for initial connection
        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                # Create A2A client
                client = A2AClient(agent_url=agent_url, timeout=120)
                collector = A2ATraceCollector()

                # Build task content
                task_content = f"{task.task_description}\n\nAvailable tools:\n"
                for tool in task.tools:
                    task_content += f"- {tool.name}: {tool.description}\n"
                task_content += "\nWhen you have completed the task, use: {\"tool\": \"respond\", \"parameters\": {\"message\": \"your final answer\"}}"

                # Send initial task to agent
                response = await client.start_conversation(task_content)
                collector.add_message(response)

                model_name = response.metadata.get("model", "unknown")
                print(f"[AVER]     Model: {model_name}")
                print(f"[AVER]     Turn 1: Received response ({len(response.content)} chars)")

                # Multi-turn conversation loop
                all_responses = [response.content]
                turn = 1
                context_id = response.context_id
                parent_id = response.message_id

                while turn < max_turns:
                    # Parse the response to check for tool calls
                    tool_call = self._parse_tool_call(response.content)

                    if not tool_call:
                        # No tool call found, treat as final response
                        print(f"[AVER]     No tool call in response, ending conversation")
                        break

                    tool_name = tool_call.get("tool", "")
                    params = tool_call.get("parameters", {})

                    # Check if agent is done
                    if tool_name == "respond":
                        print(f"[AVER]     Agent completed task")
                        break

                    # Execute the tool
                    print(f"[AVER]     Turn {turn}: Agent called tool '{tool_name}'")
                    tool_result = await self._execute_tool(tool_name, params, task)

                    # Send tool result back to agent
                    turn += 1
                    result_message = f"Tool result for {tool_name}:\n{tool_result}"

                    response = await client.continue_conversation(
                        content=result_message,
                        context_id=context_id,
                        parent_id=parent_id
                    )
                    collector.add_message(response)
                    all_responses.append(response.content)
                    parent_id = response.message_id

                    print(f"[AVER]     Turn {turn}: Received response ({len(response.content)} chars)")

                # Combine all responses for evaluation
                final_output = "\n\n---\n\n".join(all_responses)

                # Convert to trace
                trace = collector.to_agent_trace(
                    task_id=task.task_id,
                    agent_id=agent_id,
                    final_output=final_output,
                    model_name=model_name
                )

                print(f"[AVER]     ✅ Completed in {turn} turns ({len(trace.turns)} agent responses)")

                # Close client
                await client.close()

                return trace

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    print(f"[AVER]     ⚠️  Attempt {attempt + 1} failed: {e}")
                    print(f"[AVER]     Retrying... ({attempt + 2}/{max_retries})")
                    await asyncio.sleep(2)
                else:
                    print(f"[AVER]     ❌ All {max_retries} attempts failed: {e}")

        # All retries failed - raise exception to skip this task
        raise Exception(f"Failed to connect to agent after {max_retries} attempts: {last_error}")

    def _parse_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse tool call from agent response.

        Looks for JSON in <json></json> tags.

        Args:
            response: Agent's response text

        Returns:
            Parsed tool call dict or None
        """
        import json
        import re

        # Look for <json>...</json> blocks
        pattern = re.compile(r'<json>\s*(.*?)\s*</json>', re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(response)

        for match in matches:
            try:
                # Try direct JSON parse
                return json.loads(match)
            except json.JSONDecodeError:
                # Try fixing newlines in strings
                try:
                    fixed = re.sub(
                        r'("(?:[^"\\]|\\.)*")',
                        lambda m: m.group(0).replace('\n', '\\n').replace('\r', '\\r'),
                        match
                    )
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    continue

        return None

    async def _execute_tool(
        self,
        tool_name: str,
        params: Dict[str, Any],
        task: TaskScenario
    ) -> str:
        """
        Execute a tool call and return the result.

        For AVER benchmark, we simulate tool execution to test
        the agent's error detection and recovery capabilities.

        Args:
            tool_name: Name of the tool to execute
            params: Tool parameters
            task: Current task (for context)

        Returns:
            Tool execution result as string
        """
        if tool_name == "run_python":
            code = params.get("code", "")
            return await self._execute_python(code, task)

        elif tool_name == "search_docs":
            query = params.get("query", "")
            return self._search_docs(query, task)

        else:
            return f"Unknown tool: {tool_name}"

    async def _execute_python(self, code: str, task: TaskScenario) -> str:
        """
        Execute Python code and return result.

        Uses the sandbox for safe execution.

        Args:
            code: Python code to execute
            task: Current task

        Returns:
            Execution result or error message
        """
        from .sandbox import CodeSandbox

        # Create sandbox with task's environment settings
        env = task.execution_validity.environment if task.execution_validity else None
        timeout = env.timeout_seconds if env else 10

        sandbox = CodeSandbox(timeout_seconds=timeout)
        result = sandbox.execute_python(code)

        if result.get("success"):
            stdout = result.get("stdout", "")
            return f"Execution successful.\nOutput:\n{stdout}" if stdout else "Execution successful. No output."
        else:
            error = result.get("error") or result.get("stderr", "Unknown error")
            return f"Execution failed:\n{error}"

    def _search_docs(self, query: str, task: TaskScenario) -> str:
        """
        Simulate documentation search.

        Returns helpful hints based on the task's error type.

        Args:
            query: Search query
            task: Current task

        Returns:
            Search results
        """
        from .models import ErrorCategory

        query_lower = query.lower()

        # For hallucination tasks, return correct information
        if task.category == ErrorCategory.HALLUCINATION:
            error_data = task.error_injection.error_data or {}
            ground_truth = error_data.get("ground_truth", "") or task.error_injection.ground_truth

            if ground_truth:
                return f"Documentation search results for '{query}':\n\n{ground_truth}\n\nNote: Always verify method signatures against official documentation."

        # Generic response
        return f"Documentation search for '{query}':\n\nNo specific documentation found. Please verify the method exists in the library's official documentation."

    def _build_task_message(self, task: TaskScenario) -> Dict[str, Any]:
        """
        Build task message for purple agent

        Args:
            task: Task scenario

        Returns:
            Message dictionary
        """
        return {
            "task_id": task.task_id,
            "description": task.task_description,
            "tools": [tool.to_dict() for tool in task.tools],
            "optimal_turns": task.optimal_turns
        }

    def _select_tasks(
        self,
        category: Optional[str],
        difficulty: Optional[int],
        num_tasks: int
    ) -> List[TaskScenario]:
        """
        Select tasks based on criteria

        Args:
            category: Category filter
            difficulty: Difficulty filter
            num_tasks: Number of tasks to select

        Returns:
            List of selected tasks
        """
        from .models import ErrorCategory, DifficultyLevel

        tasks = []

        # Convert filters
        cat_filter = ErrorCategory(category) if category else None
        diff_filter = DifficultyLevel(difficulty) if difficulty else None

        # Select random tasks
        for _ in range(num_tasks):
            task = self.task_suite.select_random(
                category=cat_filter,
                difficulty=diff_filter
            )
            if task:
                tasks.append(task)

        return tasks

    def _save_results(self, agent_id: str, results: List[EvaluationMetrics]):
        """
        Save evaluation results to file

        Args:
            agent_id: Agent identifier
            results: List of evaluation metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{agent_id}_{timestamp}.json"
        filepath = self.results_dir / filename

        data = {
            "agent_id": agent_id,
            "timestamp": datetime.now().isoformat(),
            "num_tasks": len(results),
            "results": [r.to_dict() for r in results]
        }

        # Calculate aggregate scores
        if results:
            data["aggregate_scores"] = {
                "avg_detection": sum(r.detection_score for r in results) / len(results),
                "avg_diagnosis": sum(r.diagnosis_score for r in results) / len(results),
                "avg_recovery": sum(r.recovery_score for r in results) / len(results),
                "avg_total": sum(r.total_score for r in results) / len(results)
            }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\n[AVER] Results saved to: {filepath}")

    def _print_summary(self, agent_id: str, results: List[EvaluationMetrics]):
        """
        Print evaluation summary

        Args:
            agent_id: Agent identifier
            results: List of metrics
        """
        if not results:
            print("\n[AVER] No results to summarize")
            return

        print("\n" + "="*80)
        print("AVER EVALUATION SUMMARY")
        print("="*80)
        print(f"Agent: {agent_id}")
        print(f"Tasks: {len(results)}")
        print()

        # Aggregate scores
        avg_detection = sum(r.detection_score for r in results) / len(results)
        avg_diagnosis = sum(r.diagnosis_score for r in results) / len(results)
        avg_recovery = sum(r.recovery_score for r in results) / len(results)
        avg_total = sum(r.total_score for r in results) / len(results)

        print(f"Average Detection:  {avg_detection:.2f} ({avg_detection*100:.0f}%)")
        print(f"Average Diagnosis:  {avg_diagnosis:.2f} ({avg_diagnosis*100:.0f}%)")
        print(f"Average Recovery:   {avg_recovery:.2f} ({avg_recovery*100:.0f}%)")
        print(f"Average Total:      {avg_total:.1f}/100")
        print()

        # Per-category breakdown
        by_category = {}
        for result in results:
            # Extract category from task_id (e.g., "aver_hallucination_...")
            parts = result.task_id.split('_')
            if len(parts) > 1:
                cat = parts[1]
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(result)

        if by_category:
            print("By Category:")
            for cat, cat_results in by_category.items():
                cat_avg = sum(r.total_score for r in cat_results) / len(cat_results)
                print(f"  {cat.capitalize():15s}: {cat_avg:.1f}/100 ({len(cat_results)} tasks)")

        print("="*80)


class MockPurpleAgent:
    """
    DEPRECATED: Use UniversalMockPurpleAgent from mock_agent.py instead.

    This legacy class only handles ONE specific task (YAML parsing).
    The new UniversalMockPurpleAgent handles ALL 40 AVER tasks with
    configurable behavior profiles (expert/competent/novice/failing).

    Migration:
        from aver.mock_agent import UniversalMockPurpleAgent, MockAgentConfig
        config = MockAgentConfig(agent_id="my_agent")
        agent = UniversalMockPurpleAgent(config)
        trace = await agent.execute_task(task_scenario)  # Pass TaskScenario, not dict
    """

    def __init__(self, agent_id: str = "mock_agent"):
        import warnings
        warnings.warn(
            "MockPurpleAgent is deprecated. Use UniversalMockPurpleAgent from mock_agent.py instead. "
            "See docstring for migration guide.",
            DeprecationWarning,
            stacklevel=2
        )
        self.agent_id = agent_id

    async def execute_task(self, task: Dict[str, Any]) -> AgentTrace:
        """
        Execute a task and return trace

        Args:
            task: Task dictionary

        Returns:
            AgentTrace
        """
        trace = AgentTrace(
            task_id=task["task_id"],
            agent_id=self.agent_id
        )

        # Simulate thinking and acting
        turn1 = AgentTurn(
            turn_number=1,
            reasoning="I need to write a Python function to parse YAML files. "
                     "The task suggests using yamlparser library.",
            action="Attempt to import yamlparser",
            tool="run_python",
            tool_input={"code": "import yamlparser"},
            tool_output="ModuleNotFoundError: No module named 'yamlparser'"
        )
        trace.add_turn(turn1)

        turn2 = AgentTurn(
            turn_number=2,
            reasoning="yamlparser doesn't exist. I should search for the correct library.",
            action="Search for Python YAML libraries",
            tool="search_docs",
            tool_input={"query": "python yaml parsing"},
            tool_output="PyYAML is the standard library for YAML parsing in Python. "
                       "Use yaml.safe_load() to parse YAML files safely."
        )
        trace.add_turn(turn2)

        turn3 = AgentTurn(
            turn_number=3,
            reasoning="I'll use the yaml module with safe_load() method.",
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


async def main():
    """
    Example usage of AVER Green Agent
    """
    print("="*80)
    print("AVER Green Agent - Example Usage")
    print("="*80)
    print()

    # Initialize green agent
    green_agent = AVERGreenAgent(
        tasks_dir="tasks",
        results_dir="results"
    )

    # Show task statistics
    stats = green_agent.task_suite.get_statistics()
    print("Task Suite Statistics:")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  By category: {stats['by_category']}")
    print()

    # Create mock purple agent for testing
    print("Testing with MockPurpleAgent...")
    print()

    # Run assessment on specific task
    results = await green_agent.assess_agent(
        agent_url="http://localhost:8000",  # Placeholder
        agent_id="mock_agent_v1",
        task_id="aver_hallucination_code_api_2_001"
    )

    print("\nAssessment complete!")


if __name__ == "__main__":
    asyncio.run(main())
