"""
Universal Mock Purple Agent

A mock agent that can handle ALL AVER benchmark tasks with configurable behavior profiles.
Generates realistic traces for testing and demonstration purposes.

Usage:
    from aver.mock_agent import UniversalMockPurpleAgent, MockAgentConfig

    config = MockAgentConfig(agent_id="test_agent")
    agent = UniversalMockPurpleAgent(config)
    trace = await agent.execute_task(task_scenario)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import hashlib
import random

from .models import TaskScenario, AgentTrace, AgentTurn, ErrorCategory


class BehaviorProfile(Enum):
    """Defines how the mock agent will behave on a task"""
    EXPERT = "expert"       # Score 80-100: Pre-execution detection, full recovery
    COMPETENT = "competent" # Score 50-70: Post-execution detection, partial recovery
    NOVICE = "novice"       # Score 20-40: Implicit detection only, incomplete
    FAILING = "failing"     # Score 0-15: No detection, proceeds with error


@dataclass
class MockAgentConfig:
    """Configuration for mock agent behavior"""
    agent_id: str = "mock_universal_agent"
    model_name: str = "mock-gpt-4"

    # Behavior distribution (probabilities for each behavior)
    expert_rate: float = 0.50     # 50% of tasks get expert behavior
    competent_rate: float = 0.25  # 25% competent
    novice_rate: float = 0.15     # 15% novice
    # failing_rate is implicit: 1 - sum = 10%

    # Reproducibility
    deterministic: bool = True    # Use task_id hash for consistent results
    seed: Optional[int] = 42

    # Output verbosity
    verbose: bool = True
    num_turns: int = 3  # Default turns per trace


class TraceGenerator:
    """
    Generates task-aware traces based on TaskScenario structure.

    Uses the task's detection_signals, recovery_criteria, and error_injection
    to generate appropriate traces that will score correctly with the evaluator.
    """

    def __init__(self, config: MockAgentConfig):
        self.config = config

    def generate(self, task: TaskScenario, behavior: BehaviorProfile) -> AgentTrace:
        """
        Generate trace appropriate for the behavior profile.

        Args:
            task: Full TaskScenario with all metadata
            behavior: How the agent should behave

        Returns:
            AgentTrace with appropriate turns and final output
        """
        trace = AgentTrace(
            task_id=task.task_id,
            agent_id=self.config.agent_id,
            model_name=self.config.model_name
        )

        if behavior == BehaviorProfile.EXPERT:
            self._generate_expert_trace(task, trace)
        elif behavior == BehaviorProfile.COMPETENT:
            self._generate_competent_trace(task, trace)
        elif behavior == BehaviorProfile.NOVICE:
            self._generate_novice_trace(task, trace)
        else:
            self._generate_failing_trace(task, trace)

        return trace

    def _extract_task_info(self, task: TaskScenario) -> Dict[str, Any]:
        """Extract key information from task for template filling"""
        error_data = task.error_injection.error_data
        return {
            "category": task.category.value,
            "difficulty": task.difficulty.value,
            "misleading": error_data.get("misleading_text", "the suggested approach"),
            "ground_truth": task.error_injection.ground_truth or "the correct approach",
            "error_type": task.error_injection.error_type or "error",
            "explicit_signals": task.detection_signals.explicit or ["error detected"],
            "implicit_signals": task.detection_signals.implicit or ["verify", "check"],
            "success_criteria": task.recovery_criteria.success or ["correct solution"],
            "partial_criteria": task.recovery_criteria.partial or ["partial solution"],
            "failure_criteria": task.recovery_criteria.failure or ["incorrect"],
            "tools": [t.name for t in task.tools] if task.tools else ["run_python"],
            "expected_output": task.expected_output or "",
            "task_description": task.task_description[:200] if task.task_description else ""
        }

    def _extract_key_term(self, text: str) -> str:
        """Extract key term from text for search queries"""
        if not text:
            return "term"
        words = text.split()
        stopwords = {"the", "use", "with", "for", "and", "a", "an", "to", "is", "in", "it"}
        for word in words:
            clean_word = word.strip(".,!?()[]{}\"'")
            if len(clean_word) > 3 and clean_word.lower() not in stopwords:
                return clean_word
        return words[0] if words else "term"

    def _generate_expert_trace(self, task: TaskScenario, trace: AgentTrace):
        """
        Expert: Pre-execution detection, correct diagnosis, full recovery.

        Expected score: 80-100
        - Detection: 0.7+ (explicit signal BEFORE execution)
        - Diagnosis: 1.0 (identifies ground truth)
        - Recovery: 1.0 (all success criteria met)
        """
        info = self._extract_task_info(task)

        # Turn 1: Pre-execution detection (BEFORE running any code)
        explicit_signal = info["explicit_signals"][0]
        key_term = self._extract_key_term(info["misleading"])

        turn1 = AgentTurn(
            turn_number=1,
            reasoning=f"Looking at this task, I need to verify the suggested approach first. "
                      f"The task mentions: '{info['misleading'][:80]}...'. "
                      f"Let me check if this is correct before implementing. "
                      f"After reviewing: {explicit_signal}. This seems problematic.",
            action="Verify task requirements before implementation",
            tool="search_docs" if "search_docs" in info["tools"] else info["tools"][0],
            tool_input={"query": f"verify {key_term}"},
            tool_output=f"Warning: {key_term} may not be the correct approach. Consider alternatives."
        )
        trace.add_turn(turn1)

        # Turn 2: Diagnosis with ground truth
        turn2 = AgentTurn(
            turn_number=2,
            reasoning=f"I've identified the issue. The error type is: {info['error_type']}. "
                      f"The correct approach is: {info['ground_truth']}. "
                      f"I will implement the solution using this correct method instead.",
            action="Plan correct implementation based on diagnosis",
            tool=None,
            tool_input=None,
            tool_output=None
        )
        trace.add_turn(turn2)

        # Turn 3: Recovery implementation
        exec_tool = "run_python" if "run_python" in info["tools"] else info["tools"][0]
        turn3 = AgentTurn(
            turn_number=3,
            reasoning="Implementing the correct solution based on the identified fix.",
            action="Implement correct solution",
            tool=exec_tool,
            tool_input={"code": info["expected_output"][:500] if info["expected_output"] else "# Correct implementation"},
            tool_output="Execution successful"
        )
        trace.add_turn(turn3)

        # Final output includes ALL success criteria
        trace.final_output = self._build_success_output(task, info)

    def _generate_competent_trace(self, task: TaskScenario, trace: AgentTrace):
        """
        Competent: Post-execution detection (trial-and-error), partial recovery.

        Expected score: 50-70
        - Detection: 0.35 (explicit signal AFTER execution - penalty applied)
        - Diagnosis: 0.5 (partial identification)
        - Recovery: 0.5-0.7 (partial criteria met)
        """
        info = self._extract_task_info(task)

        # Turn 1: Naive execution attempt (triggers error)
        exec_tool = "run_python" if "run_python" in info["tools"] else info["tools"][0]
        turn1 = AgentTurn(
            turn_number=1,
            reasoning=f"I'll implement the solution as described in the task using {info['misleading'][:50]}...",
            action="Attempt implementation as specified",
            tool=exec_tool,
            tool_input={"code": f"# Attempting: {info['misleading'][:50]}"},
            tool_output=f"Error: {info['explicit_signals'][0] if info['explicit_signals'] else 'execution failed'}"
        )
        trace.add_turn(turn1)

        # Turn 2: Post-execution detection (after seeing error)
        explicit_signal = info["explicit_signals"][0] if info["explicit_signals"] else "detected issue"
        turn2 = AgentTurn(
            turn_number=2,
            reasoning=f"The execution failed. I see that: {explicit_signal}. "
                      f"I need to find an alternative approach. "
                      f"The issue seems to be related to {info['error_type']}.",
            action="Search for correct approach after error",
            tool="search_docs" if "search_docs" in info["tools"] else info["tools"][0],
            tool_input={"query": info["ground_truth"][:50]},
            tool_output=f"Found: {info['ground_truth']}"
        )
        trace.add_turn(turn2)

        # Turn 3: Partial recovery
        turn3 = AgentTurn(
            turn_number=3,
            reasoning="Implementing fix based on search results.",
            action="Implement alternative solution",
            tool=exec_tool,
            tool_input={"code": "# Partial fix implementation"},
            tool_output="Partial success"
        )
        trace.add_turn(turn3)

        # Final output: Some success criteria + some partial
        trace.final_output = self._build_partial_output(task, info)

    def _generate_novice_trace(self, task: TaskScenario, trace: AgentTrace):
        """
        Novice: Weak detection (implicit only), incomplete recovery.

        Expected score: 20-40
        - Detection: 0.3 (implicit signal only)
        - Diagnosis: 0.0-0.5 (vague or missing)
        - Recovery: 0.3-0.5 (partial criteria only)
        """
        info = self._extract_task_info(task)

        # Turn 1: Attempt and fail
        exec_tool = "run_python" if "run_python" in info["tools"] else info["tools"][0]
        turn1 = AgentTurn(
            turn_number=1,
            reasoning="Implementing the task as described.",
            action="Execute implementation",
            tool=exec_tool,
            tool_input={"code": f"# {info['misleading'][:30]}"},
            tool_output="Error occurred"
        )
        trace.add_turn(turn1)

        # Turn 2: Vague implicit detection (action contains implicit signal)
        implicit_signal = info["implicit_signals"][0] if info["implicit_signals"] else "check docs"
        turn2 = AgentTurn(
            turn_number=2,
            reasoning="Something went wrong. Let me check what's happening.",
            action=implicit_signal,  # Implicit signal appears in action
            tool="search_docs" if "search_docs" in info["tools"] else info["tools"][0],
            tool_input={"query": "how to fix error"},
            tool_output="General documentation results"
        )
        trace.add_turn(turn2)

        # Turn 3: Incomplete fix
        turn3 = AgentTurn(
            turn_number=3,
            reasoning="Trying a different approach.",
            action="Attempt fix",
            tool=exec_tool,
            tool_input={"code": "# Incomplete implementation"},
            tool_output="Partial output"
        )
        trace.add_turn(turn3)

        # Final output: Only partial criteria
        trace.final_output = self._build_novice_output(task, info)

    def _generate_failing_trace(self, task: TaskScenario, trace: AgentTrace):
        """
        Failing: No detection, proceeds with error.

        Expected score: 0-15
        - Detection: 0.0 (no signals)
        - Diagnosis: 0.0 (no identification)
        - Recovery: 0.0 (failure criteria triggered)
        """
        info = self._extract_task_info(task)

        # Turn 1: Proceed without checking
        exec_tool = "run_python" if "run_python" in info["tools"] else info["tools"][0]
        turn1 = AgentTurn(
            turn_number=1,
            reasoning="I will implement exactly as the task describes. This looks straightforward.",
            action="Implement as specified without verification",
            tool=exec_tool,
            tool_input={"code": f"# Using {info['misleading'][:30]}"},
            tool_output="Executed"
        )
        trace.add_turn(turn1)

        # Turn 2: Continue with bad approach
        turn2 = AgentTurn(
            turn_number=2,
            reasoning="Continuing with the implementation as planned.",
            action="Complete implementation",
            tool=exec_tool,
            tool_input={"code": "# Completing with original approach"},
            tool_output="Done"
        )
        trace.add_turn(turn2)

        # Final output: Contains failure criteria
        trace.final_output = self._build_failure_output(task, info)

    def _build_success_output(self, task: TaskScenario, info: Dict[str, Any]) -> str:
        """Build final output that contains all success criteria"""
        # Use expected_output if available
        if task.expected_output:
            return task.expected_output

        # Otherwise, join all success criteria
        parts = []
        for criterion in info["success_criteria"]:
            parts.append(criterion)

        return "\n".join(parts) if parts else "Correct implementation completed"

    def _build_partial_output(self, task: TaskScenario, info: Dict[str, Any]) -> str:
        """Build output with some success + some partial criteria"""
        parts = []

        # Include first success criterion if available
        if info["success_criteria"]:
            parts.append(info["success_criteria"][0])

        # Include partial criteria
        parts.extend(info["partial_criteria"])

        return "\n".join(parts) if parts else "Partial implementation"

    def _build_novice_output(self, task: TaskScenario, info: Dict[str, Any]) -> str:
        """Build output with only partial criteria"""
        if info["partial_criteria"]:
            return "\n".join(info["partial_criteria"])
        return "Incomplete solution"

    def _build_failure_output(self, task: TaskScenario, info: Dict[str, Any]) -> str:
        """Build output with failure criteria"""
        if info["failure_criteria"]:
            return info["failure_criteria"][0]

        # Fallback: include misleading text (wrong approach)
        return info["misleading"][:100] if info["misleading"] else "Failed implementation"


class UniversalMockPurpleAgent:
    """
    Universal mock purple agent that handles ALL AVER tasks.

    Generates realistic traces with configurable success rates and behaviors.
    Uses deterministic behavior selection for reproducible demos.

    Usage:
        config = MockAgentConfig(
            agent_id="demo_agent",
            expert_rate=0.5,
            deterministic=True
        )
        agent = UniversalMockPurpleAgent(config)
        trace = await agent.execute_task(task_scenario)
    """

    def __init__(self, config: MockAgentConfig = None):
        """
        Initialize universal mock agent.

        Args:
            config: Configuration for behavior distribution and reproducibility
        """
        self.config = config or MockAgentConfig()
        self.trace_generator = TraceGenerator(self.config)
        self.rng = self._init_rng()

    def _init_rng(self) -> random.Random:
        """Initialize random number generator for non-deterministic mode"""
        rng = random.Random()
        if self.config.seed is not None:
            rng.seed(self.config.seed)
        return rng

    def _get_behavior(self, task: TaskScenario) -> BehaviorProfile:
        """
        Determine behavior profile for this task.

        Uses task_id hash for deterministic, reproducible results.
        Harder tasks (higher difficulty) are less likely to get expert behavior.

        Args:
            task: The task scenario

        Returns:
            BehaviorProfile for this task
        """
        if self.config.deterministic:
            # Use task_id hash for reproducible results
            hash_val = int(hashlib.md5(task.task_id.encode()).hexdigest(), 16) % 100
        else:
            hash_val = self.rng.randint(0, 99)

        # Adjust thresholds by difficulty (harder = less likely to succeed)
        # Difficulty 1: no penalty, Difficulty 4: -15% expert rate
        difficulty_penalty = (task.difficulty.value - 1) * 5

        expert_threshold = int(self.config.expert_rate * 100) - difficulty_penalty
        competent_threshold = expert_threshold + int(self.config.competent_rate * 100)
        novice_threshold = competent_threshold + int(self.config.novice_rate * 100)

        if hash_val < expert_threshold:
            return BehaviorProfile.EXPERT
        elif hash_val < competent_threshold:
            return BehaviorProfile.COMPETENT
        elif hash_val < novice_threshold:
            return BehaviorProfile.NOVICE
        else:
            return BehaviorProfile.FAILING

    async def execute_task(self, task: TaskScenario) -> AgentTrace:
        """
        Execute task and return trace.

        Args:
            task: Full TaskScenario object

        Returns:
            AgentTrace with realistic behavior based on behavior profile
        """
        behavior = self._get_behavior(task)

        if self.config.verbose:
            print(f"      [MockAgent] Behavior: {behavior.value}")

        trace = self.trace_generator.generate(task, behavior)

        return trace


# For backwards compatibility
MockPurpleAgent = UniversalMockPurpleAgent


if __name__ == "__main__":
    import asyncio

    async def test_mock_agent():
        """Test the universal mock agent"""
        from .task_suite import TaskSuite

        print("Testing Universal Mock Purple Agent")
        print("=" * 60)

        # Load tasks
        suite = TaskSuite("tasks")
        suite.load_all_tasks()

        # Create agent
        config = MockAgentConfig(
            agent_id="test_agent",
            deterministic=True,
            verbose=True
        )
        agent = UniversalMockPurpleAgent(config)

        # Test on a few tasks
        stats = {"expert": 0, "competent": 0, "novice": 0, "failing": 0}

        for task in suite.tasks[:10]:
            behavior = agent._get_behavior(task)
            stats[behavior.value] += 1

            trace = await agent.execute_task(task)
            print(f"Task: {task.task_id}")
            print(f"  Behavior: {behavior.value}")
            print(f"  Turns: {len(trace.turns)}")
            print(f"  Final output length: {len(trace.final_output)}")
            print()

        print("Behavior distribution (10 tasks):")
        for behavior, count in stats.items():
            print(f"  {behavior}: {count}")

    asyncio.run(test_mock_agent())
