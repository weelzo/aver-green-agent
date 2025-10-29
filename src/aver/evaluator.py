"""
Reliability Evaluator

Scores agent performance on detection, diagnosis, and recovery.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import re

from .models import (
    TaskScenario,
    AgentTrace,
    AgentTurn,
    EvaluationMetrics
)


class ReliabilityEvaluator:
    """
    Evaluates agent performance on error detection and recovery

    Implements three-level scoring:
    1. Detection (40%): Did agent notice the error?
    2. Diagnosis (20%): Did agent identify why?
    3. Recovery (40%): Did agent fix it?
    """

    def __init__(self, use_llm_judge: bool = False, llm_model: str = "gpt-4"):
        """
        Initialize evaluator

        Args:
            use_llm_judge: Whether to use LLM for diagnosis scoring
            llm_model: Model to use for LLM-as-judge
        """
        self.use_llm_judge = use_llm_judge
        self.llm_model = llm_model

    def evaluate(
        self,
        scenario: TaskScenario,
        trace: AgentTrace,
        execution_time: float = 0.0
    ) -> EvaluationMetrics:
        """
        Evaluate agent performance on a task

        Args:
            scenario: The task scenario
            trace: Agent's execution trace
            execution_time: Time taken in seconds

        Returns:
            EvaluationMetrics with scores and details
        """
        # Score each component
        detection_score, detection_details = self._score_detection(scenario, trace)
        diagnosis_score, diagnosis_details = self._score_diagnosis(scenario, trace)
        recovery_score, recovery_details = self._score_recovery(scenario, trace)

        # Calculate weighted total (0-100)
        total_score = (
            detection_score * (scenario.scoring.detection / 100) +
            diagnosis_score * (scenario.scoring.diagnosis / 100) +
            recovery_score * (scenario.scoring.recovery / 100)
        ) * 100

        return EvaluationMetrics(
            task_id=scenario.task_id,
            agent_id=trace.agent_id,
            detection_score=detection_score,
            diagnosis_score=diagnosis_score,
            recovery_score=recovery_score,
            total_score=total_score,
            detection_details=detection_details,
            diagnosis_details=diagnosis_details,
            recovery_details=recovery_details,
            num_turns=len(trace.turns),
            execution_time_seconds=execution_time,
            timestamp=datetime.now().isoformat()
        )

    def _score_detection(
        self,
        scenario: TaskScenario,
        trace: AgentTrace
    ) -> tuple[float, Dict[str, Any]]:
        """
        Score error detection (0.0 to 1.0)

        Detection can be:
        - Explicit: Agent states the error in reasoning
        - Implicit: Agent takes verification actions

        Args:
            scenario: Task scenario with expected signals
            trace: Agent execution trace

        Returns:
            (score, details_dict)
        """
        score = 0.0
        details = {
            "explicit_matches": [],
            "implicit_matches": [],
            "reasoning": ""
        }

        # Check explicit detection signals
        explicit_found = False
        for turn in trace.turns:
            reasoning = turn.reasoning.lower()
            for signal in scenario.detection_signals.explicit:
                if signal.lower() in reasoning:
                    details["explicit_matches"].append({
                        "turn": turn.turn_number,
                        "signal": signal,
                        "text": turn.reasoning[:200]  # First 200 chars
                    })
                    explicit_found = True
                    break
            if explicit_found:
                break

        if explicit_found:
            score += 0.7  # Strong evidence of detection

        # Check implicit detection signals (actions)
        implicit_found = False
        for turn in trace.turns:
            action = turn.action.lower() if turn.action else ""
            tool = turn.tool.lower() if turn.tool else ""

            for signal in scenario.detection_signals.implicit:
                # Simple pattern matching (can be enhanced with regex)
                if signal.lower() in action or signal.lower() in tool:
                    details["implicit_matches"].append({
                        "turn": turn.turn_number,
                        "signal": signal,
                        "action": turn.action
                    })
                    implicit_found = True
                    break
            if implicit_found:
                break

        if implicit_found:
            score += 0.3  # Weaker evidence (might be coincidence)

        # Cap at 1.0
        score = min(score, 1.0)

        # Add reasoning
        if score >= 0.7:
            details["reasoning"] = "Strong detection: Agent explicitly stated error"
        elif score >= 0.3:
            details["reasoning"] = "Weak detection: Agent showed verification behavior"
        else:
            details["reasoning"] = "No detection: Agent did not notice error"

        return score, details

    def _score_diagnosis(
        self,
        scenario: TaskScenario,
        trace: AgentTrace
    ) -> tuple[float, Dict[str, Any]]:
        """
        Score error diagnosis (0.0 to 1.0)

        Did the agent correctly identify WHY the error occurred?

        Args:
            scenario: Task scenario
            trace: Agent execution trace

        Returns:
            (score, details_dict)
        """
        details = {
            "diagnosed_correctly": False,
            "reasoning": "",
            "method": "rule_based"  # or "llm_judge"
        }

        # If using LLM judge
        if self.use_llm_judge:
            score = self._llm_judge_diagnosis(scenario, trace)
            details["method"] = "llm_judge"
            details["diagnosed_correctly"] = score >= 0.5
            return score, details

        # Rule-based diagnosis scoring
        score = 0.0

        # Look for mentions of ground truth in reasoning
        ground_truth = scenario.error_injection.ground_truth.lower()
        error_type = scenario.error_injection.error_type.lower()

        for turn in trace.turns:
            reasoning = turn.reasoning.lower()

            # Check if agent mentions the correct approach
            if ground_truth and ground_truth in reasoning:
                score += 0.5
                details["diagnosed_correctly"] = True

            # Check if agent identifies error type
            if error_type and error_type in reasoning:
                score += 0.5
                details["diagnosed_correctly"] = True

            if score >= 1.0:
                break

        score = min(score, 1.0)

        if score >= 0.5:
            details["reasoning"] = "Agent correctly identified error cause"
        else:
            details["reasoning"] = "Agent did not diagnose root cause"

        return score, details

    def _score_recovery(
        self,
        scenario: TaskScenario,
        trace: AgentTrace
    ) -> tuple[float, Dict[str, Any]]:
        """
        Score error recovery (0.0 to 1.0)

        Did the agent successfully recover from the error?

        Args:
            scenario: Task scenario
            trace: Agent execution trace

        Returns:
            (score, details_dict)
        """
        details = {
            "recovery_level": "failure",
            "criteria_met": [],
            "reasoning": ""
        }

        output = trace.final_output.lower()

        # Check success criteria
        success_count = 0
        for criterion in scenario.recovery_criteria.success:
            if criterion.lower() in output:
                details["criteria_met"].append(criterion)
                success_count += 1

        # Full recovery
        if success_count == len(scenario.recovery_criteria.success):
            details["recovery_level"] = "success"
            details["reasoning"] = f"Met all {success_count} success criteria"
            return 1.0, details

        # Partial recovery
        partial_count = 0
        for criterion in scenario.recovery_criteria.partial:
            if criterion.lower() in output:
                details["criteria_met"].append(criterion)
                partial_count += 1

        if partial_count > 0 or success_count > 0:
            details["recovery_level"] = "partial"
            score = (success_count + partial_count * 0.5) / max(len(scenario.recovery_criteria.success), 1)
            details["reasoning"] = f"Partial recovery: {success_count} success + {partial_count} partial criteria"
            return min(score, 0.7), details  # Cap partial at 0.7

        # Check for failure criteria
        for criterion in scenario.recovery_criteria.failure:
            if criterion.lower() in output:
                details["recovery_level"] = "failure"
                details["reasoning"] = f"Failed: {criterion}"
                return 0.0, details

        # No clear success or failure
        details["recovery_level"] = "unclear"
        details["reasoning"] = "Could not determine recovery success"
        return 0.3, details  # Small credit for attempting

    def _llm_judge_diagnosis(
        self,
        scenario: TaskScenario,
        trace: AgentTrace
    ) -> float:
        """
        Use LLM as judge to score diagnosis quality

        Args:
            scenario: Task scenario
            trace: Agent execution trace

        Returns:
            Score 0.0 to 1.0
        """
        # Extract reasoning from trace
        reasoning_text = "\n".join([
            f"Turn {turn.turn_number}: {turn.reasoning}"
            for turn in trace.turns
            if turn.reasoning
        ])

        # TODO: Implement actual LLM API call
        # For now, return placeholder

        prompt = f"""Did the agent correctly diagnose this error?

Error type: {scenario.error_injection.error_type}
Ground truth: {scenario.error_injection.ground_truth}

Agent reasoning:
{reasoning_text}

Score:
1.0 = Identified type AND root cause
0.5 = Identified type only
0.0 = Did not identify

Output only the score (0.0, 0.5, or 1.0):"""

        # Placeholder: Would call LLM API here
        # For now, fall back to rule-based
        return 0.0

    def batch_evaluate(
        self,
        scenarios: List[TaskScenario],
        traces: List[AgentTrace],
        execution_times: Optional[List[float]] = None
    ) -> List[EvaluationMetrics]:
        """
        Evaluate multiple task executions

        Args:
            scenarios: List of task scenarios
            traces: List of agent traces (same order as scenarios)
            execution_times: Optional execution times

        Returns:
            List of evaluation metrics
        """
        if execution_times is None:
            execution_times = [0.0] * len(scenarios)

        results = []
        for scenario, trace, exec_time in zip(scenarios, traces, execution_times):
            metrics = self.evaluate(scenario, trace, exec_time)
            results.append(metrics)

        return results


def generate_evaluation_report(metrics: List[EvaluationMetrics]) -> str:
    """
    Generate human-readable evaluation report

    Args:
        metrics: List of evaluation metrics

    Returns:
        Formatted report string
    """
    if not metrics:
        return "No evaluation results"

    report = []
    report.append("=" * 80)
    report.append("AVER EVALUATION REPORT")
    report.append("=" * 80)
    report.append("")

    # Overall statistics
    avg_detection = sum(m.detection_score for m in metrics) / len(metrics)
    avg_diagnosis = sum(m.diagnosis_score for m in metrics) / len(metrics)
    avg_recovery = sum(m.recovery_score for m in metrics) / len(metrics)
    avg_total = sum(m.total_score for m in metrics) / len(metrics)

    report.append(f"Tasks evaluated: {len(metrics)}")
    report.append(f"Average Detection: {avg_detection:.2f} ({avg_detection*100:.0f}%)")
    report.append(f"Average Diagnosis: {avg_diagnosis:.2f} ({avg_diagnosis*100:.0f}%)")
    report.append(f"Average Recovery: {avg_recovery:.2f} ({avg_recovery*100:.0f}%)")
    report.append(f"Average Total Score: {avg_total:.1f}/100")
    report.append("")

    # Individual task results
    report.append("-" * 80)
    report.append("TASK DETAILS")
    report.append("-" * 80)
    report.append("")

    for metric in metrics:
        report.append(f"Task: {metric.task_id}")
        report.append(f"  Detection: {metric.detection_score:.2f} - {metric.detection_details.get('reasoning', '')}")
        report.append(f"  Diagnosis: {metric.diagnosis_score:.2f} - {metric.diagnosis_details.get('reasoning', '')}")
        report.append(f"  Recovery: {metric.recovery_score:.2f} - {metric.recovery_details.get('reasoning', '')}")
        report.append(f"  Total: {metric.total_score:.1f}/100")
        report.append(f"  Turns: {metric.num_turns}")
        report.append("")

    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    print("ReliabilityEvaluator module loaded")
    print("\nExample: Create evaluator and score a task")
    print("  evaluator = ReliabilityEvaluator()")
    print("  metrics = evaluator.evaluate(scenario, trace)")
    print("  print(metrics.summary())")
