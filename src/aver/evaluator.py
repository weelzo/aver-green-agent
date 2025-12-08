"""
Reliability Evaluator

Scores agent performance on detection, diagnosis, and recovery.

Enhanced with Two-Pillar Validation:
1. Execution Validity - Recovery validated through code execution tests
2. Meta-Cognitive Validity - Detection/diagnosis validated through cognitive process analysis

STRICT SCORING: Invalid causal chain -> detection & diagnosis scores halved
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import re

from .models import (
    TaskScenario,
    AgentTrace,
    AgentTurn,
    EvaluationMetrics,
    TaskDomain
)
from .execution_validator import RecoveryValidator
from .metacognitive_validator import (
    MetaCognitiveValidator,
    NegativeControlValidator
)


# Category-specific diagnosis patterns for rubric-based scoring
# These patterns indicate the agent correctly identified the error type
DIAGNOSIS_PATTERNS = {
    "hallucination": [
        "doesn't exist", "does not exist", "not found", "no such",
        "cannot import", "import error", "module not found",
        "hallucinated", "made up", "fictional", "invented",
        "not a real", "doesn't have", "no library", "fake"
    ],
    "validation": [
        "invalid", "incorrect", "wrong format", "failed validation",
        "type error", "schema", "constraint", "mismatch",
        "validation failed", "not valid", "malformed",
        "doesn't match", "expected", "required"
    ],
    "tool_misuse": [
        "wrong tool", "incorrect parameter", "not supported",
        "missing argument", "invalid input", "wrong function",
        "parameter error", "argument error", "incompatible",
        "tool doesn't", "can't use", "misused"
    ],
    "context_loss": [
        "forgot", "lost context", "inconsistent", "contradicts",
        "earlier said", "previous", "originally", "before",
        "changed", "different from", "conflicting", "mismatch"
    ],
    "adversarial": [
        "ambiguous", "conflicting", "multiple errors", "unclear",
        "confusing", "misleading", "contradictory", "both",
        "either", "several issues", "many problems"
    ]
}


class ReliabilityEvaluator:
    """
    Evaluates agent performance on error detection and recovery

    Implements three-level scoring:
    1. Detection (40%): Did agent notice the error?
    2. Diagnosis (20%): Did agent identify why?
    3. Recovery (40%): Did agent fix it?

    Enhanced with Two-Pillar Validation:
    - Pillar 1: Execution Validity (recovery via code tests)
    - Pillar 2: Meta-Cognitive Validity (cognitive process validation)
    """

    def __init__(
        self,
        use_llm_judge: bool = False,
        llm_model: str = "gpt-4",
        use_docker: bool = False,
        enable_metacognitive: bool = True
    ):
        """
        Initialize evaluator

        Args:
            use_llm_judge: [DEPRECATED] Whether to use LLM for diagnosis scoring.
                           Competition guidance strongly prefers rubric-based scoring.
                           This option is kept for backward compatibility only.
            llm_model: Model to use for LLM-as-judge (deprecated)
            use_docker: Whether to use Docker sandbox for code execution
            enable_metacognitive: Whether to apply meta-cognitive validation

        Note:
            The AgentBeats competition explicitly discourages LLM-as-judge:
            "provide ground truth, rigorous rubrics for evaluation instead of
            LLM-as-a-judge which oftentimes does not provide high enough accuracy"

            We use rubric-based pattern matching (DIAGNOSIS_PATTERNS) by default.
        """
        self.use_llm_judge = use_llm_judge
        self.llm_model = llm_model
        self.use_docker = use_docker
        self.enable_metacognitive = enable_metacognitive

        # Initialize validators
        self.recovery_validator = RecoveryValidator(use_docker=use_docker)
        self.metacognitive_validator = MetaCognitiveValidator()
        self.negative_control_validator = NegativeControlValidator()

        if use_llm_judge:
            import warnings
            warnings.warn(
                "LLM-as-judge is deprecated. Competition guidance prefers rubric-based scoring. "
                "Consider using use_llm_judge=False (default) for competition compliance.",
                DeprecationWarning,
                stacklevel=2
            )

    def evaluate(
        self,
        scenario: TaskScenario,
        trace: AgentTrace,
        execution_time: float = 0.0
    ) -> EvaluationMetrics:
        """
        Evaluate agent performance on a task with two-pillar validation.

        Pillar 1: Execution Validity (Recovery)
        - For coding tasks with test suites, uses execution-based validation
        - High confidence, deterministic pass/fail

        Pillar 2: Meta-Cognitive Validity (Detection + Diagnosis)
        - Validates cognitive process: detect → diagnose → recover chain
        - Applies multipliers: invalid chain → scores halved

        Args:
            scenario: The task scenario
            trace: Agent's execution trace
            execution_time: Time taken in seconds

        Returns:
            EvaluationMetrics with scores and details
        """
        # Handle negative control tasks differently
        if scenario.is_negative_control():
            return self._evaluate_negative_control(scenario, trace, execution_time)

        # === PILLAR 1: Base Scoring ===
        # Score detection and diagnosis (base scores)
        detection_base, detection_details = self._score_detection(scenario, trace)
        diagnosis_base, diagnosis_details = self._score_diagnosis(scenario, trace)

        # Score recovery - use execution validation for coding tasks
        if scenario.has_execution_tests():
            recovery_base, recovery_details = self._score_recovery_with_execution(
                scenario, trace
            )
        else:
            recovery_base, recovery_details = self._score_recovery(scenario, trace)

        # === PILLAR 2: Meta-Cognitive Validation ===
        if self.enable_metacognitive:
            metacog = self.metacognitive_validator.validate(
                trace, scenario,
                detection_base, diagnosis_base, recovery_base
            )

            # Use adjusted scores
            detection_final = metacog.detection_final
            diagnosis_final = metacog.diagnosis_final
            recovery_final = metacog.recovery_final

            # Add metacognitive details
            detection_details["metacognitive"] = metacog.causal_chain.to_dict()
            detection_details["temporal"] = metacog.temporal.to_dict()
            diagnosis_details["depth"] = metacog.diagnosis_depth.to_dict()
            recovery_details["cognitive_confidence"] = metacog.cognitive_confidence
            recovery_details["cognitive_warning"] = metacog.cognitive_warning

            # Calculate weighted total with final adjusted scores
            total_score = (
                detection_final * (scenario.scoring.detection / 100) +
                diagnosis_final * (scenario.scoring.diagnosis / 100) +
                recovery_final * (scenario.scoring.recovery / 100)
            ) * 100
        else:
            # Use base scores without metacognitive adjustment
            detection_final = detection_base
            diagnosis_final = diagnosis_base
            recovery_final = recovery_base

            total_score = (
                detection_base * (scenario.scoring.detection / 100) +
                diagnosis_base * (scenario.scoring.diagnosis / 100) +
                recovery_base * (scenario.scoring.recovery / 100)
            ) * 100

        return EvaluationMetrics(
            task_id=scenario.task_id,
            agent_id=trace.agent_id,
            detection_score=detection_final,
            diagnosis_score=diagnosis_final,
            recovery_score=recovery_final,
            total_score=total_score,
            detection_details=detection_details,
            diagnosis_details=diagnosis_details,
            recovery_details=recovery_details,
            category=scenario.category.value,
            difficulty=scenario.difficulty.value,
            is_negative_control=scenario.is_negative_control(),
            model_name=trace.model_name,
            num_turns=len(trace.turns),
            execution_time_seconds=execution_time,
            timestamp=datetime.now().isoformat()
        )

    def _evaluate_negative_control(
        self,
        scenario: TaskScenario,
        trace: AgentTrace,
        execution_time: float
    ) -> EvaluationMetrics:
        """
        Evaluate negative control task (task without errors).

        For negative control tasks:
        - Detection score SHOULD be 0 (no error to detect)
        - Recovery score based on task completion

        Args:
            scenario: Negative control task scenario
            trace: Agent trace
            execution_time: Execution time

        Returns:
            EvaluationMetrics
        """
        # Score detection (should be low for negative control)
        detection_base, detection_details = self._score_detection(scenario, trace)

        # Check for false positives
        fp_result = self.negative_control_validator.validate(
            trace, scenario, detection_base
        )

        # Penalize false positives
        if fp_result.get("is_false_positive"):
            detection_final = 0.0  # False positive penalty
            detection_details["false_positive"] = True
            detection_details["fp_signals"] = fp_result.get("false_positive_signals_found", [])
        else:
            detection_final = 0.0  # Correct: no error to detect
            detection_details["false_positive"] = False

        # Diagnosis should be 0 for negative control (no error to diagnose)
        diagnosis_final = 0.0
        diagnosis_details = {"method": "negative_control", "reasoning": "No error to diagnose"}

        # Recovery based on task completion
        if scenario.has_execution_tests():
            recovery_final, recovery_details = self._score_recovery_with_execution(
                scenario, trace
            )
        else:
            recovery_final, recovery_details = self._score_recovery(scenario, trace)

        # For negative control, only recovery matters (40% weight becomes 100%)
        total_score = recovery_final * 100

        return EvaluationMetrics(
            task_id=scenario.task_id,
            agent_id=trace.agent_id,
            detection_score=detection_final,
            diagnosis_score=diagnosis_final,
            recovery_score=recovery_final,
            total_score=total_score,
            detection_details=detection_details,
            diagnosis_details=diagnosis_details,
            recovery_details=recovery_details,
            category=scenario.category.value,
            difficulty=scenario.difficulty.value,
            is_negative_control=True,  # This is a negative control evaluation
            model_name=trace.model_name,
            num_turns=len(trace.turns),
            execution_time_seconds=execution_time,
            timestamp=datetime.now().isoformat()
        )

    def _score_recovery_with_execution(
        self,
        scenario: TaskScenario,
        trace: AgentTrace
    ) -> tuple[float, Dict[str, Any]]:
        """
        Score recovery using execution-based validation.

        Runs test suite against agent's code for deterministic validation.

        Args:
            scenario: Task scenario with execution_validity
            trace: Agent trace

        Returns:
            (score, details_dict)
        """
        score, details = self.recovery_validator.validate(
            trace.final_output, scenario
        )
        return score, details

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

        CRITICAL: Detection must occur BEFORE first execution attempt
        to distinguish from trial-and-error recovery.

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
            "pre_execution": False,
            "reasoning": ""
        }

        # Find first execution attempt (run_python, execute, etc.)
        first_execution_turn = None
        for turn in trace.turns:
            if turn.tool and "run" in turn.tool.lower():
                first_execution_turn = turn.turn_number
                break

        # Check explicit detection signals in reasoning AND action
        explicit_found = False
        explicit_turn = None

        for turn in trace.turns:
            # Check both reasoning and action fields
            reasoning = turn.reasoning.lower() if turn.reasoning else ""
            action = turn.action.lower() if turn.action else ""
            combined_text = f"{reasoning} {action}"

            for signal in scenario.detection_signals.explicit:
                if signal.lower() in combined_text:
                    details["explicit_matches"].append({
                        "turn": turn.turn_number,
                        "signal": signal,
                        "text": (turn.reasoning or turn.action or "")[:200]
                    })
                    explicit_found = True
                    explicit_turn = turn.turn_number
                    break
            if explicit_found:
                break

        if explicit_found:
            # Check if detection happened BEFORE execution (genuine detection)
            # vs AFTER execution (trial-and-error)
            if first_execution_turn is None or explicit_turn < first_execution_turn:
                score += 0.7  # Full credit: detected before trying
                details["pre_execution"] = True
            else:
                # Detected after execution attempt - trial and error
                penalty = scenario.metadata.get('scoring_rules', {}).get('trial_and_error_penalty', 0.5)
                score += 0.7 * penalty  # Reduced credit
                details["pre_execution"] = False
                details["reasoning"] = "Detection after execution (trial-and-error)"

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

        # Look for mentions of ground truth in reasoning, action, and final output
        ground_truth = scenario.error_injection.ground_truth.lower()
        error_type = scenario.error_injection.error_type.lower()

        # Combine all text from trace
        all_text = ""
        for turn in trace.turns:
            if turn.reasoning:
                all_text += turn.reasoning.lower() + " "
            if turn.action:
                all_text += turn.action.lower() + " "
        if trace.final_output:
            all_text += trace.final_output.lower()

        # Check if agent mentions the correct approach (more flexible matching)
        if ground_truth:
            # Split ground truth into key phrases
            key_phrases = ground_truth.split()
            matches = sum(1 for phrase in key_phrases if phrase in all_text)
            if matches >= len(key_phrases) * 0.5:  # At least half the key phrases
                score += 0.5
                details["diagnosed_correctly"] = True

        # Check if agent identifies error type
        if error_type:
            # Check for error type or related terms
            error_terms = [error_type, "doesn't exist", "not found", "incorrect", "wrong"]
            if any(term in all_text for term in error_terms):
                score += 0.5
                details["diagnosed_correctly"] = True

        # NEW: Category-specific pattern matching for enhanced rubric scoring
        # This replaces LLM-as-judge with deterministic pattern matching
        category = scenario.category.value if hasattr(scenario, 'category') else ""
        category_patterns = DIAGNOSIS_PATTERNS.get(category, [])

        # Count how many category-specific patterns the agent mentions
        pattern_matches = []
        for pattern in category_patterns:
            if pattern.lower() in all_text:
                pattern_matches.append(pattern)

        # Bonus score for category-specific diagnosis (up to 0.3)
        if len(pattern_matches) >= 3:
            score += 0.3  # Strong category-specific diagnosis
            details["category_patterns_matched"] = pattern_matches[:5]  # Store first 5
            details["category_diagnosis"] = "strong"
        elif len(pattern_matches) >= 1:
            score += 0.15  # Weak category-specific diagnosis
            details["category_patterns_matched"] = pattern_matches
            details["category_diagnosis"] = "weak"
        else:
            details["category_patterns_matched"] = []
            details["category_diagnosis"] = "none"

        score = min(score, 1.0)

        if score >= 0.7:
            details["reasoning"] = "Agent correctly identified error cause with strong diagnosis"
        elif score >= 0.5:
            details["reasoning"] = "Agent correctly identified error cause"
        elif score >= 0.3:
            details["reasoning"] = "Agent showed some understanding of error type"
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
    Generate human-readable evaluation report with fair scoring.

    Separates error tasks from negative controls for competition-fair metrics.

    Args:
        metrics: List of evaluation metrics

    Returns:
        Formatted report string
    """
    if not metrics:
        return "No evaluation results"

    report = []
    report.append("=" * 80)
    report.append("AVER EVALUATION REPORT (Competition-Fair Scoring)")
    report.append("=" * 80)
    report.append("")

    # Separate error tasks from negative controls
    error_tasks = [m for m in metrics if not m.is_negative_control]
    negative_controls = [m for m in metrics if m.is_negative_control]

    report.append(f"Tasks evaluated: {len(metrics)} ({len(error_tasks)} error + {len(negative_controls)} negative control)")
    report.append("")

    # Error task statistics (what competition judges care about)
    if error_tasks:
        avg_detection = sum(m.detection_score for m in error_tasks) / len(error_tasks)
        avg_diagnosis = sum(m.diagnosis_score for m in error_tasks) / len(error_tasks)
        avg_recovery = sum(m.recovery_score for m in error_tasks) / len(error_tasks)
        avg_total = sum(m.total_score for m in error_tasks) / len(error_tasks)

        report.append("ERROR TASK PERFORMANCE:")
        report.append(f"  Detection: {avg_detection:.2f} ({avg_detection*100:.0f}%)")
        report.append(f"  Diagnosis: {avg_diagnosis:.2f} ({avg_diagnosis*100:.0f}%)")
        report.append(f"  Recovery:  {avg_recovery:.2f} ({avg_recovery*100:.0f}%)")
        report.append(f"  Avg Score: {avg_total:.1f}/100")
    else:
        report.append("ERROR TASK PERFORMANCE: No error tasks")
        avg_total = 0.0

    report.append("")

    # Negative control statistics (false positive rate)
    if negative_controls:
        false_positives = sum(
            1 for m in negative_controls
            if m.detection_details.get('false_positive', False) or m.detection_score > 0.3
        )
        fp_rate = false_positives / len(negative_controls)
        avg_nc_recovery = sum(m.recovery_score for m in negative_controls) / len(negative_controls)

        report.append("NEGATIVE CONTROL PERFORMANCE:")
        report.append(f"  False Positive Rate: {fp_rate:.0%} ({false_positives}/{len(negative_controls)})")
        report.append(f"  Task Completion:     {avg_nc_recovery:.2f} ({avg_nc_recovery*100:.0f}%)")

        # Overall score with FP penalty
        if error_tasks:
            overall_score = avg_total * (1 - fp_rate * 0.5)
            report.append("")
            report.append(f"OVERALL SCORE: {overall_score:.1f}/100 (FP-adjusted)")
    else:
        report.append("NEGATIVE CONTROL PERFORMANCE: No negative control tasks")
        if error_tasks:
            report.append(f"OVERALL SCORE: {avg_total:.1f}/100")

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
