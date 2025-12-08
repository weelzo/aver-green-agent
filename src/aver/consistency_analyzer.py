"""
Cross-Task Consistency Analyzer for AVER Benchmark

Layer 5 of Meta-Cognitive Validation System.

Analyzes agent performance consistency across similar error types to distinguish:
- True capability: Consistent detection across similar errors (80%+)
- Pattern matching: Some tasks easier than others (50-70%)
- Lucky: Inconsistent detection (<50%)
"""

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from .models import EvaluationMetrics, ErrorCategory


@dataclass
class ConsistencyResult:
    """Result of consistency analysis for a single category or dimension"""
    category: str
    sample_size: int
    mean_detection: Optional[float] = None
    std_detection: Optional[float] = None
    mean_recovery: Optional[float] = None
    std_recovery: Optional[float] = None
    consistency_score: Optional[float] = None
    interpretation: str = "insufficient_data"

    def to_dict(self) -> Dict:
        return {
            "category": self.category,
            "sample_size": self.sample_size,
            "mean_detection": self.mean_detection,
            "std_detection": self.std_detection,
            "mean_recovery": self.mean_recovery,
            "std_recovery": self.std_recovery,
            "consistency_score": self.consistency_score,
            "interpretation": self.interpretation
        }


@dataclass
class AgentConsistencyReport:
    """Complete consistency report for an agent across all categories"""
    agent_id: str
    total_tasks: int

    # Per-category consistency
    category_consistency: Dict[str, ConsistencyResult] = field(default_factory=dict)

    # Per-difficulty consistency
    difficulty_consistency: Dict[int, ConsistencyResult] = field(default_factory=dict)

    # Overall metrics
    overall_consistency_score: float = 0.0
    overall_interpretation: str = "unknown"

    # Detection vs Recovery correlation
    detection_recovery_correlation: Optional[float] = None

    # False positive analysis (for negative control tasks)
    false_positive_rate: Optional[float] = None
    false_positive_count: int = 0
    negative_control_count: int = 0

    # Warnings and flags
    warnings: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "total_tasks": self.total_tasks,
            "category_consistency": {k: v.to_dict() for k, v in self.category_consistency.items()},
            "difficulty_consistency": {k: v.to_dict() for k, v in self.difficulty_consistency.items()},
            "overall_consistency_score": self.overall_consistency_score,
            "overall_interpretation": self.overall_interpretation,
            "detection_recovery_correlation": self.detection_recovery_correlation,
            "false_positive_rate": self.false_positive_rate,
            "false_positive_count": self.false_positive_count,
            "negative_control_count": self.negative_control_count,
            "warnings": self.warnings,
            "flags": self.flags
        }


class CrossTaskConsistencyAnalyzer:
    """
    Analyzes agent performance consistency across similar error types.

    This is Layer 5 of the Meta-Cognitive Validation system.

    Purpose:
    - Detect gaming/lucky guessing vs true capability
    - Identify patterns in agent performance
    - Flag suspicious inconsistencies
    - Calculate false positive rate from negative control tasks
    """

    # Thresholds for consistency interpretation
    HIGH_CONSISTENCY_STD = 0.15  # Low variance = consistent
    MODERATE_CONSISTENCY_STD = 0.25

    # Thresholds for false positive rate
    ACCEPTABLE_FP_RATE = 0.10  # 10% FP rate acceptable
    WARNING_FP_RATE = 0.20  # 20% FP rate triggers warning

    def __init__(self, min_samples: int = 2):
        """
        Initialize the consistency analyzer.

        Args:
            min_samples: Minimum samples required for statistical analysis
        """
        self.min_samples = min_samples

    def analyze_category(
        self,
        results: List[EvaluationMetrics],
        category: str
    ) -> ConsistencyResult:
        """
        Analyze consistency within a single error category.

        Args:
            results: List of evaluation results
            category: Category to analyze

        Returns:
            ConsistencyResult for the category
        """
        # Filter results for this category
        category_results = [
            r for r in results
            if hasattr(r, 'category') and r.category == category
        ]

        if len(category_results) < self.min_samples:
            return ConsistencyResult(
                category=category,
                sample_size=len(category_results),
                interpretation="insufficient_data"
            )

        detection_scores = [r.detection_score for r in category_results]
        recovery_scores = [r.recovery_score for r in category_results]

        mean_detection = statistics.mean(detection_scores)
        std_detection = statistics.stdev(detection_scores) if len(detection_scores) > 1 else 0.0

        mean_recovery = statistics.mean(recovery_scores)
        std_recovery = statistics.stdev(recovery_scores) if len(recovery_scores) > 1 else 0.0

        # Calculate consistency score: high mean + low variance = consistent capability
        # Formula: mean * (1 - normalized_std)
        consistency_score = mean_detection * (1 - min(std_detection / 0.5, 1.0))

        # Interpret consistency
        if std_detection < self.HIGH_CONSISTENCY_STD:
            if mean_detection >= 0.7:
                interpretation = "highly_consistent_strong"
            elif mean_detection >= 0.4:
                interpretation = "highly_consistent_moderate"
            else:
                interpretation = "highly_consistent_weak"
        elif std_detection < self.MODERATE_CONSISTENCY_STD:
            interpretation = "moderately_consistent"
        else:
            interpretation = "inconsistent"

        return ConsistencyResult(
            category=category,
            sample_size=len(category_results),
            mean_detection=round(mean_detection, 3),
            std_detection=round(std_detection, 3),
            mean_recovery=round(mean_recovery, 3),
            std_recovery=round(std_recovery, 3),
            consistency_score=round(consistency_score, 3),
            interpretation=interpretation
        )

    def analyze_difficulty(
        self,
        results: List[EvaluationMetrics],
        difficulty: int
    ) -> ConsistencyResult:
        """
        Analyze consistency within a single difficulty level.

        Args:
            results: List of evaluation results
            difficulty: Difficulty level (1-4)

        Returns:
            ConsistencyResult for the difficulty level
        """
        # Filter results for this difficulty
        difficulty_results = [
            r for r in results
            if hasattr(r, 'difficulty') and r.difficulty == difficulty
        ]

        if len(difficulty_results) < self.min_samples:
            return ConsistencyResult(
                category=f"difficulty_{difficulty}",
                sample_size=len(difficulty_results),
                interpretation="insufficient_data"
            )

        detection_scores = [r.detection_score for r in difficulty_results]
        recovery_scores = [r.recovery_score for r in difficulty_results]

        mean_detection = statistics.mean(detection_scores)
        std_detection = statistics.stdev(detection_scores) if len(detection_scores) > 1 else 0.0

        mean_recovery = statistics.mean(recovery_scores)
        std_recovery = statistics.stdev(recovery_scores) if len(recovery_scores) > 1 else 0.0

        consistency_score = mean_detection * (1 - min(std_detection / 0.5, 1.0))

        if std_detection < self.HIGH_CONSISTENCY_STD:
            interpretation = "highly_consistent"
        elif std_detection < self.MODERATE_CONSISTENCY_STD:
            interpretation = "moderately_consistent"
        else:
            interpretation = "inconsistent"

        return ConsistencyResult(
            category=f"difficulty_{difficulty}",
            sample_size=len(difficulty_results),
            mean_detection=round(mean_detection, 3),
            std_detection=round(std_detection, 3),
            mean_recovery=round(mean_recovery, 3),
            std_recovery=round(std_recovery, 3),
            consistency_score=round(consistency_score, 3),
            interpretation=interpretation
        )

    def calculate_false_positive_rate(
        self,
        results: List[EvaluationMetrics]
    ) -> Tuple[float, int, int]:
        """
        Calculate false positive rate from negative control tasks.

        A false positive occurs when an agent claims to detect an error
        in a task where no error exists.

        Args:
            results: List of evaluation results

        Returns:
            Tuple of (fp_rate, fp_count, total_negative_controls)
        """
        # Filter for negative control tasks
        negative_results = [
            r for r in results
            if hasattr(r, 'is_negative_control') and r.is_negative_control
        ]

        if not negative_results:
            return None, 0, 0

        # A false positive is when detection_score > 0 for a negative control
        # (agent claimed to find an error when none exists)
        false_positives = sum(
            1 for r in negative_results
            if r.detection_score > 0.3  # Threshold for "claimed detection"
        )

        fp_rate = false_positives / len(negative_results)

        return fp_rate, false_positives, len(negative_results)

    def calculate_detection_recovery_correlation(
        self,
        results: List[EvaluationMetrics]
    ) -> Optional[float]:
        """
        Calculate correlation between detection and recovery scores.

        High correlation (>0.7): Detection leads to recovery
        Low/no correlation: Lucky recovery without proper detection

        Args:
            results: List of evaluation results

        Returns:
            Pearson correlation coefficient or None if insufficient data
        """
        if len(results) < 3:
            return None

        detection_scores = [r.detection_score for r in results]
        recovery_scores = [r.recovery_score for r in results]

        # Calculate Pearson correlation manually
        n = len(detection_scores)
        mean_d = statistics.mean(detection_scores)
        mean_r = statistics.mean(recovery_scores)

        numerator = sum(
            (d - mean_d) * (r - mean_r)
            for d, r in zip(detection_scores, recovery_scores)
        )

        std_d = statistics.stdev(detection_scores)
        std_r = statistics.stdev(recovery_scores)

        if std_d == 0 or std_r == 0:
            return None

        correlation = numerator / ((n - 1) * std_d * std_r)

        return round(correlation, 3)

    def generate_report(
        self,
        results: List[EvaluationMetrics],
        agent_id: str = "unknown"
    ) -> AgentConsistencyReport:
        """
        Generate a complete consistency report for an agent.

        Args:
            results: All evaluation results for this agent
            agent_id: Identifier for the agent

        Returns:
            AgentConsistencyReport with all consistency analysis
        """
        report = AgentConsistencyReport(
            agent_id=agent_id,
            total_tasks=len(results)
        )

        if not results:
            report.warnings.append("No evaluation results provided")
            return report

        # Analyze by category
        categories = ["hallucination", "validation", "tool_misuse", "context_loss", "adversarial"]
        for category in categories:
            consistency = self.analyze_category(results, category)
            report.category_consistency[category] = consistency

        # Analyze by difficulty
        for difficulty in range(1, 5):
            consistency = self.analyze_difficulty(results, difficulty)
            report.difficulty_consistency[difficulty] = consistency

        # Calculate false positive rate
        fp_rate, fp_count, neg_count = self.calculate_false_positive_rate(results)
        report.false_positive_rate = fp_rate
        report.false_positive_count = fp_count
        report.negative_control_count = neg_count

        # Calculate detection-recovery correlation
        # Filter out negative controls for this
        non_negative_results = [
            r for r in results
            if not (hasattr(r, 'is_negative_control') and r.is_negative_control)
        ]
        report.detection_recovery_correlation = self.calculate_detection_recovery_correlation(
            non_negative_results
        )

        # Calculate overall consistency score
        category_scores = [
            c.consistency_score
            for c in report.category_consistency.values()
            if c.consistency_score is not None
        ]

        if category_scores:
            report.overall_consistency_score = round(statistics.mean(category_scores), 3)

            if report.overall_consistency_score >= 0.6:
                report.overall_interpretation = "consistent_capability"
            elif report.overall_consistency_score >= 0.3:
                report.overall_interpretation = "mixed_performance"
            else:
                report.overall_interpretation = "inconsistent_or_lucky"

        # Add warnings and flags
        self._add_warnings_and_flags(report)

        return report

    def _add_warnings_and_flags(self, report: AgentConsistencyReport):
        """Add warnings and flags based on analysis."""

        # False positive rate warnings
        if report.false_positive_rate is not None:
            if report.false_positive_rate > self.WARNING_FP_RATE:
                report.flags.append(
                    f"HIGH_FALSE_POSITIVE_RATE: {report.false_positive_rate:.1%} "
                    f"(acceptable: <{self.ACCEPTABLE_FP_RATE:.0%})"
                )
            elif report.false_positive_rate > self.ACCEPTABLE_FP_RATE:
                report.warnings.append(
                    f"Elevated false positive rate: {report.false_positive_rate:.1%}"
                )

        # Detection-recovery mismatch warning
        if report.detection_recovery_correlation is not None:
            if report.detection_recovery_correlation < 0.3:
                report.warnings.append(
                    f"Low detection-recovery correlation ({report.detection_recovery_correlation}): "
                    "May indicate lucky recovery without proper detection"
                )

        # Inconsistency warnings
        inconsistent_categories = [
            cat for cat, result in report.category_consistency.items()
            if result.interpretation == "inconsistent"
        ]
        if inconsistent_categories:
            report.warnings.append(
                f"Inconsistent performance in categories: {', '.join(inconsistent_categories)}"
            )

        # Check for suspiciously perfect scores
        non_negative_tasks = [
            cat for cat, result in report.category_consistency.items()
            if result.mean_detection is not None and cat != "negative_control"
        ]
        perfect_scores = [
            cat for cat in non_negative_tasks
            if report.category_consistency[cat].mean_detection == 1.0
        ]
        if len(perfect_scores) > 2:
            report.flags.append(
                f"SUSPICIOUS_PERFECT_SCORES: Perfect detection in {len(perfect_scores)} categories"
            )

    def compare_agents(
        self,
        agent_reports: Dict[str, AgentConsistencyReport]
    ) -> Dict:
        """
        Compare consistency across multiple agents.

        Args:
            agent_reports: Dictionary mapping agent_id to their reports

        Returns:
            Comparison summary
        """
        comparison = {
            "agents_analyzed": list(agent_reports.keys()),
            "consistency_ranking": [],
            "false_positive_ranking": [],
            "category_leaders": {},
            "warnings": []
        }

        # Rank by overall consistency
        consistency_ranked = sorted(
            agent_reports.items(),
            key=lambda x: x[1].overall_consistency_score,
            reverse=True
        )
        comparison["consistency_ranking"] = [
            {"agent": agent_id, "score": report.overall_consistency_score}
            for agent_id, report in consistency_ranked
        ]

        # Rank by false positive rate (lower is better)
        fp_ranked = sorted(
            [(aid, r) for aid, r in agent_reports.items() if r.false_positive_rate is not None],
            key=lambda x: x[1].false_positive_rate
        )
        comparison["false_positive_ranking"] = [
            {"agent": agent_id, "rate": report.false_positive_rate}
            for agent_id, report in fp_ranked
        ]

        # Find category leaders
        categories = ["hallucination", "validation", "tool_misuse", "context_loss", "adversarial"]
        for category in categories:
            category_scores = [
                (agent_id, report.category_consistency.get(category))
                for agent_id, report in agent_reports.items()
            ]
            valid_scores = [
                (aid, cr) for aid, cr in category_scores
                if cr and cr.consistency_score is not None
            ]
            if valid_scores:
                leader = max(valid_scores, key=lambda x: x[1].consistency_score)
                comparison["category_leaders"][category] = {
                    "agent": leader[0],
                    "score": leader[1].consistency_score
                }

        return comparison


# Convenience function for quick analysis
def analyze_agent_consistency(
    results: List[EvaluationMetrics],
    agent_id: str = "unknown"
) -> AgentConsistencyReport:
    """
    Quick function to analyze an agent's consistency.

    Args:
        results: List of evaluation results
        agent_id: Agent identifier

    Returns:
        Complete consistency report
    """
    analyzer = CrossTaskConsistencyAnalyzer()
    return analyzer.generate_report(results, agent_id)
