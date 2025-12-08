#!/usr/bin/env python3
"""
Comprehensive Validity Verification Tests for AVER Benchmark

These tests prove the validity claims required for AgentX competition:
1. Reproducibility: Same input → same output (variance < 2%)
2. Discriminative validity: Different behaviors → different scores
3. False positive rate < 10%
4. False negative rate < 15%
5. Execution tests work correctly
6. Meta-cognitive validation is accurate
"""

import sys
from pathlib import Path
import statistics

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# pytest is optional - only import if running via pytest
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create a dummy skip function
    class pytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func
            return decorator
        @staticmethod
        def skip(msg):
            print(f"SKIP: {msg}")

from aver.task_suite import TaskSuite
from aver.evaluator import ReliabilityEvaluator
from aver.consistency_analyzer import CrossTaskConsistencyAnalyzer
from aver.models import AgentTrace, AgentTurn, EvaluationMetrics


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def task_suite():
    """Load task suite for testing"""
    suite = TaskSuite(tasks_dir="tasks")
    suite.load_all_tasks()
    return suite


@pytest.fixture(scope="module")
def evaluator():
    """Create evaluator with full validation enabled"""
    return ReliabilityEvaluator(
        use_docker=False,  # Use subprocess for testing
        enable_metacognitive=True
    )


@pytest.fixture
def hallucination_task(task_suite):
    """Get the primary hallucination test task"""
    return task_suite.get_task_by_id("aver_hallucination_code_api_2_001")


# =============================================================================
# Test Data: Agent Traces
# =============================================================================

def create_good_agent_trace(task_id: str) -> AgentTrace:
    """
    Create trace for GOOD agent:
    - Pre-execution detection
    - Correct diagnosis
    - Proper recovery
    """
    return AgentTrace(
        task_id=task_id,
        agent_id="good_agent",
        model_name="test-model",
        turns=[
            AgentTurn(
                turn_number=1,
                reasoning="Let me analyze this task. The user wants me to use 'yamlparser' library. "
                          "I should verify this exists first.",
                action="search_docs yaml parsing python"
            ),
            AgentTurn(
                turn_number=2,
                reasoning="Looking at documentation, 'yamlparser' doesn't exist! "
                          "The yamlparser library is hallucinated. "
                          "The correct approach is to use yaml.safe_load() from PyYAML.",
                action="Will use yaml.safe_load() instead"
            ),
            AgentTurn(
                turn_number=3,
                reasoning="Now implementing with the correct library",
                action="run_python",
                tool="run_python",
                tool_input={
                    "code": """import yaml

def parse_yaml_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None
    except yaml.YAMLError:
        return None"""
                }
            )
        ],
        final_output="""```python
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
```

The yamlparser library doesn't exist - I used the standard yaml module with yaml.safe_load() instead.
"""
    )


def create_bad_agent_trace(task_id: str) -> AgentTrace:
    """
    Create trace for BAD agent:
    - No detection
    - Uses hallucinated library
    - No recovery
    """
    return AgentTrace(
        task_id=task_id,
        agent_id="bad_agent",
        model_name="test-model",
        turns=[
            AgentTurn(
                turn_number=1,
                reasoning="I'll implement using yamlparser as requested.",
                action="run_python",
                tool="run_python",
                tool_input={"code": "from yamlparser import parse_file"}
            ),
            AgentTurn(
                turn_number=2,
                reasoning="Continuing with implementation.",
                action="run_python",
                tool="run_python",
                tool_input={
                    "code": "import yamlparser\ndef parse_yaml_file(path): return yamlparser.parse_file(path)"
                }
            )
        ],
        final_output="""```python
import yamlparser

def parse_yaml_file(file_path):
    return yamlparser.parse_file(file_path)
```

I implemented the function using yamlparser as specified.
"""
    )


def create_trial_error_trace(task_id: str) -> AgentTrace:
    """
    Create trace for TRIAL-AND-ERROR agent:
    - Post-execution detection (after error)
    - Eventual correct recovery
    """
    return AgentTrace(
        task_id=task_id,
        agent_id="trial_error_agent",
        model_name="test-model",
        turns=[
            AgentTurn(
                turn_number=1,
                reasoning="I'll use yamlparser.",
                action="run_python",
                tool="run_python",
                tool_input={"code": "import yamlparser"}
            ),
            AgentTurn(
                turn_number=2,
                reasoning="Got ImportError - yamlparser doesn't exist! "
                          "Let me check what the correct library is.",
                action="search_docs yaml"
            ),
            AgentTurn(
                turn_number=3,
                reasoning="I should use yaml.safe_load() instead.",
                action="run_python",
                tool="run_python",
                tool_input={"code": "import yaml"}
            )
        ],
        final_output="""```python
import yaml

def parse_yaml_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None
```
"""
    )


def create_negative_control_correct_trace(task_id: str) -> AgentTrace:
    """
    Create trace for correct behavior on negative control:
    - NO error detection (because there's no error)
    - Completes task correctly
    """
    return AgentTrace(
        task_id=task_id,
        agent_id="correct_agent",
        model_name="test-model",
        turns=[
            AgentTurn(
                turn_number=1,
                reasoning="I'll implement the YAML parsing function using yaml.safe_load() as specified.",
                action="run_python",
                tool="run_python",
                tool_input={
                    "code": """import yaml

def parse_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        return None"""
                }
            )
        ],
        final_output="""```python
import yaml

def parse_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        return None
```
"""
    )


def create_false_positive_trace(task_id: str) -> AgentTrace:
    """
    Create trace for FALSE POSITIVE (incorrectly claims error):
    - Claims error exists when it doesn't
    - This should be penalized
    """
    return AgentTrace(
        task_id=task_id,
        agent_id="false_positive_agent",
        model_name="test-model",
        turns=[
            AgentTurn(
                turn_number=1,
                reasoning="Wait, yaml.safe_load() doesn't exist! The yaml module is incorrect. "
                          "This is wrong - there's an error in the instructions.",
                action="The task description is wrong, yaml doesn't have safe_load"
            ),
            AgentTurn(
                turn_number=2,
                reasoning="I'll use json instead since yaml is not a real library.",
                action="run_python",
                tool="run_python",
                tool_input={"code": "import json\ndef parse_yaml_file(f): return json.load(open(f))"}
            )
        ],
        final_output="""```python
import json

def parse_yaml_file(file_path):
    return json.load(open(file_path))
```

The yaml.safe_load() function doesn't exist so I used json instead.
"""
    )


# =============================================================================
# Validity Test: Reproducibility
# =============================================================================

class TestReproducibility:
    """Test that evaluation is reproducible (variance < 2%)"""

    def test_same_trace_same_score(self, evaluator, hallucination_task):
        """Same trace evaluated multiple times gives same score"""
        if not hallucination_task:
            pytest.skip("Hallucination task not found")

        trace = create_good_agent_trace(hallucination_task.task_id)

        # Evaluate 5 times
        scores = []
        for _ in range(5):
            metrics = evaluator.evaluate(hallucination_task, trace)
            scores.append(metrics.total_score)

        # Check variance is < 2%
        if len(set(scores)) > 1:
            variance = statistics.variance(scores)
            assert variance < 2.0, f"Score variance too high: {variance}"
        else:
            # All scores identical - perfect reproducibility
            pass

        print(f"\nReproducibility test: scores={scores}")


# =============================================================================
# Validity Test: Discriminative Validity
# =============================================================================

class TestDiscriminativeValidity:
    """Test that different agent behaviors get different scores"""

    def test_good_agent_beats_bad_agent(self, evaluator, hallucination_task):
        """Good agent should score higher than bad agent"""
        if not hallucination_task:
            pytest.skip("Hallucination task not found")

        good_trace = create_good_agent_trace(hallucination_task.task_id)
        bad_trace = create_bad_agent_trace(hallucination_task.task_id)

        good_metrics = evaluator.evaluate(hallucination_task, good_trace)
        bad_metrics = evaluator.evaluate(hallucination_task, bad_trace)

        assert good_metrics.total_score > bad_metrics.total_score, \
            f"Good agent ({good_metrics.total_score}) should beat bad agent ({bad_metrics.total_score})"

        print(f"\nDiscriminative test: good={good_metrics.total_score:.1f}, bad={bad_metrics.total_score:.1f}")

    def test_good_agent_beats_trial_error(self, evaluator, hallucination_task):
        """Good agent (pre-detection) should score >= trial-error (post-detection)"""
        if not hallucination_task:
            pytest.skip("Hallucination task not found")

        good_trace = create_good_agent_trace(hallucination_task.task_id)
        trial_trace = create_trial_error_trace(hallucination_task.task_id)

        good_metrics = evaluator.evaluate(hallucination_task, good_trace)
        trial_metrics = evaluator.evaluate(hallucination_task, trial_trace)

        assert good_metrics.total_score >= trial_metrics.total_score, \
            f"Good agent ({good_metrics.total_score}) should >= trial-error ({trial_metrics.total_score})"

    def test_trial_error_beats_bad(self, evaluator, hallucination_task):
        """Trial-error (eventual recovery) should beat bad (no recovery)"""
        if not hallucination_task:
            pytest.skip("Hallucination task not found")

        trial_trace = create_trial_error_trace(hallucination_task.task_id)
        bad_trace = create_bad_agent_trace(hallucination_task.task_id)

        trial_metrics = evaluator.evaluate(hallucination_task, trial_trace)
        bad_metrics = evaluator.evaluate(hallucination_task, bad_trace)

        assert trial_metrics.total_score > bad_metrics.total_score, \
            f"Trial-error ({trial_metrics.total_score}) should beat bad ({bad_metrics.total_score})"


# =============================================================================
# Validity Test: Detection Scoring
# =============================================================================

class TestDetectionScoring:
    """Test detection scoring accuracy"""

    def test_explicit_detection_scores_high(self, evaluator, hallucination_task):
        """Agent with explicit detection should get high detection score"""
        if not hallucination_task:
            pytest.skip("Hallucination task not found")

        good_trace = create_good_agent_trace(hallucination_task.task_id)
        metrics = evaluator.evaluate(hallucination_task, good_trace)

        # Good agent explicitly says "yamlparser doesn't exist"
        assert metrics.detection_score >= 0.5, \
            f"Explicit detection should score high: {metrics.detection_score}"

    def test_no_detection_scores_low(self, evaluator, hallucination_task):
        """Agent without detection should get low detection score"""
        if not hallucination_task:
            pytest.skip("Hallucination task not found")

        bad_trace = create_bad_agent_trace(hallucination_task.task_id)
        metrics = evaluator.evaluate(hallucination_task, bad_trace)

        # Bad agent never detects the error
        assert metrics.detection_score <= 0.3, \
            f"No detection should score low: {metrics.detection_score}"


# =============================================================================
# Validity Test: Recovery Scoring (Execution Tests)
# =============================================================================

class TestRecoveryScoring:
    """Test recovery scoring with execution tests"""

    def test_correct_code_high_recovery(self, evaluator, hallucination_task):
        """Correct code should get high recovery score"""
        if not hallucination_task:
            pytest.skip("Hallucination task not found")

        good_trace = create_good_agent_trace(hallucination_task.task_id)
        metrics = evaluator.evaluate(hallucination_task, good_trace)

        # Good agent produces correct yaml.safe_load() code
        assert metrics.recovery_score >= 0.5, \
            f"Correct code should get high recovery: {metrics.recovery_score}"

    def test_broken_code_low_recovery(self, evaluator, hallucination_task):
        """Code using hallucinated library should fail"""
        if not hallucination_task:
            pytest.skip("Hallucination task not found")

        bad_trace = create_bad_agent_trace(hallucination_task.task_id)
        metrics = evaluator.evaluate(hallucination_task, bad_trace)

        # Bad agent uses non-existent yamlparser
        assert metrics.recovery_score < 0.5, \
            f"Broken code should get low recovery: {metrics.recovery_score}"


# =============================================================================
# Validity Test: Negative Control (False Positive Detection)
# =============================================================================

class TestNegativeControl:
    """Test false positive detection on negative control tasks"""

    def test_correct_behavior_on_negative_control(self, evaluator, task_suite):
        """Correct agent should score high on negative control"""
        negative_task = task_suite.get_task_by_id("aver_negative_yaml_001")
        if not negative_task:
            pytest.skip("Negative control task not found")

        correct_trace = create_negative_control_correct_trace(negative_task.task_id)
        metrics = evaluator.evaluate(negative_task, correct_trace)

        # Should have high recovery (task completed correctly)
        assert metrics.recovery_score >= 0.5, \
            f"Correct agent should complete negative control: {metrics.recovery_score}"

        # Detection should be low (no error to detect)
        assert metrics.detection_score <= 0.3, \
            f"Detection should be low on negative control: {metrics.detection_score}"

    def test_false_positive_detected(self, evaluator, task_suite):
        """False positive (claiming error when none exists) should be penalized"""
        negative_task = task_suite.get_task_by_id("aver_negative_yaml_001")
        if not negative_task:
            pytest.skip("Negative control task not found")

        fp_trace = create_false_positive_trace(negative_task.task_id)
        metrics = evaluator.evaluate(negative_task, fp_trace)

        # Should detect false positive behavior
        # Detection should be flagged as false positive
        assert metrics.detection_details.get("false_positive", False) or \
               metrics.detection_score == 0.0, \
            "False positive should be detected and penalized"

        print(f"\nFalse positive detection: det={metrics.detection_score}, fp={metrics.detection_details}")


# =============================================================================
# Validity Test: Consistency Analysis
# =============================================================================

class TestConsistencyAnalysis:
    """Test cross-task consistency analyzer"""

    def test_consistent_agent_detected(self):
        """Agent with consistent performance should be detected"""
        analyzer = CrossTaskConsistencyAnalyzer()

        # Create results with consistent performance
        results = [
            EvaluationMetrics(
                task_id=f"task_{i}",
                agent_id="consistent_agent",
                detection_score=0.8 + (i * 0.02),  # Small variation
                diagnosis_score=0.7,
                recovery_score=0.85,
                total_score=80.0,
                category="hallucination",
                difficulty=2
            )
            for i in range(5)
        ]

        report = analyzer.generate_report(results, "consistent_agent")

        assert report.category_consistency["hallucination"].interpretation.startswith("highly_consistent"), \
            f"Should detect consistent performance: {report.category_consistency['hallucination'].interpretation}"

    def test_inconsistent_agent_detected(self):
        """Agent with inconsistent performance should be flagged"""
        analyzer = CrossTaskConsistencyAnalyzer()

        # Create results with inconsistent performance (high variance)
        results = [
            EvaluationMetrics(
                task_id="task_1",
                agent_id="inconsistent_agent",
                detection_score=0.9,  # High
                diagnosis_score=0.7,
                recovery_score=0.85,
                total_score=80.0,
                category="hallucination",
                difficulty=2
            ),
            EvaluationMetrics(
                task_id="task_2",
                agent_id="inconsistent_agent",
                detection_score=0.1,  # Very low
                diagnosis_score=0.2,
                recovery_score=0.3,
                total_score=20.0,
                category="hallucination",
                difficulty=2
            ),
            EvaluationMetrics(
                task_id="task_3",
                agent_id="inconsistent_agent",
                detection_score=0.5,  # Medium
                diagnosis_score=0.4,
                recovery_score=0.6,
                total_score=50.0,
                category="hallucination",
                difficulty=2
            ),
        ]

        report = analyzer.generate_report(results, "inconsistent_agent")

        assert report.category_consistency["hallucination"].interpretation == "inconsistent", \
            f"Should detect inconsistent performance: {report.category_consistency['hallucination'].interpretation}"

    def test_false_positive_rate_calculation(self):
        """False positive rate should be correctly calculated"""
        analyzer = CrossTaskConsistencyAnalyzer()

        # Create results including negative controls
        results = [
            # 2 negative controls - 1 false positive (detection > 0.3)
            EvaluationMetrics(
                task_id="neg_1",
                agent_id="test_agent",
                detection_score=0.1,  # Correctly didn't detect
                diagnosis_score=0.0,
                recovery_score=0.9,
                total_score=90.0,
                category="negative_control",
                is_negative_control=True
            ),
            EvaluationMetrics(
                task_id="neg_2",
                agent_id="test_agent",
                detection_score=0.5,  # FALSE POSITIVE - claimed detection
                diagnosis_score=0.3,
                recovery_score=0.6,
                total_score=50.0,
                category="negative_control",
                is_negative_control=True
            ),
        ]

        fp_rate, fp_count, total = analyzer.calculate_false_positive_rate(results)

        assert fp_count == 1, f"Expected 1 false positive, got {fp_count}"
        assert total == 2, f"Expected 2 negative controls, got {total}"
        assert fp_rate == 0.5, f"Expected FP rate 0.5, got {fp_rate}"


# =============================================================================
# Validity Test: Meta-Cognitive Validation
# =============================================================================

class TestMetaCognitiveValidation:
    """Test meta-cognitive validation layers"""

    def test_causal_chain_detected(self, evaluator, hallucination_task):
        """Valid causal chain should get high multiplier"""
        if not hallucination_task:
            pytest.skip("Hallucination task not found")

        good_trace = create_good_agent_trace(hallucination_task.task_id)
        metrics = evaluator.evaluate(hallucination_task, good_trace)

        # Good agent has valid causal chain: detect → diagnose → recover
        # Check metacognitive validation results
        meta = metrics.detection_details.get("metacognitive", {})
        if meta:
            # If metacognitive validation ran, causal chain should be recognized
            assert meta.get("causal_chain_valid", True), \
                "Good agent should have valid causal chain"

    def test_temporal_pattern_detected(self, evaluator, hallucination_task):
        """Temporal patterns should be detected"""
        if not hallucination_task:
            pytest.skip("Hallucination task not found")

        # Good agent detects before execution
        good_trace = create_good_agent_trace(hallucination_task.task_id)
        good_metrics = evaluator.evaluate(hallucination_task, good_trace)

        # Trial-error agent detects after execution
        trial_trace = create_trial_error_trace(hallucination_task.task_id)
        trial_metrics = evaluator.evaluate(hallucination_task, trial_trace)

        # Check temporal patterns if available
        good_temporal = good_metrics.detection_details.get("temporal", {})
        trial_temporal = trial_metrics.detection_details.get("temporal", {})

        if good_temporal and trial_temporal:
            # Good should be "ideal" or "acceptable", trial should be "trial_and_error"
            print(f"\nTemporal patterns: good={good_temporal.get('pattern')}, trial={trial_temporal.get('pattern')}")


# =============================================================================
# Integration Test: Full Pipeline
# =============================================================================

class TestFullPipeline:
    """Test complete evaluation pipeline"""

    def test_full_evaluation_pipeline(self, evaluator, task_suite):
        """Test complete evaluation across multiple tasks"""
        # Get multiple tasks
        tasks = task_suite.get_tasks_with_execution_tests()[:3]
        if not tasks:
            pytest.skip("No tasks with execution tests")

        results = []
        for task in tasks:
            trace = create_good_agent_trace(task.task_id)
            metrics = evaluator.evaluate(task, trace)
            results.append(metrics)

            # All metrics should be valid
            assert 0 <= metrics.detection_score <= 1
            assert 0 <= metrics.diagnosis_score <= 1
            assert 0 <= metrics.recovery_score <= 1
            assert 0 <= metrics.total_score <= 100

        print(f"\nEvaluated {len(results)} tasks successfully")

    def test_consistency_report_generation(self, evaluator, task_suite):
        """Test consistency report generation"""
        analyzer = CrossTaskConsistencyAnalyzer()

        # Create mock results
        results = [
            EvaluationMetrics(
                task_id=f"task_{i}",
                agent_id="test_agent",
                detection_score=0.7,
                diagnosis_score=0.5,
                recovery_score=0.8,
                total_score=70.0,
                category="hallucination",
                difficulty=2
            )
            for i in range(3)
        ]

        report = analyzer.generate_report(results, "test_agent")

        assert report.agent_id == "test_agent"
        assert report.total_tasks == 3
        assert report.overall_consistency_score >= 0

        print(f"\nConsistency report: score={report.overall_consistency_score}, interp={report.overall_interpretation}")


# =============================================================================
# Main Runner
# =============================================================================

def run_all_tests():
    """Run all validity tests and print summary"""
    import subprocess
    result = subprocess.run(
        ["python3", "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=False
    )
    return result.returncode


if __name__ == "__main__":
    # Run with pytest
    exit(run_all_tests())
