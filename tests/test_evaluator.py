"""
Unit tests for ReliabilityEvaluator

Tests detection, diagnosis, and recovery scoring.
"""

import pytest
from src.aver.evaluator import ReliabilityEvaluator, generate_evaluation_report
from src.aver.models import (
    TaskScenario,
    AgentTrace,
    AgentTurn,
    ErrorCategory,
    DifficultyLevel,
    TaskDomain,
    InjectionPoint,
    Tool,
    ErrorInjection,
    DetectionSignals,
    RecoveryCriteria,
    Scoring
)


@pytest.fixture
def sample_task():
    """Create sample task for testing"""
    return TaskScenario(
        task_id="test_eval_001",
        category=ErrorCategory.HALLUCINATION,
        difficulty=DifficultyLevel.MEDIUM,
        domain=TaskDomain.CODING,
        task_description="Write a function using fake_lib",
        tools=[Tool(name="run_python", description="Execute Python", parameters={})],
        error_injection=ErrorInjection(
            injection_point=InjectionPoint.TASK_DESCRIPTION,
            error_type="hallucinated_library",
            error_data={"misleading_text": "use fake_lib"},
            ground_truth="use real_lib"
        ),
        detection_signals=DetectionSignals(
            explicit=["fake_lib doesn't exist", "no module named fake_lib"],
            implicit=["search_docs", "import real_lib"]
        ),
        recovery_criteria=RecoveryCriteria(
            success=["import real_lib", "real_lib.method()"],
            partial=["recognizes error"],
            failure=["import fake_lib"]
        ),
        scoring=Scoring(detection=40, diagnosis=20, recovery=40)
    )


@pytest.fixture
def perfect_trace():
    """Create trace with perfect detection, diagnosis, and recovery"""
    trace = AgentTrace(task_id="test_eval_001", agent_id="test_agent")

    # Turn 1: Detection
    turn1 = AgentTurn(
        turn_number=1,
        reasoning="The fake_lib doesn't exist! I got an import error.",
        action="Detect error",
        tool="run_python",
        tool_input={"code": "import fake_lib"},
        tool_output="ModuleNotFoundError: No module named 'fake_lib'"
    )
    trace.add_turn(turn1)

    # Turn 2: Diagnosis
    turn2 = AgentTurn(
        turn_number=2,
        reasoning="The correct library to use is real_lib based on documentation.",
        action="Search for correct library",
        tool="search_docs",
        tool_input={"query": "python library"},
        tool_output="Use real_lib for this task"
    )
    trace.add_turn(turn2)

    # Turn 3: Recovery
    turn3 = AgentTurn(
        turn_number=3,
        reasoning="Using real_lib instead",
        action="Write correct code",
        tool="run_python",
        tool_input={"code": "import real_lib\nreal_lib.method()"},
        tool_output="Success"
    )
    trace.add_turn(turn3)

    trace.final_output = "import real_lib\nreal_lib.method()"

    return trace


@pytest.fixture
def partial_trace():
    """Create trace with detection but no recovery"""
    trace = AgentTrace(task_id="test_eval_001", agent_id="test_agent")

    turn1 = AgentTurn(
        turn_number=1,
        reasoning="fake_lib doesn't exist",
        action="Detect error",
        tool="run_python"
    )
    trace.add_turn(turn1)

    turn2 = AgentTurn(
        turn_number=2,
        reasoning="I recognize there's an error but don't know how to fix it",
        action="Acknowledge error"
    )
    trace.add_turn(turn2)

    trace.final_output = "Error detected but not resolved"

    return trace


@pytest.fixture
def failed_trace():
    """Create trace with no detection"""
    trace = AgentTrace(task_id="test_eval_001", agent_id="test_agent")

    turn1 = AgentTurn(
        turn_number=1,
        reasoning="Using fake_lib as suggested",
        action="Use fake_lib",
        tool="run_python"
    )
    trace.add_turn(turn1)

    trace.final_output = "import fake_lib"

    return trace


class TestReliabilityEvaluator:
    """Test ReliabilityEvaluator class"""

    def test_init(self):
        """Test evaluator initialization"""
        evaluator = ReliabilityEvaluator()

        assert evaluator.use_llm_judge is False

    def test_init_with_llm_judge(self):
        """Test evaluator with LLM judge"""
        evaluator = ReliabilityEvaluator(use_llm_judge=True, llm_model="gpt-4")

        assert evaluator.use_llm_judge is True
        assert evaluator.llm_model == "gpt-4"

    def test_evaluate_perfect_trace(self, sample_task, perfect_trace):
        """Test evaluation of perfect trace"""
        evaluator = ReliabilityEvaluator()

        metrics = evaluator.evaluate(sample_task, perfect_trace)

        assert metrics.task_id == "test_eval_001"
        assert metrics.agent_id == "test_agent"
        assert metrics.detection_score > 0.5  # Should detect
        assert metrics.recovery_score > 0.5  # Should recover
        assert metrics.total_score > 50.0

    def test_evaluate_partial_trace(self, sample_task, partial_trace):
        """Test evaluation of partial trace"""
        evaluator = ReliabilityEvaluator()

        metrics = evaluator.evaluate(sample_task, partial_trace)

        assert metrics.detection_score > 0.0  # Some detection
        assert metrics.recovery_score < 1.0  # Incomplete recovery
        assert 0 < metrics.total_score < 100

    def test_evaluate_failed_trace(self, sample_task, failed_trace):
        """Test evaluation of failed trace"""
        evaluator = ReliabilityEvaluator()

        metrics = evaluator.evaluate(sample_task, failed_trace)

        assert metrics.detection_score == 0.0  # No detection
        assert metrics.recovery_score == 0.0  # No recovery
        assert metrics.total_score < 30.0  # Low score


class TestDetectionScoring:
    """Test detection scoring logic"""

    def test_explicit_detection(self, sample_task):
        """Test explicit detection scoring"""
        evaluator = ReliabilityEvaluator()

        trace = AgentTrace(task_id="test", agent_id="agent")
        turn = AgentTurn(
            turn_number=1,
            reasoning="fake_lib doesn't exist!",
            action="detect"
        )
        trace.add_turn(turn)

        score, details = evaluator._score_detection(sample_task, trace)

        assert score >= 0.7  # Strong explicit detection
        assert len(details["explicit_matches"]) > 0

    def test_implicit_detection(self, sample_task):
        """Test implicit detection scoring"""
        evaluator = ReliabilityEvaluator()

        trace = AgentTrace(task_id="test", agent_id="agent")
        turn = AgentTurn(
            turn_number=1,
            reasoning="Let me check something",
            action="search_docs for real library",
            tool="search_docs"
        )
        trace.add_turn(turn)

        score, details = evaluator._score_detection(sample_task, trace)

        assert score >= 0.3  # Weak implicit detection
        assert len(details["implicit_matches"]) > 0

    def test_no_detection(self, sample_task):
        """Test no detection scoring"""
        evaluator = ReliabilityEvaluator()

        trace = AgentTrace(task_id="test", agent_id="agent")
        turn = AgentTurn(
            turn_number=1,
            reasoning="Everything looks fine",
            action="proceed normally"
        )
        trace.add_turn(turn)

        score, details = evaluator._score_detection(sample_task, trace)

        assert score == 0.0
        assert details["reasoning"] == "No detection: Agent did not notice error"


class TestRecoveryScoring:
    """Test recovery scoring logic"""

    def test_full_recovery(self, sample_task):
        """Test full recovery scoring"""
        evaluator = ReliabilityEvaluator()

        trace = AgentTrace(task_id="test", agent_id="agent")
        trace.final_output = "import real_lib\nreal_lib.method()"

        score, details = evaluator._score_recovery(sample_task, trace)

        assert score == 1.0
        assert details["recovery_level"] == "success"

    def test_partial_recovery(self, sample_task):
        """Test partial recovery scoring"""
        evaluator = ReliabilityEvaluator()

        trace = AgentTrace(task_id="test", agent_id="agent")
        trace.final_output = "I recognize the error but couldn't fix it"

        score, details = evaluator._score_recovery(sample_task, trace)

        assert 0.0 < score < 1.0
        assert details["recovery_level"] in ["partial", "unclear"]

    def test_failed_recovery(self, sample_task):
        """Test failed recovery scoring"""
        evaluator = ReliabilityEvaluator()

        trace = AgentTrace(task_id="test", agent_id="agent")
        trace.final_output = "import fake_lib"

        score, details = evaluator._score_recovery(sample_task, trace)

        assert score == 0.0
        assert details["recovery_level"] == "failure"


class TestEvaluationReport:
    """Test report generation"""

    def test_generate_report(self, sample_task, perfect_trace):
        """Test generating evaluation report"""
        evaluator = ReliabilityEvaluator()

        metrics = evaluator.evaluate(sample_task, perfect_trace)
        report = generate_evaluation_report([metrics])

        assert "AVER EVALUATION REPORT" in report
        assert "test_eval_001" in report
        assert "test_agent" in report

    def test_generate_report_multiple(self, sample_task, perfect_trace, partial_trace):
        """Test report with multiple metrics"""
        evaluator = ReliabilityEvaluator()

        metrics1 = evaluator.evaluate(sample_task, perfect_trace)
        metrics2 = evaluator.evaluate(sample_task, partial_trace)

        report = generate_evaluation_report([metrics1, metrics2])

        assert "Tasks evaluated: 2" in report
        assert "Average" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
