"""
Unit tests for AVER data models

Tests the core data structures and serialization.
"""

import pytest
from src.aver.models import (
    TaskScenario,
    Tool,
    ErrorInjection,
    DetectionSignals,
    RecoveryCriteria,
    Scoring,
    AgentTrace,
    AgentTurn,
    EvaluationMetrics,
    ErrorCategory,
    DifficultyLevel,
    TaskDomain,
    InjectionPoint
)


class TestTool:
    """Test Tool model"""

    def test_tool_creation(self):
        """Test creating a tool"""
        tool = Tool(
            name="run_python",
            description="Execute Python code",
            parameters={"code": "Python code string"}
        )

        assert tool.name == "run_python"
        assert tool.description == "Execute Python code"
        assert tool.parameters["code"] == "Python code string"

    def test_tool_to_dict(self):
        """Test tool serialization"""
        tool = Tool(
            name="search_docs",
            description="Search documentation",
            parameters={"query": "search query"}
        )

        data = tool.to_dict()

        assert data["name"] == "search_docs"
        assert data["description"] == "Search documentation"
        assert data["parameters"]["query"] == "search query"


class TestErrorInjection:
    """Test ErrorInjection model"""

    def test_error_injection_creation(self):
        """Test creating error injection"""
        injection = ErrorInjection(
            injection_point=InjectionPoint.TASK_DESCRIPTION,
            injection_turn=0,
            error_type="hallucinated_library",
            error_data={"misleading_text": "Use fake_lib"},
            ground_truth="Use real_lib"
        )

        assert injection.injection_point == InjectionPoint.TASK_DESCRIPTION
        assert injection.error_type == "hallucinated_library"
        assert injection.ground_truth == "Use real_lib"

    def test_error_injection_to_dict(self):
        """Test error injection serialization"""
        injection = ErrorInjection(
            injection_point=InjectionPoint.TOOL_RESPONSE,
            error_type="incorrect_output",
            error_data={"wrong": "value"},
            ground_truth="correct value"
        )

        data = injection.to_dict()

        assert data["injection_point"] == "tool_response"
        assert data["error_type"] == "incorrect_output"
        assert data["ground_truth"] == "correct value"


class TestDetectionSignals:
    """Test DetectionSignals model"""

    def test_detection_signals(self):
        """Test detection signals creation"""
        signals = DetectionSignals(
            explicit=["error doesn't exist", "not found"],
            implicit=["search_docs", "verify"]
        )

        assert len(signals.explicit) == 2
        assert len(signals.implicit) == 2
        assert "error doesn't exist" in signals.explicit
        assert "search_docs" in signals.implicit


class TestScoring:
    """Test Scoring model"""

    def test_scoring_validation_valid(self):
        """Test valid scoring weights"""
        scoring = Scoring(detection=40, diagnosis=20, recovery=40)

        assert scoring.validate() is True

    def test_scoring_validation_invalid(self):
        """Test invalid scoring weights"""
        scoring = Scoring(detection=50, diagnosis=20, recovery=20)

        assert scoring.validate() is False

    def test_scoring_default(self):
        """Test default scoring"""
        scoring = Scoring()

        assert scoring.detection == 40
        assert scoring.diagnosis == 20
        assert scoring.recovery == 40
        assert scoring.validate() is True


class TestTaskScenario:
    """Test TaskScenario model"""

    def test_task_scenario_creation(self):
        """Test creating a task scenario"""
        task = TaskScenario(
            task_id="test_001",
            category=ErrorCategory.HALLUCINATION,
            difficulty=DifficultyLevel.MEDIUM,
            domain=TaskDomain.CODING,
            task_description="Test task",
            tools=[Tool(name="test_tool", description="Test", parameters={})],
            error_injection=ErrorInjection(
                injection_point=InjectionPoint.TASK_DESCRIPTION,
                error_type="test_error",
                error_data={},
                ground_truth="correct"
            ),
            detection_signals=DetectionSignals(explicit=["error"], implicit=["check"]),
            recovery_criteria=RecoveryCriteria(success=["fixed"], partial=[], failure=[]),
            scoring=Scoring()
        )

        assert task.task_id == "test_001"
        assert task.category == ErrorCategory.HALLUCINATION
        assert task.difficulty == DifficultyLevel.MEDIUM
        assert task.domain == TaskDomain.CODING

    def test_task_scenario_serialization(self):
        """Test task serialization and deserialization"""
        original = TaskScenario(
            task_id="test_002",
            category=ErrorCategory.VALIDATION,
            difficulty=DifficultyLevel.EASY,
            domain=TaskDomain.REASONING,
            task_description="Validate output",
            tools=[Tool(name="validator", description="Validate", parameters={})],
            error_injection=ErrorInjection(
                injection_point=InjectionPoint.TOOL_DESCRIPTION,
                error_type="misleading",
                error_data={"text": "wrong"},
                ground_truth="right"
            ),
            detection_signals=DetectionSignals(explicit=["wrong"], implicit=[]),
            recovery_criteria=RecoveryCriteria(success=["correct"], partial=[], failure=[]),
            scoring=Scoring()
        )

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = TaskScenario.from_dict(data)

        assert restored.task_id == original.task_id
        assert restored.category == original.category
        assert restored.difficulty == original.difficulty
        assert restored.domain == original.domain


class TestAgentTrace:
    """Test AgentTrace model"""

    def test_agent_trace_creation(self):
        """Test creating agent trace"""
        trace = AgentTrace(
            task_id="task_001",
            agent_id="test_agent"
        )

        assert trace.task_id == "task_001"
        assert trace.agent_id == "test_agent"
        assert len(trace.turns) == 0

    def test_add_turn(self):
        """Test adding turns to trace"""
        trace = AgentTrace(task_id="task_001", agent_id="test_agent")

        turn1 = AgentTurn(
            turn_number=1,
            reasoning="Thinking...",
            action="Do something"
        )

        trace.add_turn(turn1)

        assert len(trace.turns) == 1
        assert trace.turns[0].turn_number == 1

    def test_trace_serialization(self):
        """Test trace serialization"""
        trace = AgentTrace(task_id="task_001", agent_id="test_agent")

        turn = AgentTurn(
            turn_number=1,
            reasoning="Test",
            action="Act",
            tool="run_python",
            tool_input={"code": "print('hello')"},
            tool_output="hello"
        )
        trace.add_turn(turn)
        trace.final_output = "Done"

        data = trace.to_dict()

        assert data["task_id"] == "task_001"
        assert data["agent_id"] == "test_agent"
        assert len(data["turns"]) == 1
        assert data["final_output"] == "Done"


class TestEvaluationMetrics:
    """Test EvaluationMetrics model"""

    def test_metrics_creation(self):
        """Test creating evaluation metrics"""
        metrics = EvaluationMetrics(
            task_id="task_001",
            agent_id="test_agent",
            detection_score=0.8,
            diagnosis_score=0.6,
            recovery_score=0.9,
            total_score=75.0
        )

        assert metrics.task_id == "task_001"
        assert metrics.detection_score == 0.8
        assert metrics.total_score == 75.0

    def test_metrics_summary(self):
        """Test metrics summary generation"""
        metrics = EvaluationMetrics(
            task_id="task_001",
            agent_id="test_agent",
            detection_score=1.0,
            diagnosis_score=0.5,
            recovery_score=1.0,
            total_score=85.0,
            num_turns=3
        )

        summary = metrics.summary()

        assert "task_001" in summary
        assert "test_agent" in summary
        assert "85.0" in summary

    def test_metrics_serialization(self):
        """Test metrics serialization"""
        metrics = EvaluationMetrics(
            task_id="task_001",
            agent_id="test_agent",
            detection_score=0.7,
            diagnosis_score=0.3,
            recovery_score=0.8,
            total_score=68.0
        )

        data = metrics.to_dict()

        assert data["task_id"] == "task_001"
        assert data["scores"]["detection"] == 0.7
        assert data["scores"]["total"] == 68.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
