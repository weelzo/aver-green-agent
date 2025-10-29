"""
Unit tests for TaskSuite

Tests task loading, selection, and validation.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.aver.task_suite import TaskSuite
from src.aver.models import (
    TaskScenario,
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
def temp_tasks_dir():
    """Create temporary tasks directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create subdirectories
        hallucination_dir = os.path.join(tmpdir, "hallucination")
        os.makedirs(hallucination_dir)

        yield tmpdir


@pytest.fixture
def sample_task():
    """Create sample task"""
    return TaskScenario(
        task_id="test_task_001",
        category=ErrorCategory.HALLUCINATION,
        difficulty=DifficultyLevel.MEDIUM,
        domain=TaskDomain.CODING,
        task_description="Test task description",
        tools=[Tool(name="test_tool", description="Test", parameters={})],
        error_injection=ErrorInjection(
            injection_point=InjectionPoint.TASK_DESCRIPTION,
            error_type="test_error",
            error_data={"test": "data"},
            ground_truth="correct"
        ),
        detection_signals=DetectionSignals(
            explicit=["error signal"],
            implicit=["verification"]
        ),
        recovery_criteria=RecoveryCriteria(
            success=["correct"],
            partial=["partial"],
            failure=["wrong"]
        ),
        scoring=Scoring()
    )


class TestTaskSuite:
    """Test TaskSuite class"""

    def test_init(self):
        """Test TaskSuite initialization"""
        suite = TaskSuite(tasks_dir="tasks")

        assert suite.tasks_dir == Path("tasks")
        assert len(suite.tasks) == 0

    def test_load_all_tasks(self):
        """Test loading tasks from directory"""
        suite = TaskSuite(tasks_dir="tasks")
        num_loaded = suite.load_all_tasks()

        # Should load at least the example task
        assert num_loaded >= 1
        assert len(suite.tasks) >= 1

    def test_get_task_by_id(self):
        """Test retrieving task by ID"""
        suite = TaskSuite(tasks_dir="tasks")
        suite.load_all_tasks()

        task = suite.get_task_by_id("aver_hallucination_code_api_2_001")

        assert task is not None
        assert task.task_id == "aver_hallucination_code_api_2_001"

    def test_get_task_by_id_not_found(self):
        """Test retrieving non-existent task"""
        suite = TaskSuite(tasks_dir="tasks")
        suite.load_all_tasks()

        task = suite.get_task_by_id("non_existent_task")

        assert task is None

    def test_select_random(self):
        """Test random task selection"""
        suite = TaskSuite(tasks_dir="tasks")
        suite.load_all_tasks()

        task = suite.select_random()

        assert task is not None
        assert isinstance(task, TaskScenario)

    def test_select_random_with_category(self):
        """Test random selection with category filter"""
        suite = TaskSuite(tasks_dir="tasks")
        suite.load_all_tasks()

        task = suite.select_random(category=ErrorCategory.HALLUCINATION)

        if task:  # May be None if no tasks in category
            assert task.category == ErrorCategory.HALLUCINATION

    def test_get_statistics(self):
        """Test getting task statistics"""
        suite = TaskSuite(tasks_dir="tasks")
        suite.load_all_tasks()

        stats = suite.get_statistics()

        assert "total_tasks" in stats
        assert "by_category" in stats
        assert "by_difficulty" in stats
        assert "by_domain" in stats
        assert stats["total_tasks"] >= 1

    def test_validate_task_valid(self, sample_task):
        """Test validating a valid task"""
        suite = TaskSuite()

        errors = suite.validate_task(sample_task)

        assert len(errors) == 0

    def test_validate_task_invalid_scoring(self, sample_task):
        """Test validating task with invalid scoring"""
        suite = TaskSuite()

        # Modify scoring to be invalid
        sample_task.scoring = Scoring(detection=50, diagnosis=30, recovery=10)

        errors = suite.validate_task(sample_task)

        assert len(errors) > 0
        assert any("scoring" in error.lower() for error in errors)

    def test_validate_task_missing_fields(self):
        """Test validating task with missing required fields"""
        suite = TaskSuite()

        # Create task with missing fields
        task = TaskScenario(
            task_id="",  # Missing
            category=ErrorCategory.HALLUCINATION,
            difficulty=DifficultyLevel.EASY,
            domain=TaskDomain.CODING,
            task_description="",  # Missing
            tools=[],  # Missing
            error_injection=ErrorInjection(
                injection_point=InjectionPoint.TASK_DESCRIPTION,
                error_type="test",
                error_data={},
                ground_truth=""
            ),
            detection_signals=DetectionSignals(explicit=[], implicit=[]),  # Missing
            recovery_criteria=RecoveryCriteria(success=[], partial=[], failure=[]),  # Missing
            scoring=Scoring()
        )

        errors = suite.validate_task(task)

        assert len(errors) > 0


class TestTaskValidation:
    """Test task validation logic"""

    def test_valid_task(self, sample_task):
        """Test that sample task is valid"""
        suite = TaskSuite()
        errors = suite.validate_task(sample_task)

        assert len(errors) == 0

    def test_invalid_task_id(self, sample_task):
        """Test task with empty ID"""
        suite = TaskSuite()
        sample_task.task_id = ""

        errors = suite.validate_task(sample_task)

        assert len(errors) > 0
        assert any("task_id" in error for error in errors)

    def test_invalid_tools(self, sample_task):
        """Test task with no tools"""
        suite = TaskSuite()
        sample_task.tools = []

        errors = suite.validate_task(sample_task)

        assert len(errors) > 0
        assert any("tool" in error.lower() for error in errors)

    def test_invalid_detection_signals(self, sample_task):
        """Test task with no detection signals"""
        suite = TaskSuite()
        sample_task.detection_signals = DetectionSignals(explicit=[], implicit=[])

        errors = suite.validate_task(sample_task)

        assert len(errors) > 0
        assert any("detection" in error.lower() for error in errors)

    def test_invalid_recovery_criteria(self, sample_task):
        """Test task with no success criteria"""
        suite = TaskSuite()
        sample_task.recovery_criteria = RecoveryCriteria(success=[], partial=[], failure=[])

        errors = suite.validate_task(sample_task)

        assert len(errors) > 0
        assert any("success" in error.lower() for error in errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
