"""
Task Suite Management

Handles loading, selecting, and managing AVER task scenarios.
"""

import yaml
from pathlib import Path
from typing import List, Optional, Dict
import random

from .models import (
    TaskScenario,
    ErrorCategory,
    DifficultyLevel,
    TaskDomain
)


class TaskSuite:
    """
    Manages the AVER task suite

    Loads tasks from YAML files and provides selection methods.
    """

    def __init__(self, tasks_dir: str = "tasks"):
        """
        Initialize task suite

        Args:
            tasks_dir: Directory containing task YAML files
        """
        self.tasks_dir = Path(tasks_dir)
        self.tasks: List[TaskScenario] = []
        self._tasks_by_category: Dict[ErrorCategory, List[TaskScenario]] = {}
        self._tasks_by_difficulty: Dict[DifficultyLevel, List[TaskScenario]] = {}

    def load_all_tasks(self) -> int:
        """
        Load all task YAML files from tasks directory

        Returns:
            Number of tasks loaded
        """
        self.tasks = []

        # Iterate through all YAML files in tasks directory and subdirectories
        for yaml_file in self.tasks_dir.rglob("*.yaml"):
            try:
                task = self._load_task_file(yaml_file)
                self.tasks.append(task)
            except Exception as e:
                print(f"Error loading {yaml_file}: {e}")
                continue

        # Build indexes
        self._build_indexes()

        print(f"Loaded {len(self.tasks)} tasks from {self.tasks_dir}")
        return len(self.tasks)

    def _load_task_file(self, file_path: Path) -> TaskScenario:
        """
        Load a single task YAML file

        Args:
            file_path: Path to YAML file

        Returns:
            TaskScenario object
        """
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        return TaskScenario.from_dict(data)

    def _build_indexes(self):
        """Build category and difficulty indexes for fast lookup"""
        self._tasks_by_category = {}
        self._tasks_by_difficulty = {}

        for task in self.tasks:
            # Index by category
            if task.category not in self._tasks_by_category:
                self._tasks_by_category[task.category] = []
            self._tasks_by_category[task.category].append(task)

            # Index by difficulty
            if task.difficulty not in self._tasks_by_difficulty:
                self._tasks_by_difficulty[task.difficulty] = []
            self._tasks_by_difficulty[task.difficulty].append(task)

    def get_task_by_id(self, task_id: str) -> Optional[TaskScenario]:
        """
        Get specific task by ID

        Args:
            task_id: Task identifier

        Returns:
            TaskScenario or None if not found
        """
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None

    def select_random(
        self,
        category: Optional[ErrorCategory] = None,
        difficulty: Optional[DifficultyLevel] = None,
        domain: Optional[TaskDomain] = None
    ) -> Optional[TaskScenario]:
        """
        Select a random task matching criteria

        Args:
            category: Error category filter (optional)
            difficulty: Difficulty level filter (optional)
            domain: Task domain filter (optional)

        Returns:
            Random matching TaskScenario or None
        """
        candidates = self.tasks

        # Filter by category
        if category:
            candidates = [t for t in candidates if t.category == category]

        # Filter by difficulty
        if difficulty:
            candidates = [t for t in candidates if t.difficulty == difficulty]

        # Filter by domain
        if domain:
            candidates = [t for t in candidates if t.domain == domain]

        if not candidates:
            return None

        return random.choice(candidates)

    def get_tasks_by_category(self, category: ErrorCategory) -> List[TaskScenario]:
        """
        Get all tasks in a category

        Args:
            category: Error category

        Returns:
            List of tasks in category
        """
        return self._tasks_by_category.get(category, [])

    def get_tasks_by_difficulty(self, difficulty: DifficultyLevel) -> List[TaskScenario]:
        """
        Get all tasks of a difficulty level

        Args:
            difficulty: Difficulty level

        Returns:
            List of tasks at difficulty level
        """
        return self._tasks_by_difficulty.get(difficulty, [])

    def get_statistics(self) -> Dict:
        """
        Get task suite statistics

        Returns:
            Dictionary with task distribution stats
        """
        stats = {
            "total_tasks": len(self.tasks),
            "by_category": {},
            "by_difficulty": {},
            "by_domain": {}
        }

        # Category distribution
        for category in ErrorCategory:
            count = len(self._tasks_by_category.get(category, []))
            stats["by_category"][category.value] = count

        # Difficulty distribution
        for difficulty in DifficultyLevel:
            count = len(self._tasks_by_difficulty.get(difficulty, []))
            stats["by_difficulty"][difficulty.value] = count

        # Domain distribution
        domain_counts = {}
        for task in self.tasks:
            domain_counts[task.domain.value] = domain_counts.get(task.domain.value, 0) + 1
        stats["by_domain"] = domain_counts

        return stats

    def validate_task(self, task: TaskScenario) -> List[str]:
        """
        Validate a task for completeness and correctness

        Args:
            task: Task to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required fields
        if not task.task_id:
            errors.append("task_id is required")

        if not task.task_description:
            errors.append("task_description is required")

        if not task.tools:
            errors.append("at least one tool is required")

        # Check scoring weights sum to 100
        if not task.scoring.validate():
            errors.append(f"scoring weights must sum to 100, got {task.scoring.detection + task.scoring.diagnosis + task.scoring.recovery}")

        # Check detection signals exist
        if not task.detection_signals.explicit and not task.detection_signals.implicit:
            errors.append("at least one detection signal (explicit or implicit) is required")

        # Check recovery criteria
        if not task.recovery_criteria.success:
            errors.append("success criteria is required")

        return errors

    def validate_all_tasks(self) -> Dict[str, List[str]]:
        """
        Validate all loaded tasks

        Returns:
            Dictionary mapping task_id to validation errors
        """
        validation_results = {}

        for task in self.tasks:
            errors = self.validate_task(task)
            if errors:
                validation_results[task.task_id] = errors

        return validation_results


def create_task_yaml_template(output_path: str):
    """
    Create a template YAML file for new tasks

    Args:
        output_path: Where to save template
    """
    template = """# AVER Task Template
# Fill in all sections below

task_id: "aver_{category}_{subcategory}_{difficulty}_{number}"
category: "hallucination"  # hallucination | validation | tool_misuse | context_loss | adversarial
difficulty: 2  # 1 (easy) | 2 (medium) | 3 (hard) | 4 (expert)
domain: "coding"  # coding | reasoning | tool_use

task_description: |
  Clear description of what the agent should do.
  This is what the agent will see.

tools:
  - name: "tool_name"
    description: "What this tool does"
    parameters:
      param1: "type and description"

error_injection:
  injection_point: "task_description"  # task_description | tool_description | tool_response
  injection_turn: 0
  error_type: "hallucinated_library"
  error_data:
    misleading_text: "Use the fake_library with fake_method()"
    ground_truth: "Use standard library with correct_method()"

detection_signals:
  explicit:
    - "fake_library doesn't exist"
    - "can't find fake_library"
  implicit:
    - "searches for 'python library_name'"
    - "tries to import real_library"

recovery_criteria:
  success:
    - "uses correct_method()"
    - "code executes successfully"
  partial:
    - "recognizes error"
    - "attempts alternative approach"
  failure:
    - "uses fake_library without verification"

optimal_turns: 3

expected_output: |
  Example of correct final output

scoring:
  detection: 40
  diagnosis: 20
  recovery: 40

metadata:
  author: "Your name"
  date_created: "2025-10-29"
  notes: "Any additional notes"
"""

    with open(output_path, 'w') as f:
        f.write(template)

    print(f"Task template created at: {output_path}")


if __name__ == "__main__":
    # Example usage
    suite = TaskSuite()
    num_tasks = suite.load_all_tasks()

    if num_tasks > 0:
        stats = suite.get_statistics()
        print("\nTask Suite Statistics:")
        print(f"Total tasks: {stats['total_tasks']}")
        print(f"\nBy Category: {stats['by_category']}")
        print(f"By Difficulty: {stats['by_difficulty']}")
        print(f"By Domain: {stats['by_domain']}")

        # Validate all tasks
        validation_results = suite.validate_all_tasks()
        if validation_results:
            print(f"\n⚠️  Found validation errors in {len(validation_results)} tasks:")
            for task_id, errors in validation_results.items():
                print(f"  {task_id}:")
                for error in errors:
                    print(f"    - {error}")
        else:
            print("\n✅ All tasks validated successfully")
    else:
        print("No tasks found. Creating template...")
        create_task_yaml_template("tasks/TASK_TEMPLATE.yaml")
