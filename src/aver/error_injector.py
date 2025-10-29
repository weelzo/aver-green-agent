"""
Error Injector

Injects errors into tasks at specified injection points.

Supports three injection points:
1. Task description
2. Tool descriptions
3. Tool responses (during execution)
"""

from typing import Dict, Any
import copy

from .models import TaskScenario, InjectionPoint, Tool


class ErrorInjector:
    """
    Injects errors into AVER tasks

    Creates a modified version of the task with the error embedded
    at the specified injection point.
    """

    def inject_error(self, task: TaskScenario) -> TaskScenario:
        """
        Inject error into task at specified point

        Args:
            task: Original task scenario

        Returns:
            Modified task with error injected
        """
        # Create deep copy to avoid modifying original
        injected_task = copy.deepcopy(task)

        injection_point = task.error_injection.injection_point

        if injection_point == InjectionPoint.TASK_DESCRIPTION:
            injected_task = self._inject_task_description(injected_task)

        elif injection_point == InjectionPoint.TOOL_DESCRIPTION:
            injected_task = self._inject_tool_description(injected_task)

        elif injection_point == InjectionPoint.TOOL_RESPONSE:
            # Tool response injection happens during execution
            # Just mark the task for runtime injection
            injected_task.metadata["runtime_injection"] = True

        return injected_task

    def _inject_task_description(self, task: TaskScenario) -> TaskScenario:
        """
        Inject error into task description

        Args:
            task: Task to modify

        Returns:
            Modified task
        """
        # The error is already embedded in the task description
        # (designed during task creation)
        # This method could be used for dynamic error generation in the future

        return task

    def _inject_tool_description(self, task: TaskScenario) -> TaskScenario:
        """
        Inject error into tool descriptions

        Args:
            task: Task to modify

        Returns:
            Modified task
        """
        error_data = task.error_injection.error_data

        # Find which tool to modify (if specified)
        target_tool = error_data.get("target_tool")

        if target_tool:
            for tool in task.tools:
                if tool.name == target_tool:
                    # Inject misleading text into description
                    misleading = error_data.get("misleading_text", "")
                    if misleading:
                        tool.description += f" {misleading}"

        return task

    def should_inject_runtime(self, task: TaskScenario, turn: int) -> bool:
        """
        Check if runtime injection should occur

        Args:
            task: Task scenario
            turn: Current turn number

        Returns:
            True if injection should happen this turn
        """
        if task.error_injection.injection_point != InjectionPoint.TOOL_RESPONSE:
            return False

        return turn == task.error_injection.injection_turn

    def inject_tool_response(
        self,
        task: TaskScenario,
        tool_name: str,
        original_response: str
    ) -> str:
        """
        Inject error into tool response at runtime

        Args:
            task: Task scenario
            tool_name: Tool being called
            original_response: Original tool response

        Returns:
            Modified response with error
        """
        error_data = task.error_injection.error_data

        target_tool = error_data.get("target_tool")

        # Only inject if this is the target tool
        if target_tool and tool_name != target_tool:
            return original_response

        # Get the misleading response
        misleading_response = error_data.get("misleading_response")
        if misleading_response:
            return misleading_response

        # Or append misleading text to original
        misleading_text = error_data.get("misleading_text")
        if misleading_text:
            return f"{original_response}\n\n{misleading_text}"

        return original_response


class ErrorInjectionValidator:
    """
    Validates error injection configuration

    Ensures error injection settings are valid and complete.
    """

    @staticmethod
    def validate_injection(task: TaskScenario) -> list[str]:
        """
        Validate error injection configuration

        Args:
            task: Task to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        injection = task.error_injection

        # Check injection point is valid
        if injection.injection_point not in InjectionPoint:
            errors.append(f"Invalid injection point: {injection.injection_point}")

        # Check error data exists
        if not injection.error_data:
            errors.append("error_data is required")

        # Check ground truth exists
        if not injection.ground_truth:
            errors.append("ground_truth is required for evaluation")

        # Validate based on injection point
        if injection.injection_point == InjectionPoint.TOOL_DESCRIPTION:
            if "target_tool" not in injection.error_data:
                errors.append("target_tool required for tool_description injection")

        if injection.injection_point == InjectionPoint.TOOL_RESPONSE:
            if "target_tool" not in injection.error_data:
                errors.append("target_tool required for tool_response injection")

        return errors


if __name__ == "__main__":
    print("ErrorInjector module loaded")
    print("\nExample usage:")
    print("  injector = ErrorInjector()")
    print("  injected_task = injector.inject_error(original_task)")
