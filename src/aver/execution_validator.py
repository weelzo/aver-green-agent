"""
Execution Validator for AVER Benchmark

Validates agent recovery through actual code execution and test suites.
This provides deterministic, high-confidence validation of recovery success.

Key Features:
- Extracts code from agent output (markdown blocks, raw code)
- Runs test suites with setup/test/teardown phases
- Calculates weighted scores based on test results
- Supports both positive tests (should pass) and negative tests (should fail)
- Handles fallback gracefully when execution fails

This is the EXECUTION VALIDITY pillar of AVER's two-pillar validation system.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .sandbox import CodeSandbox
from .models import (
    TaskScenario,
    ExecutionEnvironment,
    TestCase,
    ExecutionResult
)


# =============================================================================
# CODE EXTRACTION
# =============================================================================

class CodeExtractor:
    """
    Extracts executable code from agent output.

    Handles various formats:
    - Markdown code blocks (```python ... ```)
    - JSON tool calls (<json>{"tool": "run_python", "parameters": {"code": ...}}</json>)
    - Indented code blocks
    - Raw Python code
    - Multiple code blocks (concatenated)
    """

    # Patterns for code extraction
    MARKDOWN_PYTHON_BLOCK = re.compile(
        r'```(?:python|py)?\s*\n(.*?)```',
        re.DOTALL | re.IGNORECASE
    )

    GENERIC_CODE_BLOCK = re.compile(
        r'```\s*\n(.*?)```',
        re.DOTALL
    )

    # Pattern for JSON tool calls (used by purple agents)
    JSON_TOOL_BLOCK = re.compile(
        r'<json>\s*(.*?)\s*</json>',
        re.DOTALL | re.IGNORECASE
    )

    def extract(self, output: str) -> Optional[str]:
        """
        Extract Python code from agent output.

        Args:
            output: Agent's final output

        Returns:
            Extracted code or None if no code found
        """
        if not output:
            return None

        # Try JSON tool calls first (purple agent format)
        json_code = self._extract_from_json_tool(output)
        if json_code:
            return json_code

        # Try markdown Python blocks
        python_blocks = self.MARKDOWN_PYTHON_BLOCK.findall(output)
        if python_blocks:
            return self._combine_blocks(python_blocks)

        # Try generic code blocks
        generic_blocks = self.GENERIC_CODE_BLOCK.findall(output)
        if generic_blocks:
            # Filter to likely Python code
            python_like = [b for b in generic_blocks if self._is_python_like(b)]
            if python_like:
                return self._combine_blocks(python_like)

        # Try to extract raw Python code
        raw_code = self._extract_raw_python(output)
        if raw_code:
            return raw_code

        return None

    def _extract_from_json_tool(self, output: str) -> Optional[str]:
        """Extract code from JSON tool call format"""
        import json
        import re

        matches = self.JSON_TOOL_BLOCK.findall(output)
        code_blocks = []

        for match in matches:
            try:
                # First try direct parsing
                data = json.loads(match)
            except json.JSONDecodeError:
                # If JSON fails, try escaping newlines within string values
                # This handles cases where LLM outputs literal newlines in code
                try:
                    # Escape unescaped newlines inside strings
                    fixed_match = re.sub(
                        r'("(?:[^"\\]|\\.)*")',
                        lambda m: m.group(0).replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t'),
                        match
                    )
                    data = json.loads(fixed_match)
                except json.JSONDecodeError:
                    continue

            # Handle run_python tool calls
            if data.get("tool") == "run_python":
                params = data.get("parameters", {})
                code = params.get("code")
                if code:
                    code_blocks.append(code)

        if code_blocks:
            return self._combine_blocks(code_blocks)
        return None

    def _combine_blocks(self, blocks: List[str]) -> str:
        """Combine multiple code blocks"""
        # Filter empty blocks and strip
        blocks = [b.strip() for b in blocks if b.strip()]
        return "\n\n".join(blocks)

    def _is_python_like(self, code: str) -> bool:
        """Check if code looks like Python"""
        python_indicators = [
            "def ", "class ", "import ", "from ", "print(",
            "if ", "for ", "while ", "return ", "= ", "# "
        ]
        return any(ind in code for ind in python_indicators)

    def _extract_raw_python(self, text: str) -> Optional[str]:
        """Extract raw Python code from text"""
        lines = text.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            # Detect start of code
            if not in_code:
                if self._is_code_line(line):
                    in_code = True
                    code_lines.append(line)
            else:
                # Continue collecting code
                if self._is_code_line(line) or line.strip() == "":
                    code_lines.append(line)
                elif line.startswith(' ') or line.startswith('\t'):
                    code_lines.append(line)
                else:
                    # End of code block
                    break

        if code_lines:
            return '\n'.join(code_lines)
        return None

    def _is_code_line(self, line: str) -> bool:
        """Check if line is likely code"""
        line = line.strip()
        if not line:
            return False

        code_starts = [
            'def ', 'class ', 'import ', 'from ', 'if ', 'for ',
            'while ', 'try:', 'except', 'with ', 'return ', '@',
            '#', '"""', "'''", 'print(', 'assert '
        ]

        return any(line.startswith(s) for s in code_starts)


# =============================================================================
# TEST RUNNER
# =============================================================================

@dataclass
class TestResult:
    """Result of a single test case execution"""
    name: str
    passed: bool
    weight: float
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    test_type: str = "positive"


class TestRunner:
    """
    Runs test cases against agent-generated code.

    Each test case has:
    - setup: Code to prepare test environment
    - test: The actual test (assertions)
    - teardown: Cleanup code

    For positive tests, the test should PASS.
    For negative tests, the test should FAIL (e.g., checking for absence of hallucinated library).
    """

    def __init__(self, sandbox: CodeSandbox):
        """
        Initialize test runner.

        Args:
            sandbox: Code sandbox for safe execution
        """
        self.sandbox = sandbox

    def run_test(
        self,
        agent_code: str,
        test_case: TestCase,
        env: ExecutionEnvironment
    ) -> TestResult:
        """
        Run a single test case.

        Args:
            agent_code: Code generated by agent
            test_case: Test case to run
            env: Execution environment config

        Returns:
            TestResult with pass/fail and details
        """
        # Build complete test script with environment constraints
        test_script = self._build_test_script(agent_code, test_case, env)

        # Execute
        result = self.sandbox.execute_python(test_script)

        # Determine pass/fail based on test type
        if test_case.test_type == "positive":
            # Positive test: should succeed (returncode 0)
            passed = result.get("success", False)
        else:
            # Negative test: should fail (non-zero returncode)
            # E.g., test that hallucinated library is NOT used
            passed = not result.get("success", False)

        return TestResult(
            name=test_case.name,
            passed=passed,
            weight=test_case.weight,
            error=result.get("error") or (result.get("stderr") if not passed else None),
            stdout=result.get("stdout", ""),
            stderr=result.get("stderr", ""),
            test_type=test_case.test_type
        )

    def _build_test_script(
        self,
        agent_code: str,
        test_case: TestCase,
        env: ExecutionEnvironment
    ) -> str:
        """
        Build complete test script.

        Order:
        1. Environment constraints (allowed imports)
        2. Agent's code (defines functions)
        3. Setup code (prepare environment)
        4. Test code (assertions)
        5. Teardown code (cleanup)

        Args:
            agent_code: Agent's generated code
            test_case: Test case with setup/test/teardown
            env: Execution environment config

        Returns:
            Complete executable test script
        """
        parts = []

        # Add import restriction comment (for documentation)
        if env.allowed_imports:
            parts.append(f"# Allowed imports: {', '.join(env.allowed_imports)}")
            parts.append(f"# Timeout: {env.timeout_seconds}s")
            parts.append("")

        # Agent's code first (definitions)
        parts.append("# Agent Code")
        parts.append(agent_code)
        parts.append("")

        # Setup
        if test_case.setup:
            parts.append("# Test Setup")
            parts.append(test_case.setup)
            parts.append("")

        # Test
        if test_case.test:
            parts.append("# Test")
            parts.append(test_case.test)
            parts.append("")

        # Teardown (in finally block to ensure cleanup)
        if test_case.teardown:
            # Wrap everything in try/finally for cleanup
            script = "\n".join(parts)
            return f"""
try:
{self._indent(script)}
finally:
    # Teardown
{self._indent(test_case.teardown)}
"""

        return "\n".join(parts)

    def _indent(self, code: str, spaces: int = 4) -> str:
        """Indent code block"""
        indent = " " * spaces
        return "\n".join(indent + line for line in code.split("\n"))


# =============================================================================
# EXECUTION VALIDATOR
# =============================================================================

class ExecutionValidator:
    """
    Main execution validator for AVER benchmark.

    Validates agent recovery through code execution:
    1. Extracts code from agent output
    2. Validates syntax
    3. Runs test suite
    4. Calculates weighted score

    If execution fails completely, falls back to criteria-based scoring
    with a capped maximum score.
    """

    def __init__(self, use_docker: bool = False):
        """
        Initialize validator.

        Args:
            use_docker: Whether to use Docker sandbox (more secure)
        """
        self.use_docker = use_docker
        self.code_extractor = CodeExtractor()

    def validate(
        self,
        agent_output: str,
        task: TaskScenario
    ) -> ExecutionResult:
        """
        Validate agent's recovery through execution.

        Args:
            agent_output: Agent's final output
            task: Task scenario with test suite

        Returns:
            ExecutionResult with scores and details
        """
        # Check if task has execution validity config
        exec_config = task.execution_validity
        if not exec_config or not exec_config.enabled:
            return ExecutionResult(
                executed=False,
                tests_passed=0,
                tests_total=0,
                weighted_score=0.0,
                execution_error="Execution validation not enabled for this task",
                confidence="low"
            )

        # Extract code from output
        code = self.code_extractor.extract(agent_output)

        if not code:
            return ExecutionResult(
                executed=False,
                tests_passed=0,
                tests_total=len(exec_config.test_suite),
                weighted_score=0.0,
                execution_error="No code found in agent output",
                confidence="low"
            )

        # Validate syntax first
        syntax_result = self._validate_syntax(code)
        if not syntax_result["valid"]:
            return ExecutionResult(
                executed=False,
                tests_passed=0,
                tests_total=len(exec_config.test_suite),
                weighted_score=0.0,
                execution_error=f"Syntax error: {syntax_result['error']}",
                test_results=[{
                    "name": "syntax_validation",
                    "passed": False,
                    "error": syntax_result["error"]
                }],
                confidence="low"
            )

        # Create sandbox and runner
        sandbox = CodeSandbox(
            timeout_seconds=exec_config.environment.timeout_seconds,
            memory_limit=f"{exec_config.environment.memory_limit_mb}m",
            use_docker=self.use_docker
        )
        runner = TestRunner(sandbox)

        # Run test suite
        test_results = []
        for test_case in exec_config.test_suite:
            result = runner.run_test(code, test_case, exec_config.environment)
            test_results.append(result)

        # Calculate weighted score
        total_weight = sum(tc.weight for tc in exec_config.test_suite)
        weighted_passed = sum(
            r.weight for r in test_results if r.passed
        )

        if total_weight > 0:
            weighted_score = weighted_passed / total_weight
        else:
            weighted_score = 0.0

        tests_passed = sum(1 for r in test_results if r.passed)

        return ExecutionResult(
            executed=True,
            tests_passed=tests_passed,
            tests_total=len(test_results),
            weighted_score=weighted_score,
            execution_error=None,
            test_results=[
                {
                    "name": r.name,
                    "passed": r.passed,
                    "weight": r.weight,
                    "test_type": r.test_type,
                    "error": r.error,
                    "stdout": r.stdout[:500] if r.stdout else "",
                    "stderr": r.stderr[:500] if r.stderr else ""
                }
                for r in test_results
            ],
            confidence="high"
        )

    def _validate_syntax(self, code: str) -> Dict[str, Any]:
        """Validate Python syntax"""
        try:
            compile(code, "<agent_code>", "exec")
            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {
                "valid": False,
                "error": f"Line {e.lineno}: {e.msg}"
            }


# =============================================================================
# FALLBACK VALIDATOR
# =============================================================================

class FallbackValidator:
    """
    Fallback validation when execution is not possible.

    Uses criteria matching with a capped maximum score.
    This ensures that even when execution fails, we can still
    provide some assessment, but with lower confidence.
    """

    def validate(
        self,
        agent_output: str,
        task: TaskScenario,
        max_score: float = 0.5
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Fallback validation using criteria matching.

        Args:
            agent_output: Agent's output
            task: Task scenario
            max_score: Maximum achievable score (default 0.5)

        Returns:
            (score, details) tuple
        """
        output_lower = agent_output.lower()
        details = {
            "validation_method": "criteria_matching",
            "confidence": "low",
            "max_possible_score": max_score,
            "criteria_matched": []
        }

        # Check success criteria
        success_matches = 0
        for criterion in task.recovery_criteria.success:
            if criterion.lower() in output_lower:
                success_matches += 1
                details["criteria_matched"].append({
                    "criterion": criterion,
                    "type": "success"
                })

        # Check failure criteria (blockers)
        for criterion in task.recovery_criteria.failure:
            if criterion.lower() in output_lower:
                details["failure_criterion_found"] = criterion
                return 0.0, details

        # Calculate score (capped at max_score)
        if task.recovery_criteria.success:
            raw_score = success_matches / len(task.recovery_criteria.success)
            score = min(raw_score, max_score)
        else:
            score = 0.0

        details["raw_score"] = raw_score if task.recovery_criteria.success else 0.0
        details["final_score"] = score

        return score, details


# =============================================================================
# COMBINED RECOVERY VALIDATOR
# =============================================================================

class RecoveryValidator:
    """
    Combined recovery validator that tries execution first, then falls back.

    Strategy:
    1. If task has execution tests → run execution validation
    2. If execution succeeds → use execution score (high confidence)
    3. If execution fails → use fallback (low confidence, capped score)
    4. If no execution tests → use fallback only
    """

    def __init__(self, use_docker: bool = False):
        """
        Initialize validator.

        Args:
            use_docker: Whether to use Docker for execution
        """
        self.execution_validator = ExecutionValidator(use_docker=use_docker)
        self.fallback_validator = FallbackValidator()

    def validate(
        self,
        agent_output: str,
        task: TaskScenario
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Validate recovery using best available method.

        Args:
            agent_output: Agent's output
            task: Task scenario

        Returns:
            (score, details) tuple
        """
        # Check if task has execution tests
        if task.has_execution_tests():
            # Try execution validation
            exec_result = self.execution_validator.validate(agent_output, task)

            if exec_result.executed:
                # Execution succeeded - use execution score
                return exec_result.weighted_score, {
                    "validation_method": "execution_test",
                    "confidence": "high",
                    "execution_result": exec_result.to_dict()
                }
            else:
                # Execution failed - use fallback with cap
                max_score = task.execution_validity.fallback_max_score
                score, details = self.fallback_validator.validate(
                    agent_output, task, max_score
                )
                details["execution_attempted"] = True
                details["execution_error"] = exec_result.execution_error
                return score, details
        else:
            # No execution tests - use fallback only
            score, details = self.fallback_validator.validate(agent_output, task)
            details["execution_tests_available"] = False
            return score, details


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def validate_recovery_execution(
    agent_output: str,
    task: TaskScenario,
    use_docker: bool = False
) -> Tuple[float, Dict[str, Any]]:
    """
    Convenience function for recovery validation.

    Args:
        agent_output: Agent's output
        task: Task scenario
        use_docker: Whether to use Docker

    Returns:
        (score, details) tuple
    """
    validator = RecoveryValidator(use_docker=use_docker)
    return validator.validate(agent_output, task)


def extract_code(output: str) -> Optional[str]:
    """
    Convenience function to extract code from output.

    Args:
        output: Agent's output

    Returns:
        Extracted code or None
    """
    extractor = CodeExtractor()
    return extractor.extract(output)
