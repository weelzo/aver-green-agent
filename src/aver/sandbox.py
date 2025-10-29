"""
Sandbox Environment

Provides safe code execution for AVER tasks.

Uses Docker containers to execute code in isolated environments.
"""

import subprocess
import tempfile
import os
from typing import Dict, Any, Optional
from pathlib import Path


class CodeSandbox:
    """
    Safe code execution environment

    Executes code in isolated Docker containers with resource limits.
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        memory_limit: str = "512m",
        use_docker: bool = False
    ):
        """
        Initialize sandbox

        Args:
            timeout_seconds: Max execution time
            memory_limit: Memory limit for container
            use_docker: Whether to use Docker (False = local subprocess)
        """
        self.timeout_seconds = timeout_seconds
        self.memory_limit = memory_limit
        self.use_docker = use_docker

    def execute_python(
        self,
        code: str,
        stdin: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code safely

        Args:
            code: Python code to execute
            stdin: Optional standard input

        Returns:
            Dict with stdout, stderr, returncode, and error info
        """
        if self.use_docker:
            return self._execute_docker(code, stdin)
        else:
            return self._execute_subprocess(code, stdin)

    def _execute_subprocess(
        self,
        code: str,
        stdin: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute code in subprocess (simple, no Docker)

        Args:
            code: Python code
            stdin: Optional input

        Returns:
            Execution result
        """
        # Create temporary file for code
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Run code with timeout
            result = subprocess.run(
                ['python3', temp_file],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0,
                "error": None,
                "timeout": False
            }

        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": "Execution timed out",
                "returncode": -1,
                "success": False,
                "error": "Timeout",
                "timeout": True
            }

        except Exception as e:
            return {
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "success": False,
                "error": str(e),
                "timeout": False
            }

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass

    def _execute_docker(
        self,
        code: str,
        stdin: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute code in Docker container

        Args:
            code: Python code
            stdin: Optional input

        Returns:
            Execution result
        """
        # Create temporary directory with code
        with tempfile.TemporaryDirectory() as tmpdir:
            code_file = os.path.join(tmpdir, "script.py")

            with open(code_file, 'w') as f:
                f.write(code)

            try:
                # Run Docker container
                cmd = [
                    "docker", "run",
                    "--rm",
                    "-v", f"{tmpdir}:/workspace",
                    "-w", "/workspace",
                    "--memory", self.memory_limit,
                    "--network", "none",  # No network access
                    "python:3.11-slim",
                    "python", "script.py"
                ]

                result = subprocess.run(
                    cmd,
                    input=stdin,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds
                )

                return {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                    "success": result.returncode == 0,
                    "error": None,
                    "timeout": False
                }

            except subprocess.TimeoutExpired:
                return {
                    "stdout": "",
                    "stderr": "Execution timed out",
                    "returncode": -1,
                    "success": False,
                    "error": "Timeout",
                    "timeout": True
                }

            except Exception as e:
                return {
                    "stdout": "",
                    "stderr": str(e),
                    "returncode": -1,
                    "success": False,
                    "error": str(e),
                    "timeout": False
                }

    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate code syntax without executing

        Args:
            code: Python code to validate

        Returns:
            Validation result
        """
        try:
            compile(code, "<string>", "exec")
            return {
                "valid": True,
                "error": None
            }
        except SyntaxError as e:
            return {
                "valid": False,
                "error": str(e),
                "line": e.lineno,
                "offset": e.offset
            }


class ToolExecutor:
    """
    Executes AVER task tools

    Provides a safe interface for agents to use tools like run_python, search_docs, etc.
    """

    def __init__(self, sandbox: Optional[CodeSandbox] = None):
        """
        Initialize tool executor

        Args:
            sandbox: Code sandbox for safe execution
        """
        self.sandbox = sandbox or CodeSandbox()

    def execute_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool

        Args:
            tool_name: Name of tool to execute
            tool_input: Tool parameters

        Returns:
            Tool output
        """
        if tool_name == "run_python":
            return self._run_python(tool_input)

        elif tool_name == "search_docs":
            return self._search_docs(tool_input)

        else:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }

    def _run_python(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Python code

        Args:
            tool_input: Must contain "code" key

        Returns:
            Execution result
        """
        code = tool_input.get("code", "")

        if not code:
            return {
                "success": False,
                "error": "No code provided"
            }

        # Validate syntax first
        validation = self.sandbox.validate_code(code)
        if not validation["valid"]:
            return {
                "success": False,
                "error": f"Syntax error: {validation['error']}",
                "stdout": "",
                "stderr": validation['error']
            }

        # Execute
        result = self.sandbox.execute_python(code)
        return result

    def _search_docs(self, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search documentation (mock implementation)

        Args:
            tool_input: Must contain "query" key

        Returns:
            Search results
        """
        query = tool_input.get("query", "").lower()

        # Mock documentation database
        docs = {
            "yaml": "PyYAML is the standard library for YAML parsing in Python. "
                   "Use yaml.safe_load() to parse YAML files safely. "
                   "Install with: pip install pyyaml",

            "json": "The json module is part of Python's standard library. "
                   "Use json.load() to parse JSON files and json.loads() for strings.",

            "requests": "The requests library is used for HTTP requests. "
                       "Install with: pip install requests",
        }

        # Simple keyword matching
        results = []
        for topic, content in docs.items():
            if topic in query or query in content.lower():
                results.append({
                    "topic": topic,
                    "content": content
                })

        if results:
            return {
                "success": True,
                "results": results,
                "output": "\n\n".join([f"{r['topic']}: {r['content']}" for r in results])
            }
        else:
            return {
                "success": True,
                "results": [],
                "output": f"No documentation found for: {query}"
            }


if __name__ == "__main__":
    print("Sandbox module loaded")
    print("\nExample usage:")
    print("  sandbox = CodeSandbox()")
    print("  result = sandbox.execute_python('print(\"Hello, AVER!\")')")
    print("  print(result['stdout'])")

    # Test
    print("\n" + "="*80)
    print("Testing sandbox...")
    sandbox = CodeSandbox()

    code = """
import sys
print("Hello from sandbox!")
print(f"Python version: {sys.version}")
"""

    result = sandbox.execute_python(code)
    print("\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Stdout: {result['stdout']}")
    print(f"  Stderr: {result['stderr']}")
