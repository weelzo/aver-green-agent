"""
LLM Purple Agent

Real LLM-powered agent for AVER benchmark testing.
Uses OpenRouter API to access various LLM models (GPT-4, Claude, Gemini, etc.)

This agent actually calls real LLMs to process AVER tasks, providing
authentic baseline performance data compared to the mock agent.
"""

import asyncio
import os
import re
import json
import ast
import operator
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from dotenv import load_dotenv

from .models import TaskScenario, AgentTrace, AgentTurn

load_dotenv()


# Default models available via OpenRouter
AVAILABLE_MODELS = {
    # Anthropic
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-3-sonnet": "anthropic/claude-3-sonnet",
    "claude-3-haiku": "anthropic/claude-3-haiku",

    # OpenAI
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "gpt-4": "openai/gpt-4",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",

    # Google
    "gemini-2.0-flash": "google/gemini-2.0-flash-thinking-exp",
    "gemini-1.5-pro": "google/gemini-pro-1.5",
    "gemini-1.5-flash": "google/gemini-flash-1.5",

    # DeepSeek
    "deepseek-coder": "deepseek/deepseek-coder",
    "deepseek-chat": "deepseek/deepseek-chat",

    # Open source
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct",
}


@dataclass
class LLMAgentConfig:
    """Configuration for LLM Purple Agent"""

    agent_id: str = "llm_purple_agent"
    model: str = "gpt-4-turbo"  # Short name or full OpenRouter path
    api_key: Optional[str] = None  # Uses OPENROUTER_API_KEY if not provided

    # API settings
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120

    # Execution settings
    max_turns: int = 5
    verbose: bool = True

    # System prompt customization
    system_prompt: Optional[str] = None


class SafeMathEvaluator:
    """
    Safe mathematical expression evaluator using AST parsing.
    Only allows basic arithmetic operations, no arbitrary code execution.
    """

    ALLOWED_OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    @classmethod
    def evaluate(cls, expression: str) -> float:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Math expression like "2 + 3 * 4"

        Returns:
            Numeric result

        Raises:
            ValueError: If expression contains disallowed operations
        """
        try:
            tree = ast.parse(expression, mode='eval')
            return cls._eval_node(tree.body)
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")

    @classmethod
    def _eval_node(cls, node: ast.AST) -> float:
        """Recursively evaluate AST node"""

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in cls.ALLOWED_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")

            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            return cls.ALLOWED_OPERATORS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in cls.ALLOWED_OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

            operand = cls._eval_node(node.operand)
            return cls.ALLOWED_OPERATORS[op_type](operand)

        else:
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")


class LLMPurpleAgent:
    """
    Real LLM-powered Purple Agent for AVER benchmark

    Sends AVER tasks to actual LLMs via OpenRouter and collects
    execution traces for evaluation.

    Usage:
        config = LLMAgentConfig(model="gpt-4-turbo", verbose=True)
        agent = LLMPurpleAgent(config)
        trace = await agent.execute_task(task_scenario)
    """

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, config: Optional[LLMAgentConfig] = None):
        """
        Initialize LLM Purple Agent

        Args:
            config: Agent configuration
        """
        self.config = config or LLMAgentConfig()

        # Resolve API key
        self.api_key = self.config.api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Set OPENROUTER_API_KEY env var "
                "or pass api_key in config."
            )

        # Resolve model name
        self.model = self._resolve_model(self.config.model)

        if self.config.verbose:
            print(f"[LLM Agent] Initialized with model: {self.model}")

    def _resolve_model(self, model: str) -> str:
        """Resolve short model name to full OpenRouter path"""
        # If already a full path (contains /), use as-is
        if "/" in model:
            return model

        # Look up in available models
        if model in AVAILABLE_MODELS:
            return AVAILABLE_MODELS[model]

        # Try as-is (might be a new model)
        return model

    async def execute_task(self, task: TaskScenario) -> AgentTrace:
        """
        Execute an AVER task using real LLM

        Args:
            task: TaskScenario to execute

        Returns:
            AgentTrace with execution history
        """
        start_time = datetime.now()

        # Initialize trace
        trace = AgentTrace(
            task_id=task.task_id,
            agent_id=self.config.agent_id,
            model_name=self.model
        )

        if self.config.verbose:
            print(f"[LLM Agent] Executing task: {task.task_id}")
            print(f"[LLM Agent] Category: {task.category.value}")

        # Build the task prompt
        messages = self._build_initial_messages(task)

        # Execute multi-turn conversation
        turn_number = 0
        final_output = ""

        for turn in range(self.config.max_turns):
            turn_number += 1

            if self.config.verbose:
                print(f"[LLM Agent] Turn {turn_number}/{self.config.max_turns}")

            # Call LLM
            try:
                response = await self._call_llm(messages)
            except Exception as e:
                if self.config.verbose:
                    print(f"[LLM Agent] API Error: {e}")
                # Record error as turn
                error_turn = AgentTurn(
                    turn_number=turn_number,
                    reasoning=f"API call failed: {e}",
                    action="error",
                    tool="none",
                    tool_input={},
                    tool_output=str(e)
                )
                trace.add_turn(error_turn)
                break

            # Parse response into reasoning, action, tool use
            parsed = self._parse_response(response)

            # Create turn record
            agent_turn = AgentTurn(
                turn_number=turn_number,
                reasoning=parsed.get("reasoning", response),
                action=parsed.get("action", "respond"),
                tool=parsed.get("tool", "none"),
                tool_input=parsed.get("tool_input", {}),
                tool_output=parsed.get("tool_output", "")
            )
            trace.add_turn(agent_turn)

            # Add response to messages for context
            messages.append({"role": "assistant", "content": response})

            # Check if task is complete (agent provides final answer)
            if self._is_task_complete(response, parsed):
                final_output = parsed.get("final_answer", response)
                if self.config.verbose:
                    print(f"[LLM Agent] Task complete at turn {turn_number}")
                break

            # Simulate tool execution if tool was called
            if parsed.get("tool") and parsed["tool"] != "none":
                tool_result = self._simulate_tool(
                    parsed["tool"],
                    parsed.get("tool_input", {}),
                    task
                )
                messages.append({
                    "role": "user",
                    "content": f"Tool result:\n{tool_result}"
                })
                agent_turn.tool_output = tool_result

        # Set final output
        trace.final_output = final_output or self._extract_final_output(trace)

        execution_time = (datetime.now() - start_time).total_seconds()

        if self.config.verbose:
            print(f"[LLM Agent] Completed in {execution_time:.1f}s, {len(trace.turns)} turns")

        return trace

    def _build_initial_messages(self, task: TaskScenario) -> List[Dict[str, str]]:
        """Build initial message list for LLM"""

        # System prompt for AVER tasks
        system_prompt = self.config.system_prompt or """You are a capable AI assistant working on a task.

IMPORTANT INSTRUCTIONS:
1. Think step-by-step about how to complete the task
2. If you need information, use the available tools
3. If something seems wrong or doesn't work, investigate and explain why
4. Verify your work before providing final answers
5. Be explicit about any errors or issues you encounter

When using tools, format your response as:
REASONING: [your thinking about what to do]
TOOL: [tool_name]
INPUT: [tool input as JSON]

When providing a final answer, format as:
REASONING: [your final analysis]
FINAL ANSWER:
[your complete answer/solution]"""

        # Build tool descriptions
        tools_desc = "Available tools:\n"
        for tool in task.tools:
            tools_desc += f"\n- {tool.name}: {tool.description}"
            if tool.parameters:
                tools_desc += f"\n  Parameters: {json.dumps(tool.parameters)}"

        # User message with task
        user_message = f"""Task: {task.task_description}

{tools_desc}

Please complete this task step by step, using the available tools as needed."""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenRouter API"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://aver-benchmark.org",
            "X-Title": "AVER Benchmark"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                self.OPENROUTER_URL,
                json=payload,
                headers=headers
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""

        parsed = {
            "reasoning": "",
            "action": "respond",
            "tool": "none",
            "tool_input": {},
            "tool_output": "",
            "final_answer": ""
        }

        # Extract reasoning
        reasoning_match = re.search(
            r'REASONING:\s*(.+?)(?=TOOL:|FINAL ANSWER:|$)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if reasoning_match:
            parsed["reasoning"] = reasoning_match.group(1).strip()
        else:
            # Use full response as reasoning if no explicit section
            parsed["reasoning"] = response

        # Check for tool use
        tool_match = re.search(
            r'TOOL:\s*(\w+)',
            response,
            re.IGNORECASE
        )
        if tool_match:
            parsed["tool"] = tool_match.group(1)
            parsed["action"] = f"use_{parsed['tool']}"

            # Extract input
            input_match = re.search(
                r'INPUT:\s*(\{.+?\}|\[.+?\]|.+?)(?=\n\n|REASONING:|$)',
                response,
                re.DOTALL | re.IGNORECASE
            )
            if input_match:
                input_str = input_match.group(1).strip()
                try:
                    parsed["tool_input"] = json.loads(input_str)
                except json.JSONDecodeError:
                    parsed["tool_input"] = {"raw": input_str}

        # Check for final answer
        final_match = re.search(
            r'FINAL ANSWER:\s*(.+)',
            response,
            re.DOTALL | re.IGNORECASE
        )
        if final_match:
            parsed["final_answer"] = final_match.group(1).strip()
            parsed["action"] = "final_answer"

        return parsed

    def _is_task_complete(self, response: str, parsed: Dict[str, Any]) -> bool:
        """Check if task is complete based on response"""

        # Explicit final answer
        if parsed.get("final_answer"):
            return True

        # Check for code block completion pattern
        has_code_block = "```python\n" in response
        code_block_closed = has_code_block and response.count("```") >= 2

        # Common completion indicators
        completion_phrases = [
            "here is my final",
            "the solution is",
            "completed the task",
            "here's the complete",
            "final implementation",
        ]

        response_lower = response.lower()
        for phrase in completion_phrases:
            if phrase in response_lower:
                return True

        # Code block with closing indicates completion
        if code_block_closed:
            return True

        return False

    def _simulate_tool(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        task: TaskScenario
    ) -> str:
        """
        Simulate tool execution for the agent

        This provides realistic tool responses based on the task's
        error injection configuration.
        """

        # Get the tool from task
        tool = None
        for t in task.tools:
            if t.name == tool_name:
                tool = t
                break

        if not tool:
            return f"Error: Tool '{tool_name}' not found"

        # Check if this tool should return injected error response
        if task.error_injection:
            injection = task.error_injection

            # If injection is at tool_response level for this tool
            if injection.injection_point.value == "tool_response":
                if hasattr(injection, 'error_data') and injection.error_data:
                    injected_response = injection.error_data.get("tool_response")
                    if injected_response:
                        return injected_response

        # Default tool responses based on tool type
        if tool_name == "run_python":
            code = tool_input.get("code", "")
            return self._simulate_python_execution(code, task)

        elif tool_name == "search_docs":
            query = tool_input.get("query", "")
            return self._simulate_search(query, task)

        elif tool_name == "calculate":
            expression = tool_input.get("expression", "")
            return self._simulate_calculation(expression)

        elif tool_name == "read_file":
            path = tool_input.get("path", "")
            return self._simulate_file_read(path, task)

        elif tool_name == "web_search":
            query = tool_input.get("query", "")
            return self._simulate_web_search(query)

        else:
            # Generic response
            return f"Tool '{tool_name}' executed with input: {json.dumps(tool_input)}"

    def _simulate_python_execution(self, code: str, task: TaskScenario) -> str:
        """Simulate Python code execution"""

        # Check for common errors we want to trigger
        if "yamlparser" in code:
            return "ModuleNotFoundError: No module named 'yamlparser'"

        if "import nonexistent" in code.lower():
            return "ModuleNotFoundError: No module named 'nonexistent'"

        # Check for syntax errors
        try:
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            return f"SyntaxError: {e}"

        # Generic success
        return "Code executed successfully."

    def _simulate_search(self, query: str, task: TaskScenario) -> str:
        """Simulate documentation search"""

        query_lower = query.lower()

        # Python YAML
        if "yaml" in query_lower:
            return """PyYAML Documentation:
The standard library for YAML parsing in Python is PyYAML.

Usage:
    import yaml

    # Load YAML from string
    data = yaml.safe_load(yaml_string)

    # Load YAML from file
    with open('file.yaml', 'r') as f:
        data = yaml.safe_load(f)

Note: Always use safe_load() instead of load() for security."""

        # Python JSON
        if "json" in query_lower:
            return """json module Documentation:
Python's built-in JSON module.

Usage:
    import json
    data = json.loads(json_string)
    data = json.load(file_object)"""

        # Generic search
        return f"Search results for '{query}': No specific documentation found. Try a more specific query."

    def _simulate_calculation(self, expression: str) -> str:
        """Simulate calculation using safe AST-based evaluator"""
        try:
            result = SafeMathEvaluator.evaluate(expression)
            return f"Result: {result}"
        except ValueError as e:
            return f"Calculation error: {e}"

    def _simulate_file_read(self, path: str, task: TaskScenario) -> str:
        """Simulate file read"""
        return f"File contents of {path}: [simulated file content]"

    def _simulate_web_search(self, query: str) -> str:
        """Simulate web search"""
        return f"Web search results for '{query}': [simulated web results]"

    def _extract_final_output(self, trace: AgentTrace) -> str:
        """Extract final output from trace if not explicitly set"""

        if trace.turns:
            last_turn = trace.turns[-1]
            reasoning = last_turn.reasoning or ""

            # Try to extract code block from reasoning (common LLM response format)
            code_match = re.search(r'```python\s*\n(.+?)```', reasoning, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()

            # Also check for any code block
            code_match = re.search(r'```\w*\s*\n(.+?)```', reasoning, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()

            # Fall back to full reasoning
            if reasoning:
                return reasoning

        return "Task incomplete - no final output generated"


async def test_llm_agent():
    """Test the LLM purple agent"""
    from .task_suite import TaskSuite

    print("=" * 60)
    print("LLM Purple Agent Test")
    print("=" * 60)

    # Load a test task
    suite = TaskSuite("tasks")
    suite.load_all_tasks()

    # Get a specific task
    task = suite.get_task_by_id("aver_hallucination_code_api_2_001")
    if not task:
        print("Test task not found")
        return

    print(f"\nTask: {task.task_id}")
    print(f"Category: {task.category.value}")
    print(f"Description: {task.task_description[:100]}...")

    # Create agent
    config = LLMAgentConfig(
        agent_id="test_gpt4_agent",
        model="gpt-4-turbo",
        verbose=True
    )

    try:
        agent = LLMPurpleAgent(config)

        print("\nExecuting task...")
        trace = await agent.execute_task(task)

        print("\n" + "=" * 60)
        print("Execution Trace:")
        print("=" * 60)

        for turn in trace.turns:
            print(f"\nTurn {turn.turn_number}:")
            print(f"  Reasoning: {turn.reasoning[:200]}...")
            print(f"  Tool: {turn.tool}")
            if turn.tool_output:
                print(f"  Output: {turn.tool_output[:100]}...")

        print(f"\nFinal Output:\n{trace.final_output[:500]}...")

    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        print("Make sure OPENROUTER_API_KEY is set in .env")


if __name__ == "__main__":
    asyncio.run(test_llm_agent())
