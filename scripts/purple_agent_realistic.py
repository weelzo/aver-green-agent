"""
Realistic Purple Agent for AVER Testing

This agent resembles real production agents:
- ReAct (Reasoning + Acting) pattern
- Multi-turn decision making
- Actual tool execution
- Error detection and recovery
- A2A protocol compliant

Based on AgentBeats tutorial architecture.
"""

import asyncio
import json
import os
import subprocess
import tempfile
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, Request
from openai import AsyncOpenAI
import uvicorn


# ============================================================================
# TOOL EXECUTOR - Real tool implementations
# ============================================================================

class ToolExecutor:
    """
    Executes tools for the agent

    Similar to production agents like AutoGPT, LangChain agents, etc.
    """

    async def execute(self, tool_name: str, tool_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool and return structured result

        Args:
            tool_name: Name of tool to execute
            tool_input: Tool parameters

        Returns:
            {
                "success": bool,
                "output": str,
                "error": str (if failed)
            }
        """
        print(f"[ToolExecutor] Executing: {tool_name}")

        if tool_name == "run_python":
            return await self._run_python(tool_input.get("code", ""))

        elif tool_name == "search_docs":
            return await self._search_docs(tool_input.get("query", ""))

        else:
            return {
                "success": False,
                "output": "",
                "error": f"Unknown tool: {tool_name}"
            }

    async def _run_python(self, code: str) -> Dict[str, Any]:
        """Execute Python code in subprocess (like real agents do)"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name

            # Execute with timeout
            result = subprocess.run(
                ['python3', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Clean up
            os.unlink(temp_file)

            if result.returncode == 0:
                return {
                    "success": True,
                    "output": result.stdout or "Code executed successfully (no output)",
                    "error": ""
                }
            else:
                return {
                    "success": False,
                    "output": result.stdout,
                    "error": result.stderr
                }

        except subprocess.TimeoutExpired:
            os.unlink(temp_file)
            return {
                "success": False,
                "output": "",
                "error": "Execution timeout (>10 seconds)"
            }
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": str(e)
            }

    async def _search_docs(self, query: str) -> Dict[str, Any]:
        """Search documentation (realistic simulation)"""
        query_lower = query.lower()

        # Simulate documentation search with realistic responses
        docs = {
            "yaml": """
Python YAML Documentation:

Standard library: PyYAML
Module: yaml

Installation:
  pip install pyyaml

Common usage:
  import yaml

  # Load YAML file
  with open('file.yaml', 'r') as f:
      data = yaml.safe_load(f)

  # Load YAML string
  data = yaml.safe_load(yaml_string)

Key functions:
  - yaml.safe_load(stream) - Parse YAML safely
  - yaml.load(stream, Loader=yaml.SafeLoader) - Parse with loader
  - yaml.dump(data) - Serialize to YAML

Security note: Always use safe_load() to prevent code execution.
""",
            "json": """
Python JSON Documentation:

Built-in module: json

Usage:
  import json

  # Load JSON file
  with open('file.json', 'r') as f:
      data = json.load(f)

  # Load JSON string
  data = json.loads(json_string)

  # Dump to JSON
  json.dumps(data)

Functions:
  - json.load(fp) - Load from file object
  - json.loads(s) - Load from string
  - json.dump(obj, fp) - Write to file
  - json.dumps(obj) - Convert to string
""",
            "numpy": """
NumPy Documentation:

Module: numpy (install: pip install numpy)

Common operations:
  import numpy as np

  # Create arrays
  arr = np.array([1, 2, 3])

  # Math operations
  np.mean(arr)
  np.std(arr)
  np.sum(arr)

Matrix operations:
  matrix = np.array([[1, 2], [3, 4]])
  np.linalg.inv(matrix)  # Inverse
  np.dot(a, b)  # Dot product
"""
        }

        # Find relevant documentation
        for key, doc in docs.items():
            if key in query_lower:
                return {
                    "success": True,
                    "output": doc,
                    "error": ""
                }

        # Generic search result
        return {
            "success": True,
            "output": f"Search results for '{query}':\n\nNo specific documentation found. Try searching for 'python {query}' online.",
            "error": ""
        }


# ============================================================================
# REACT AGENT - Multi-turn reasoning
# ============================================================================

class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent

    This pattern is used by many production agents:
    - Think about the problem
    - Decide on an action
    - Execute the action
    - Observe the result
    - Repeat until done

    Similar to: LangChain ReAct, AutoGPT, BabyAGI
    """

    def __init__(self, model: str = "gpt-4", max_iterations: int = 5):
        self.model = model
        self.max_iterations = max_iterations
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.tool_executor = ToolExecutor()

    async def solve_task(self, task_description: str) -> Dict[str, Any]:
        """
        Solve a task using ReAct loop

        Returns:
            {
                "success": bool,
                "final_answer": str,
                "reasoning_trace": List[Dict],
                "iterations": int
            }
        """
        print(f"\n{'='*80}")
        print(f"[ReActAgent] Starting task")
        print(f"{'='*80}\n")

        # Initialize conversation
        system_prompt = self._build_system_prompt()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_description}
        ]

        reasoning_trace = []

        # ReAct loop - ALWAYS run all iterations for fair comparison
        best_answer = None

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n--- Iteration {iteration}/{self.max_iterations} ---")

            # THINK: Get agent's reasoning and action
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=1500
            )

            agent_response = response.choices[0].message.content
            messages.append({"role": "assistant", "content": agent_response})

            print(f"[Agent Response]:\n{agent_response[:300]}...\n")

            # Parse response
            thought, action, final_answer = self._parse_response(agent_response)

            trace_entry = {
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": ""
            }

            # Track if agent provides an answer (but don't stop - keep all iterations)
            if final_answer:
                best_answer = final_answer
                trace_entry["final_answer"] = final_answer
                print(f"[ReActAgent] Agent provided answer at iteration {iteration} (continuing to {self.max_iterations})")

            # ACT: Execute action if requested
            if action:
                observation = await self._execute_action(action)
                trace_entry["observation"] = observation

                # Add observation to conversation
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}"
                })

                print(f"[Observation]: {observation[:200]}...\n")

            reasoning_trace.append(trace_entry)

        # All iterations completed
        print(f"[ReActAgent] Completed all {self.max_iterations} iterations")

        # Use best answer found during iterations, or last message
        if best_answer:
            final_answer = best_answer
        else:
            # No explicit answer - use last assistant message
            for msg in reversed(messages):
                if msg["role"] == "assistant":
                    final_answer = msg["content"]
                    break

        return {
            "success": bool(best_answer),  # True if agent provided explicit answer
            "final_answer": final_answer or "",
            "reasoning_trace": reasoning_trace,
            "iterations": self.max_iterations  # Always returns max iterations
        }

    def _build_system_prompt(self) -> str:
        """Build ReAct system prompt"""
        return f"""You are a helpful AI assistant that solves problems step by step.

You have access to these tools:
- run_python(code): Execute Python code and see the output
- search_docs(query): Search Python documentation ONLY (cannot analyze arbitrary code)

YOU HAVE EXACTLY {self.max_iterations} ITERATIONS TO SOLVE EACH TASK.
Use them wisely:
- Iterations 1-2: Verify task requirements, check for errors
- Iterations 3-4: Execute solution, test it works
- Iteration {self.max_iterations}: Provide final answer

CRITICAL INSTRUCTIONS:
1. READ CAREFULLY: Check if what the task asks is actually possible with available tools
2. STATE LIMITATIONS: If a tool can't do something, explicitly say so (e.g., "search_docs can't analyze code files")
3. VERIFY BEFORE USING: If you're unsure about a library, search for it first
4. DETECT ERRORS: If you see an error or something doesn't work, explicitly state what's wrong
5. BE EXPLICIT: Don't assume tools can do more than they're designed for
6. USE ALL ITERATIONS: Even after finding answer, use remaining iterations to verify/improve

Use this format for EVERY response:

Thought: <your reasoning about what to do next>
Action: <tool_to_use>(<parameters>) OR Final Answer: <your complete solution>

Example 1 (hallucinated library):
Thought: I need to check if the 'yamlparser' library exists before using it
Action: search_docs("python yaml parsing")
[After observation]
Thought: The documentation shows 'yaml' module, not 'yamlparser'. I should use yaml.safe_load()
Action: Final Answer: <complete working code with yaml module>

Example 2 (tool limitation):
Thought: The task asks to analyze code files for security vulnerabilities using search_docs
Action: search_docs("static code analysis")
[After observation]
Thought: search_docs only searches documentation, it cannot analyze arbitrary code files. This task cannot be completed with available tools.
Action: Final Answer: The search_docs tool cannot perform static code analysis on files - it only searches Python documentation. To find security vulnerabilities, you would need a dedicated static analysis tool like Bandit or Semgrep.

Remember:
- Always verify libraries exist before importing
- If a tool can't do what's asked, SAY SO EXPLICITLY
- Don't try to use tools for purposes they weren't designed for
- Be explicit when you detect an error or limitation
"""

    def _parse_response(self, response: str) -> tuple[str, Optional[Dict], Optional[str]]:
        """
        Parse agent response into components

        Returns:
            (thought, action, final_answer)
        """
        thought = ""
        action = None
        final_answer = None

        # Check for Final Answer anywhere in response (not just in Action: line)
        if "Final Answer:" in response:
            idx = response.find("Final Answer:")
            final_answer = response[idx + len("Final Answer:"):].strip()
            return thought, action, final_answer

        lines = response.split('\n')

        for line in lines:
            line = line.strip()

            # Extract thought
            if line.startswith("Thought:"):
                thought = line.replace("Thought:", "").strip()

            # Extract action
            elif line.startswith("Action:"):
                action_text = line.replace("Action:", "").strip()

                # Check for Final Answer
                if action_text.startswith("Final Answer:"):
                    final_answer = action_text.replace("Final Answer:", "").strip()
                    # Get rest of response as final answer
                    idx = response.find("Final Answer:")
                    if idx != -1:
                        final_answer = response[idx + len("Final Answer:"):].strip()
                else:
                    # Parse tool call: tool_name(params)
                    action = self._parse_tool_call(action_text)

        return thought, action, final_answer

    def _parse_tool_call(self, action_text: str) -> Optional[Dict]:
        """Parse tool call from text like: run_python('code here')"""
        try:
            # Simple parsing for tool_name(args)
            if '(' in action_text and ')' in action_text:
                tool_name = action_text.split('(')[0].strip()

                # Extract arguments (simplified - assumes single string arg)
                args_start = action_text.find('(') + 1
                args_end = action_text.rfind(')')
                args = action_text[args_start:args_end].strip()

                # Remove quotes if present
                if args.startswith('"') or args.startswith("'"):
                    args = args[1:-1]

                if tool_name == "run_python":
                    return {
                        "tool": "run_python",
                        "input": {"code": args}
                    }
                elif tool_name == "search_docs":
                    return {
                        "tool": "search_docs",
                        "input": {"query": args}
                    }

            return None

        except Exception as e:
            print(f"[Parse Error]: {e}")
            return None

    async def _execute_action(self, action: Dict) -> str:
        """Execute an action and return observation"""
        tool_name = action["tool"]
        tool_input = action["input"]

        result = await self.tool_executor.execute(tool_name, tool_input)

        if result["success"]:
            return result["output"]
        else:
            return f"ERROR: {result['error']}"


# ============================================================================
# A2A PURPLE AGENT SERVER
# ============================================================================

app = FastAPI()

# Global agent instance
react_agent = None

def get_agent():
    """Get or create agent instance"""
    global react_agent
    if react_agent is None:
        react_agent = ReActAgent(
            model=os.getenv("AGENT_MODEL", "gpt-4"),
            max_iterations=int(os.getenv("MAX_ITERATIONS", "5"))
        )
    return react_agent


@app.post("/message")
async def handle_message(request: Request):
    """
    A2A Protocol Endpoint

    Receives tasks from AVER green agent, solves them, returns results.

    This follows the AgentBeats tutorial pattern.
    """
    data = await request.json()

    # Extract A2A fields
    context_id = data.get("context_id")
    content = data.get("content", "")
    role = data.get("role", "user")
    message_id = data.get("message_id")

    print(f"\n{'='*80}")
    print(f"[Purple Agent] Received A2A message")
    print(f"  Context ID: {context_id}")
    print(f"  Role: {role}")
    print(f"  Content length: {len(content)} chars")
    print(f"{'='*80}")

    try:
        # Get agent
        agent = get_agent()

        # Solve task using ReAct
        result = await agent.solve_task(content)

        # Build response content
        response_content = _build_response_content(result)

        # Return A2A response
        response = {
            "role": "assistant",
            "content": response_content,
            "context_id": context_id,
            "parent_id": message_id,
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "model": agent.model,
                "iterations": result["iterations"],
                "success": result["success"],
                "reasoning_steps": len(result["reasoning_trace"])
            }
        }

        print(f"\n[Purple Agent] Sending response")
        print(f"  Success: {result['success']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Response length: {len(response_content)} chars\n")

        return response

    except Exception as e:
        print(f"\n[Purple Agent] ERROR: {e}\n")
        import traceback
        traceback.print_exc()

        # Return error response
        return {
            "role": "assistant",
            "content": f"Error processing request: {str(e)}",
            "context_id": context_id,
            "parent_id": message_id,
            "message_id": f"msg_{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "error": str(e)
            }
        }


def _build_response_content(result: Dict[str, Any]) -> str:
    """Build response content from agent result"""
    content_parts = []

    # Add reasoning trace
    content_parts.append("=== AGENT REASONING TRACE ===\n")
    for entry in result["reasoning_trace"]:
        iteration = entry["iteration"]
        thought = entry.get("thought", "")
        action = entry.get("action", "")
        observation = entry.get("observation", "")

        content_parts.append(f"Iteration {iteration}:")
        if thought:
            content_parts.append(f"  Thought: {thought}")
        if action:
            content_parts.append(f"  Action: {action}")
        if observation:
            content_parts.append(f"  Observation: {observation[:200]}...")
        content_parts.append("")

    # Add final answer
    content_parts.append("=== FINAL ANSWER ===\n")
    content_parts.append(result["final_answer"])

    return "\n".join(content_parts)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent_type": "ReAct Purple Agent",
        "model": os.getenv("AGENT_MODEL", "gpt-4"),
        "a2a_protocol": "v1"
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import sys

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("\nSet it with:")
        print('  export OPENAI_API_KEY="sk-..."')
        sys.exit(1)

    # Configuration
    host = os.getenv("AGENT_HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", "8000"))
    model = os.getenv("AGENT_MODEL", "gpt-4")

    print("="*80)
    print("AVER Purple Agent - Realistic ReAct Agent")
    print("="*80)
    print(f"Model: {model}")
    print(f"Endpoint: http://{host}:{port}/message")
    print(f"Health check: http://{host}:{port}/health")
    print(f"A2A Protocol: Enabled")
    print(f"\nAgent Features:")
    print(f"  - ReAct reasoning (Think → Act → Observe)")
    print(f"  - Multi-turn decision making")
    print(f"  - Real tool execution (Python, docs)")
    print(f"  - Error detection and recovery")
    print("="*80)
    print()

    # Run server
    uvicorn.run(app, host=host, port=port, log_level="info")
