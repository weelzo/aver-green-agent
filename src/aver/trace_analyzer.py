"""
Trace Analyzer

Analyzes agent execution traces to extract insights and patterns.

Used by the evaluator to detect error detection signals and behaviors.
"""

from typing import List, Dict, Any, Optional
import re

from .models import AgentTrace, AgentTurn


class TraceAnalyzer:
    """
    Analyzes agent execution traces

    Provides methods to:
    - Extract reasoning patterns
    - Detect error mentions
    - Identify verification behaviors
    - Analyze tool usage patterns
    """

    def extract_reasoning(self, trace: AgentTrace) -> str:
        """
        Extract all reasoning text from trace

        Args:
            trace: Agent execution trace

        Returns:
            Combined reasoning text
        """
        reasoning_parts = []

        for turn in trace.turns:
            if turn.reasoning:
                reasoning_parts.append(f"Turn {turn.turn_number}: {turn.reasoning}")

        return "\n\n".join(reasoning_parts)

    def find_explicit_mentions(
        self,
        trace: AgentTrace,
        patterns: List[str],
        case_sensitive: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Find explicit mentions of patterns in reasoning

        Args:
            trace: Agent execution trace
            patterns: List of text patterns to search for
            case_sensitive: Whether search should be case sensitive

        Returns:
            List of matches with turn numbers and context
        """
        matches = []

        for turn in trace.turns:
            reasoning = turn.reasoning

            if not case_sensitive:
                reasoning = reasoning.lower()

            for pattern in patterns:
                search_pattern = pattern if case_sensitive else pattern.lower()

                if search_pattern in reasoning:
                    matches.append({
                        "turn": turn.turn_number,
                        "pattern": pattern,
                        "context": turn.reasoning[:200]  # First 200 chars
                    })

        return matches

    def find_tool_usage(
        self,
        trace: AgentTrace,
        tool_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find tool usage in trace

        Args:
            trace: Agent execution trace
            tool_name: Specific tool to search for (None = all tools)

        Returns:
            List of tool usage with details
        """
        usage = []

        for turn in trace.turns:
            if turn.tool:
                if tool_name is None or turn.tool == tool_name:
                    usage.append({
                        "turn": turn.turn_number,
                        "tool": turn.tool,
                        "input": turn.tool_input,
                        "output": turn.tool_output
                    })

        return usage

    def detect_verification_behavior(
        self,
        trace: AgentTrace,
        verification_patterns: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Detect verification behaviors (implicit detection signals)

        Args:
            trace: Agent execution trace
            verification_patterns: Patterns indicating verification

        Returns:
            List of detected verification behaviors
        """
        behaviors = []

        for turn in trace.turns:
            # Check action text
            action = turn.action.lower() if turn.action else ""

            for pattern in verification_patterns:
                if pattern.lower() in action:
                    behaviors.append({
                        "turn": turn.turn_number,
                        "pattern": pattern,
                        "action": turn.action,
                        "type": "action"
                    })

            # Check tool usage
            if turn.tool:
                tool_name = turn.tool.lower()

                for pattern in verification_patterns:
                    # Check if pattern matches tool name or tool input
                    if pattern.lower() in tool_name:
                        behaviors.append({
                            "turn": turn.turn_number,
                            "pattern": pattern,
                            "tool": turn.tool,
                            "type": "tool_usage"
                        })

        return behaviors

    def get_turn_summary(self, trace: AgentTrace) -> Dict[str, Any]:
        """
        Get summary statistics about trace

        Args:
            trace: Agent execution trace

        Returns:
            Summary dictionary
        """
        return {
            "num_turns": len(trace.turns),
            "tools_used": list(set(t.tool for t in trace.turns if t.tool)),
            "has_reasoning": any(t.reasoning for t in trace.turns),
            "has_final_output": bool(trace.final_output)
        }

    def detect_error_patterns(
        self,
        trace: AgentTrace,
        error_keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect common error patterns in reasoning

        Args:
            trace: Agent execution trace
            error_keywords: Keywords indicating errors (default: common patterns)

        Returns:
            List of detected error patterns
        """
        if error_keywords is None:
            error_keywords = [
                "error", "doesn't exist", "not found", "can't find",
                "invalid", "incorrect", "wrong", "failed",
                "module not found", "import error"
            ]

        patterns = []

        for turn in trace.turns:
            reasoning = turn.reasoning.lower() if turn.reasoning else ""
            output = turn.tool_output.lower() if turn.tool_output else ""

            for keyword in error_keywords:
                if keyword.lower() in reasoning or keyword.lower() in output:
                    patterns.append({
                        "turn": turn.turn_number,
                        "keyword": keyword,
                        "source": "reasoning" if keyword.lower() in reasoning else "tool_output",
                        "context": turn.reasoning[:150] if keyword.lower() in reasoning else turn.tool_output[:150]
                    })

        return patterns

    def extract_code_snippets(self, trace: AgentTrace) -> List[Dict[str, Any]]:
        """
        Extract code snippets from trace

        Args:
            trace: Agent execution trace

        Returns:
            List of code snippets with metadata
        """
        snippets = []

        # Look for code in tool inputs (especially run_python)
        for turn in trace.turns:
            if turn.tool == "run_python" and turn.tool_input:
                code = turn.tool_input.get("code", "")
                if code:
                    snippets.append({
                        "turn": turn.turn_number,
                        "code": code,
                        "output": turn.tool_output
                    })

        # Look for code in final output
        if trace.final_output:
            # Simple heuristic: look for Python keywords
            if any(keyword in trace.final_output for keyword in ["def ", "import ", "class "]):
                snippets.append({
                    "turn": "final",
                    "code": trace.final_output,
                    "output": None
                })

        return snippets

    def analyze_correction_attempts(
        self,
        trace: AgentTrace
    ) -> Dict[str, Any]:
        """
        Analyze agent's correction attempts

        Args:
            trace: Agent execution trace

        Returns:
            Analysis of correction patterns
        """
        analysis = {
            "num_retries": 0,
            "tools_retried": [],
            "reasoning_changes": []
        }

        # Track tool retries
        tool_usage = {}
        for turn in trace.turns:
            if turn.tool:
                if turn.tool in tool_usage:
                    analysis["num_retries"] += 1
                    if turn.tool not in analysis["tools_retried"]:
                        analysis["tools_retried"].append(turn.tool)
                else:
                    tool_usage[turn.tool] = turn.turn_number

        # Detect reasoning changes (simplified)
        reasoning_topics = []
        for turn in trace.turns:
            if turn.reasoning:
                # Very simple topic detection based on first few words
                words = turn.reasoning.split()[:10]
                reasoning_topics.append(words)

        # If reasoning changes significantly, might indicate correction
        if len(reasoning_topics) > 1:
            analysis["reasoning_changes"] = len(reasoning_topics)

        return analysis


class TraceFormatter:
    """
    Formats traces for display and analysis
    """

    @staticmethod
    def format_trace(trace: AgentTrace, include_details: bool = True) -> str:
        """
        Format trace as human-readable text

        Args:
            trace: Agent trace
            include_details: Whether to include full details

        Returns:
            Formatted trace string
        """
        lines = []
        lines.append("="*80)
        lines.append(f"AGENT TRACE: {trace.task_id}")
        lines.append(f"Agent: {trace.agent_id}")
        lines.append(f"Turns: {len(trace.turns)}")
        lines.append("="*80)
        lines.append("")

        for turn in trace.turns:
            lines.append(f"--- Turn {turn.turn_number} ---")

            if turn.reasoning:
                lines.append(f"Reasoning: {turn.reasoning}")

            if turn.action:
                lines.append(f"Action: {turn.action}")

            if turn.tool:
                lines.append(f"Tool: {turn.tool}")
                if include_details and turn.tool_input:
                    lines.append(f"Input: {turn.tool_input}")
                if include_details and turn.tool_output:
                    output = turn.tool_output[:200] + "..." if len(turn.tool_output) > 200 else turn.tool_output
                    lines.append(f"Output: {output}")

            lines.append("")

        if trace.final_output:
            lines.append("--- Final Output ---")
            lines.append(trace.final_output)
            lines.append("")

        lines.append("="*80)

        return "\n".join(lines)


if __name__ == "__main__":
    print("TraceAnalyzer module loaded")
    print("\nExample usage:")
    print("  analyzer = TraceAnalyzer()")
    print("  patterns = analyzer.find_explicit_mentions(trace, ['error', 'not found'])")
    print("  verification = analyzer.detect_verification_behavior(trace, ['search', 'check'])")
