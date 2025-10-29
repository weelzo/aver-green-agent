"""
A2A Protocol Client

Implements Agent-to-Agent (A2A) protocol for communication with purple agents.
Based on AgentBeats/A2A standard.
"""

import httpx
import asyncio
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from .models import AgentTrace, AgentTurn


class A2AMessage:
    """
    A2A protocol message

    Follows the Agent-to-Agent protocol specification.
    """

    def __init__(
        self,
        role: str,
        content: str,
        context_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize A2A message

        Args:
            role: Role of the message sender
            content: Message content
            context_id: Conversation context ID
            parent_id: Parent message ID
            metadata: Additional metadata
        """
        self.role = role
        self.content = content
        self.context_id = context_id or str(uuid.uuid4())
        self.parent_id = parent_id
        self.message_id = str(uuid.uuid4())
        self.timestamp = datetime.now().isoformat()
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "role": self.role,
            "content": self.content,
            "context_id": self.context_id,
            "parent_id": self.parent_id,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        """Create message from dictionary"""
        msg = cls(
            role=data["role"],
            content=data["content"],
            context_id=data.get("context_id"),
            parent_id=data.get("parent_id"),
            metadata=data.get("metadata", {})
        )
        msg.message_id = data.get("message_id", msg.message_id)
        msg.timestamp = data.get("timestamp", msg.timestamp)
        return msg


class A2AClient:
    """
    A2A Protocol Client

    Communicates with purple agents using A2A protocol.
    """

    def __init__(
        self,
        agent_url: str,
        timeout: int = 120,
        max_retries: int = 3
    ):
        """
        Initialize A2A client

        Args:
            agent_url: URL of the purple agent
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.agent_url = agent_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = httpx.AsyncClient(timeout=timeout)

    async def send_message(
        self,
        message: A2AMessage,
        stream: bool = False
    ) -> A2AMessage:
        """
        Send message to purple agent

        Args:
            message: Message to send
            stream: Whether to stream response

        Returns:
            Response message from agent
        """
        payload = message.to_dict()

        for attempt in range(self.max_retries):
            try:
                response = await self.client.post(
                    f"{self.agent_url}/message",
                    json=payload
                )

                response.raise_for_status()
                response_data = response.json()

                return A2AMessage.from_dict(response_data)

            except httpx.HTTPStatusError as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"HTTP error after {self.max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

            except httpx.RequestError as e:
                if attempt == self.max_retries - 1:
                    raise Exception(f"Request error after {self.max_retries} attempts: {e}")
                await asyncio.sleep(2 ** attempt)

    async def start_conversation(
        self,
        initial_content: str,
        role: str = "user"
    ) -> A2AMessage:
        """
        Start a new conversation

        Args:
            initial_content: Initial message content
            role: Role of the sender

        Returns:
            Response from agent
        """
        message = A2AMessage(
            role=role,
            content=initial_content
        )

        return await self.send_message(message)

    async def continue_conversation(
        self,
        content: str,
        context_id: str,
        parent_id: str,
        role: str = "user"
    ) -> A2AMessage:
        """
        Continue existing conversation

        Args:
            content: Message content
            context_id: Conversation context ID
            parent_id: Previous message ID
            role: Role of sender

        Returns:
            Response from agent
        """
        message = A2AMessage(
            role=role,
            content=content,
            context_id=context_id,
            parent_id=parent_id
        )

        return await self.send_message(message)

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class A2ATraceCollector:
    """
    Collects execution traces from A2A conversations

    Converts A2A message exchanges into AgentTrace format.
    """

    def __init__(self):
        """Initialize trace collector"""
        self.messages: List[A2AMessage] = []
        self.current_turn = 0

    def add_message(self, message: A2AMessage):
        """Add message to trace"""
        self.messages.append(message)

    def extract_turn(self, message: A2AMessage) -> AgentTurn:
        """
        Extract AgentTurn from A2A message

        Args:
            message: A2A message

        Returns:
            AgentTurn
        """
        self.current_turn += 1

        # Parse message content for reasoning, action, tool usage
        content = message.content
        metadata = message.metadata

        # Extract structured data if available
        reasoning = metadata.get("reasoning", "")
        action = metadata.get("action", "")
        tool = metadata.get("tool")
        tool_input = metadata.get("tool_input")
        tool_output = metadata.get("tool_output")

        # If no structured metadata, try to parse from content
        if not reasoning and not action:
            # Simple heuristic: first paragraph is reasoning
            parts = content.split("\n\n")
            if len(parts) > 0:
                reasoning = parts[0]
            if len(parts) > 1:
                action = parts[1]

        return AgentTurn(
            turn_number=self.current_turn,
            reasoning=reasoning or content,
            action=action,
            tool=tool,
            tool_input=tool_input,
            tool_output=tool_output,
            timestamp=message.timestamp
        )

    def to_agent_trace(
        self,
        task_id: str,
        agent_id: str,
        final_output: str = "",
        model_name: str = ""
    ) -> AgentTrace:
        """
        Convert collected messages to AgentTrace

        Args:
            task_id: Task identifier
            agent_id: Agent identifier
            final_output: Final output from agent

        Returns:
            AgentTrace
        """
        trace = AgentTrace(
            task_id=task_id,
            agent_id=agent_id,
            final_output=final_output,
            model_name=model_name
        )

        # Convert agent messages to turns (skip system/user messages)
        for message in self.messages:
            if message.role == "assistant" or message.role == "agent":
                turn = self.extract_turn(message)
                trace.add_turn(turn)

        return trace


class A2ATaskExecutor:
    """
    Executes AVER tasks using A2A protocol

    Coordinates between AVER green agent and purple agent via A2A.
    """

    def __init__(self, client: A2AClient):
        """
        Initialize task executor

        Args:
            client: A2A client for communication
        """
        self.client = client
        self.trace_collector = A2ATraceCollector()

    async def execute_task(
        self,
        task_description: str,
        tools: List[Dict[str, Any]],
        max_turns: int = 10
    ) -> AgentTrace:
        """
        Execute task with purple agent

        Args:
            task_description: Task to execute
            tools: Available tools
            max_turns: Maximum conversation turns

        Returns:
            AgentTrace with execution history
        """
        # Build initial message
        initial_message = self._build_task_message(task_description, tools)

        # Start conversation
        response = await self.client.start_conversation(initial_message)
        self.trace_collector.add_message(response)

        context_id = response.context_id
        parent_id = response.message_id

        # Continue conversation until task complete or max turns
        for turn in range(max_turns):
            # Check if agent indicates completion
            if self._is_complete(response):
                break

            # Check if agent requests tool execution
            tool_request = self._extract_tool_request(response)
            if tool_request:
                # Execute tool and send result
                tool_result = await self._execute_tool(tool_request)
                response = await self.client.continue_conversation(
                    content=tool_result,
                    context_id=context_id,
                    parent_id=parent_id,
                    role="system"
                )
                self.trace_collector.add_message(response)
                parent_id = response.message_id
            else:
                # No more interaction needed
                break

        # Extract final output
        final_output = self._extract_final_output(response)

        # Convert to trace
        trace = self.trace_collector.to_agent_trace(
            task_id="task_001",  # Will be set by caller
            agent_id="purple_agent",
            final_output=final_output
        )

        return trace

    def _build_task_message(
        self,
        task_description: str,
        tools: List[Dict[str, Any]]
    ) -> str:
        """
        Build task message for agent

        Args:
            task_description: Task description
            tools: Available tools

        Returns:
            Formatted message
        """
        message = f"{task_description}\n\n"
        message += "Available tools:\n"

        for tool in tools:
            message += f"- {tool['name']}: {tool['description']}\n"

        return message

    def _is_complete(self, message: A2AMessage) -> bool:
        """Check if agent indicates task completion"""
        content = message.content.lower()
        indicators = ["complete", "done", "finished", "final answer"]
        return any(ind in content for ind in indicators)

    def _extract_tool_request(self, message: A2AMessage) -> Optional[Dict[str, Any]]:
        """Extract tool request from agent message"""
        # Check metadata first
        if "tool_request" in message.metadata:
            return message.metadata["tool_request"]

        # Simple heuristic: look for tool usage patterns in content
        # Real implementation would use structured format
        return None

    async def _execute_tool(self, tool_request: Dict[str, Any]) -> str:
        """Execute tool and return result"""
        # Placeholder - would integrate with ToolExecutor
        tool_name = tool_request.get("tool")
        return f"Tool {tool_name} executed successfully"

    def _extract_final_output(self, message: A2AMessage) -> str:
        """Extract final output from agent response"""
        if "final_output" in message.metadata:
            return message.metadata["final_output"]
        return message.content


# Example usage
async def example_a2a_usage():
    """Example of using A2A client"""
    print("A2A Client Example")
    print("="*80)

    # Initialize client
    client = A2AClient(agent_url="http://localhost:8000")

    try:
        # Start conversation
        response = await client.start_conversation(
            "Write a function to calculate fibonacci numbers"
        )

        print(f"Context ID: {response.context_id}")
        print(f"Response: {response.content[:100]}...")

        # Continue conversation
        response2 = await client.continue_conversation(
            content="Make it iterative, not recursive",
            context_id=response.context_id,
            parent_id=response.message_id
        )

        print(f"Follow-up response: {response2.content[:100]}...")

    finally:
        await client.close()


if __name__ == "__main__":
    print("A2A Protocol Client Module")
    print("\nThis module provides:")
    print("  - A2AMessage: Protocol message structure")
    print("  - A2AClient: HTTP client for agent communication")
    print("  - A2ATraceCollector: Convert A2A messages to traces")
    print("  - A2ATaskExecutor: Execute tasks via A2A protocol")
    print("\nExample:")
    print("  client = A2AClient('http://localhost:8000')")
    print("  response = await client.start_conversation('Hello')")
