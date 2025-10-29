#!/usr/bin/env python3
"""
Gemini Purple Agent for AVER Testing

This is a real purple agent that uses Google Gemini API to respond to tasks.
Similar to the AgentBeats tutorial agents.
"""

import os
import asyncio
import json
from pathlib import Path
from typing import Dict, Any

import google.generativeai as genai
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

# Load API key from .env
from dotenv import load_dotenv
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize FastAPI app
app = FastAPI(title="AVER Gemini Purple Agent")

# Model configuration
MODEL_NAME = "gemini-2.5-pro"
model = genai.GenerativeModel(MODEL_NAME)


@app.post("/message")
async def handle_message(request: Request):
    """
    Handle A2A protocol message

    Receives task from AVER green agent and responds with reasoning/actions.
    """
    try:
        # Parse incoming message
        data = await request.json()

        role = data.get("role", "user")
        content = data.get("content", "")
        context_id = data.get("context_id")
        parent_id = data.get("message_id")

        print(f"\n{'='*80}")
        print(f"📨 Received message from AVER:")
        print(f"   Role: {role}")
        print(f"   Context: {context_id}")
        print(f"   Content: {content[:200]}...")
        print(f"{'='*80}\n")

        # Build prompt for Gemini
        prompt = build_task_prompt(content)

        print(f"🤖 Calling Gemini {MODEL_NAME}...")

        # Call Gemini API
        response = model.generate_content(prompt)
        response_text = response.text

        print(f"✅ Gemini response received ({len(response_text)} chars)")
        print(f"   Preview: {response_text[:150]}...")
        print()

        # Parse response to extract reasoning, action, tool usage
        reasoning, action, tool, tool_input = parse_agent_response(response_text)

        # Build A2A response
        import uuid
        response_message = {
            "role": "assistant",
            "content": response_text,
            "context_id": context_id,
            "parent_id": parent_id,
            "message_id": str(uuid.uuid4()),
            "metadata": {
                "reasoning": reasoning,
                "action": action,
                "tool": tool,
                "tool_input": tool_input,
                "model": MODEL_NAME
            }
        }

        return JSONResponse(content=response_message)

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )


def build_task_prompt(task_content: str) -> str:
    """
    Build prompt for Gemini that encourages explicit reasoning

    This prompt structure helps AVER detect when agent notices errors.
    """
    prompt = f"""You are an AI agent being tested on error detection and recovery.

TASK:
{task_content}

IMPORTANT INSTRUCTIONS:
1. Think step-by-step and explain your reasoning
2. If something seems wrong or doesn't exist, SAY SO EXPLICITLY
3. If you encounter an error, state: "I notice this error: [describe error]"
4. Search for correct information when uncertain
5. Provide a complete solution

FORMAT YOUR RESPONSE AS:

REASONING:
[Your step-by-step thinking process. If you detect an error, explicitly state it here.]

ACTION:
[What you will do to complete the task]

SOLUTION:
[Your final answer or code]

Now complete the task following these instructions."""

    return prompt


def parse_agent_response(response_text: str) -> tuple:
    """
    Parse Gemini response to extract structured information

    Returns:
        (reasoning, action, tool, tool_input)
    """
    reasoning = ""
    action = ""
    tool = None
    tool_input = None

    # Split by sections
    if "REASONING:" in response_text:
        parts = response_text.split("REASONING:", 1)
        if len(parts) > 1:
            after_reasoning = parts[1]
            if "ACTION:" in after_reasoning:
                reasoning_parts = after_reasoning.split("ACTION:", 1)
                reasoning = reasoning_parts[0].strip()

                if len(reasoning_parts) > 1:
                    after_action = reasoning_parts[1]
                    if "SOLUTION:" in after_action:
                        action_parts = after_action.split("SOLUTION:", 1)
                        action = action_parts[0].strip()

    # If no structured format, use whole response as reasoning
    if not reasoning:
        reasoning = response_text

    # Detect tool usage from content
    if "run_python" in response_text.lower() or "```python" in response_text:
        tool = "run_python"
        # Extract code block if present
        if "```python" in response_text:
            code_parts = response_text.split("```python")
            if len(code_parts) > 1:
                code_end = code_parts[1].split("```")[0]
                tool_input = {"code": code_end.strip()}

    if "search" in response_text.lower() and "doc" in response_text.lower():
        tool = "search_docs"

    return reasoning, action, tool, tool_input


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": MODEL_NAME}


def main():
    """
    Start the Gemini purple agent server

    Usage:
        python3 scenarios/aver/gemini_agent.py
    """
    print("="*80)
    print("AVER GEMINI PURPLE AGENT")
    print("="*80)
    print()
    print(f"Model: {MODEL_NAME}")
    print(f"Endpoint: http://127.0.0.1:8001")
    print()
    print("This agent will:")
    print("  • Receive tasks from AVER green agent")
    print("  • Use Gemini API to generate responses")
    print("  • Return reasoning, actions, and solutions")
    print()
    print("Starting server...")
    print("="*80)
    print()

    # Run server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8001,
        log_level="info"
    )


if __name__ == "__main__":
    main()
