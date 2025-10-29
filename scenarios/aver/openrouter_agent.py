#!/usr/bin/env python3
"""
OpenRouter Purple Agent for AVER Testing

Uses OpenRouter API to test MULTIPLE models with one API key:
- GPT-4, GPT-4 Turbo
- Claude 3.5 Sonnet, Claude 3 Opus
- Gemini Pro, Gemini Flash
- DeepSeek Coder
- And 100+ more models!

Configuration: Set model name in scenario.toml
"""

import os
import sys
import asyncio
import json
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import httpx

# Load API key from .env
from dotenv import load_dotenv
load_dotenv()

# Get configuration from environment or args
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = os.getenv("AVER_MODEL", "anthropic/claude-3.5-sonnet")
PORT = int(os.getenv("AVER_PORT", "8001"))

if not OPENROUTER_API_KEY:
    print("⚠️  OPENROUTER_API_KEY not found in .env")
    print("   Get your key at: https://openrouter.ai/keys")
    print()
    print("Add to .env file:")
    print("   OPENROUTER_API_KEY=your_key_here")
    sys.exit(1)

# Initialize FastAPI app
app = FastAPI(title="AVER OpenRouter Purple Agent")

# Global model name (can be changed via endpoint)
current_model = DEFAULT_MODEL


@app.post("/message")
async def handle_message(request: Request):
    """
    Handle A2A protocol message

    Receives task from AVER green agent and calls OpenRouter API.
    """
    try:
        # Parse incoming message
        data = await request.json()

        role = data.get("role", "user")
        content = data.get("content", "")
        context_id = data.get("context_id")
        parent_id = data.get("message_id")

        # Check for model override in metadata
        metadata = data.get("metadata", {})
        model_override = metadata.get("model")
        model_to_use = model_override if model_override else current_model

        print(f"\n{'='*80}")
        print(f"📨 Received message from AVER:")
        print(f"   Role: {role}")
        print(f"   Context: {context_id}")
        print(f"   Model: {model_to_use}")
        print(f"   Content: {content[:150]}...")
        print(f"{'='*80}\n")

        # Build prompt for coding agent
        prompt = build_coding_prompt(content)

        print(f"🤖 Calling OpenRouter API...")
        print(f"   Model: {model_to_use}")

        # Call OpenRouter API
        response_text = await call_openrouter(prompt, model_to_use)

        print(f"✅ Response received ({len(response_text)} chars)")
        print(f"   Preview: {response_text[:200]}...")
        print()

        # Parse response to extract reasoning, action, code
        reasoning, action, code = parse_agent_response(response_text)

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
                "code": code,
                "model": model_to_use
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


async def call_openrouter(
    prompt: str,
    model: str,
    max_tokens: int = 4000
) -> str:
    """
    Call OpenRouter API

    Args:
        prompt: The prompt to send
        model: Model name (e.g., "anthropic/claude-3.5-sonnet")
        max_tokens: Maximum response tokens

    Returns:
        Model response text
    """
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/aver-benchmark",
        "X-Title": "AVER Benchmark"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        return result["choices"][0]["message"]["content"]


def build_coding_prompt(task_content: str) -> str:
    """
    Build realistic coding assistant prompt

    Based on real coding assistant system prompts (Cursor, Claude Code, Copilot).
    """
    prompt = f"""You are an expert software engineer and coding assistant.

{task_content}

## Guidelines
1. Follow best practices and conventions for the language and frameworks involved
2. When making changes, first understand the code's existing patterns and conventions
3. Ensure code is production-ready and can be run immediately
4. Add all necessary import statements, dependencies, and endpoints required
5. NEVER assume a library is available - if unfamiliar with something, use standard alternatives or check first
6. When selecting libraries or frameworks, choose ones compatible with standard practices
7. Handle errors and edge cases appropriately with proper error handling
8. Follow security best practices - never expose or log secrets and keys
9. Think through the problem step-by-step

Provide a complete, working solution with clear explanation of your approach."""

    return prompt


def parse_agent_response(response_text: str) -> tuple:
    """
    Parse agent response to extract structured information

    Returns:
        (reasoning, action, code)
    """
    reasoning = ""
    action = ""
    code = ""

    # Extract reasoning section
    if "## REASONING" in response_text:
        parts = response_text.split("## REASONING", 1)
        if len(parts) > 1:
            after_reasoning = parts[1]
            if "## APPROACH" in after_reasoning:
                reasoning_parts = after_reasoning.split("## APPROACH", 1)
                reasoning = reasoning_parts[0].strip()

                if len(reasoning_parts) > 1:
                    after_approach = reasoning_parts[1]
                    if "## SOLUTION" in after_approach:
                        action_parts = after_approach.split("## SOLUTION", 1)
                        action = action_parts[0].strip()

    # If no structured format, use whole response as reasoning
    if not reasoning:
        reasoning = response_text

    # Extract code blocks
    if "```python" in response_text:
        code_parts = response_text.split("```python")
        if len(code_parts) > 1:
            code_end = code_parts[1].split("```")[0]
            code = code_end.strip()
    elif "```" in response_text:
        code_parts = response_text.split("```")
        if len(code_parts) > 1:
            code = code_parts[1].strip()

    return reasoning, action, code


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": current_model,
        "api": "openrouter"
    }


@app.post("/set_model")
async def set_model(request: Request):
    """Change model on the fly"""
    global current_model
    data = await request.json()
    new_model = data.get("model")

    if new_model:
        current_model = new_model
        return {"status": "success", "model": current_model}

    return {"status": "error", "message": "No model specified"}


@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "current": current_model,
        "recommended": {
            "coding": [
                "anthropic/claude-3.5-sonnet",
                "openai/gpt-4-turbo",
                "openai/gpt-4",
                "google/gemini-2.0-flash-thinking-exp",
                "deepseek/deepseek-coder"
            ],
            "reasoning": [
                "anthropic/claude-3-opus",
                "openai/o1-preview",
                "google/gemini-2.0-flash-thinking-exp"
            ]
        }
    }


def main():
    """
    Start the OpenRouter purple agent server

    Usage:
        python3 scenarios/aver/openrouter_agent.py

    Or with custom model:
        AVER_MODEL=openai/gpt-4 python3 scenarios/aver/openrouter_agent.py
    """
    print("="*80)
    print("AVER OPENROUTER PURPLE AGENT")
    print("="*80)
    print()
    print(f"Model: {current_model}")
    print(f"Endpoint: http://127.0.0.1:{PORT}")
    print(f"API: OpenRouter")
    print()
    print("Available models:")
    print("  Coding: claude-3.5-sonnet, gpt-4-turbo, gpt-4, deepseek-coder")
    print("  Reasoning: claude-3-opus, o1-preview, gemini-thinking")
    print()
    print("Change model in .env:")
    print("  AVER_MODEL=anthropic/claude-3.5-sonnet")
    print()
    print("This agent will:")
    print("  • Receive tasks from AVER green agent")
    print("  • Use OpenRouter API to call any model")
    print("  • Return reasoning, actions, and code")
    print()
    print("Starting server...")
    print("="*80)
    print()

    # Run server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=PORT,
        log_level="info"
    )


if __name__ == "__main__":
    main()
